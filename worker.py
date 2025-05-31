from message_formatter import MessageFormatter, PromptStyle, Message
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, Iterator, Union
import uvicorn
from sse_starlette.sse import EventSourceResponse
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from functools import partial
import anyio
import llama_cpp
from llama_cpp import Llama
from llama_cpp.llava_cpp import llava_eval_image_embed, llava_image_embed_make_with_bytes, clip_model_load, llava_image_embed_free, clip_free
import json
import argparse
import os
import time
import numpy as np
from PIL import Image
import ctypes
from paddleocr import PaddleOCR
import array
import urllib.request
import base64
import io
import asyncio

default_formatter = MessageFormatter()
loaded_formatter = None
ocr = PaddleOCR(use_angle_cls=True, lang="en")


def url_to_base_64(url: str) -> str: # Converts a URL to a data URL base64 string
    return "data:image/png;base64," + base64.b64encode(urllib.request.urlopen(url).read()).decode("utf-8")

def PIL_image_to_base64(image: Image) -> str:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    if not image_base64.startswith("data:image/png;base64,"):
        image_base64 = "data:image/png;base64," + image_base64
    return image_base64

def image_url_to_bytes(image_url: str) -> bytes: # Loads an image from a URL or base64 data url and returns the bytes
    print("Image URL to Bytes:",image_url)
    if image_url.startswith("data:"):
        image_bytes = base64.b64decode(image_url.split(",")[1])
        return image_bytes
    else:
        with urllib.request.urlopen(image_url) as f:
            image_bytes = f.read()
            return image_bytes

def url_to_PIL_image(image_url: str) -> Image: # Loads an image from a URL or base64 string and returns a PIL image
    image_bytes = image_url_to_bytes(image_url)
    image = Image.open(io.BytesIO(image_bytes))
    return image

def bytes_to_PIL_image(image_bytes: bytes) -> Image: # Converts bytes to a PIL image
    return Image.open(io.BytesIO(image_bytes))
        
def get_image_embed_from_bytes(image_bytes:bytes, n_threads=1):
    global clip_model
    data_array = array.array("B", image_bytes)
    c_ubyte_ptr = (
        ctypes.c_ubyte * len(data_array)
    ).from_buffer(data_array)
    print("Creating Image Embed")
    print("Clip Model:",clip_model)
    print("N Threads:",n_threads)
    print("Image Bytes Length:",len(image_bytes))
    print("C Ubyte Ptr:",c_ubyte_ptr)
    embed = (
        llava_image_embed_make_with_bytes(
            clip_model,
            n_threads,
            c_ubyte_ptr,
            len(image_bytes),
        )
    )
    return embed

def get_image_embed_from_url(url):
    image_bytes = image_url_to_bytes(url)
    return get_image_embed_from_bytes(image_bytes)

def get_image_embed_from_file(path):
    with open(path, "rb") as f:
        image_bytes = f.read()
    return get_image_embed_from_bytes(image_bytes)

def get_image_embed_from_PIL(image):
    print("PIL Image convert to bytes")
    image_bytes = image.tobytes()
    print("Image Bytes Length:",len(image_bytes))
    return get_image_embed_from_bytes(image_bytes)

def eval_image_embed(embed):
    global llama
    try:
        n_past = ctypes.c_int(llama.n_tokens)
        n_past_p = ctypes.pointer(n_past)
        
        llava_eval_image_embed(
            llama.ctx,
            embed,
            llama.n_batch,
            n_past_p,
        )
        assert llama.n_ctx() >= n_past.value
        llama.n_tokens = n_past.value
    except Exception as e:
        print(e)
        print("Failed to eval image")
    finally:
        llava_image_embed_free(embed)
        
def multimodal_eval(text, embeds): # -> prompt
    """Evaluates a multimodal prompt with text and image embeds. The text is split by "{image}" and the image embeds are inserted in the order they appear in the text. The text is then tokenized and the image embeds are evaluated in the model. The input_ids are then returned."""
    global llama
    assert len(embeds) > 0
    assert type(text) == str
    assert type(embeds) == list
    
    text_chunks = text.split("{image}")
    assert len(text_chunks) == len(embeds) + 1
    text_chunks = [chunk.encode("utf8") for chunk in text_chunks]

    llama.reset() # Reset the model
    # clear the input_ids
    llama.input_ids = np.ndarray((llama.n_ctx(),), dtype=np.intc)
    llama._ctx.kv_cache_clear()
    # print(text_chunks)
    for i, chunk in enumerate(text_chunks):
        llama.eval(llama.tokenize(chunk, add_bos=True if i == 0 else False)) # Tokenize the text chunk and evaluate it
        if i < len(embeds): # If there is an image embed to evaluate
            eval_image_embed(embeds[i]) # Evaluate the image embed
    return llama.input_ids[: llama.n_tokens].tolist()

# def get_player_perspective(paddle_ocr=False,resize_image=False,ocr_resolution=128,image_resolution=128):
#     frame = Image.fromarray(frame)

#     if paddle_ocr:
#     else:
#         ascii_block = ""

#     frame = frame.convert("RGB")
#     # base64_image = image_to_base64(frame)
#     buffered = io.BytesIO() # Don't ask me why this is needed - it just is for some reason.
#     frame.save(buffered, format="PNG")
#     return get_image_embed_from_bytes(buffered.getvalue()), ascii_block

def multimodal_prompt_format(prompt, imgs=[],ocr_resolution=256,image_resolution=1152,resize_image=False):
    image_replacements = prompt.count("{image}")
    if image_replacements != len(imgs):
        raise ValueError(f"Number of images ({len(imgs)}) does not match number of image placeholder strings in prompt ({image_replacements})")
    embeds = []
    ascii_blocks = []
    print("Prompt:",prompt)
    print("Embedding Images:",imgs)
    for img in imgs:
        print("Image Size:",img.size)
        if resize_image:
            img = img.resize((image_resolution, image_resolution))
            print("Resized Image Size:",img.size)
        img = img.convert("RGB")
        # base64_image = image_to_base64(frame)
        buffered = io.BytesIO() # Don't ask me why this is needed - it just is for some reason.
        img.save(buffered, format="PNG")
        image_embed = get_image_embed_from_bytes(buffered.getvalue())
        result = ocr.ocr(np.array(img), cls=True)
        ascii_block = get_ascii_block(result, img, ocr_resolution)
        embeds.append(image_embed)
        ascii_blocks.append(ascii_block)
    prompt = multimodal_eval(prompt, embeds)
    return prompt, ascii_blocks

def get_ascii_block(paddle_result, img, ascii_representation_max_size = 128):
    image_width, image_height = img.size
    ascii_representation_size = (0,0)
    if image_width > image_height:
        ascii_representation_size = (ascii_representation_max_size, int(ascii_representation_max_size * (image_height / image_width)))
    else:
        ascii_representation_size = (int(ascii_representation_max_size * (image_width / image_height)), ascii_representation_max_size)
    # ascii_representation = "#" * (ascii_representation_size[0]+2) + "\n"
    ascii_representation = ""

    print("ASCII Size:",ascii_representation_size)
    paddle_result = paddle_result[0]
    if paddle_result == None or len(paddle_result) == 0:
        return ""
    boxes = [line[0] for line in paddle_result]
    txts = [line[1][0] for line in paddle_result]
    _scores = [line[1][1] for line in paddle_result]
    true_area = 0
    # blank ascii_representation
    for i in range(ascii_representation_size[1]):
        blank_line = " " * ascii_representation_size[0] + "\n" # "#" + 
        true_area += len(blank_line)
        ascii_representation += blank_line
    theoretical_ascii_area = ascii_representation_size[0] * ascii_representation_size[1]
    print("Theoretical ASCII Area:",theoretical_ascii_area)
    print("True ASCII Area:",true_area)
    # write to ascii_representation
    ocr_filter = [] # list of bad strings to filter out
    for i in range(len(boxes)):
        print("Box:",boxes[i],txts[i])
        point_1 = boxes[i][0]
        point_2 = boxes[i][1]
        point_3 = boxes[i][2]
        point_4 = boxes[i][3]
        text = txts[i]
        filtered = False
        for bad_string in ocr_filter:
            if bad_string in text or text == "" or text.strip() == "" or bad_string.lower() in text.lower():
                filtered = True
                break
        if filtered:
            continue
        centered_x = int((point_1[0] + point_2[0] + point_3[0] + point_4[0]) / 4)
        centered_y = int((point_1[1] + point_2[1] + point_3[1] + point_4[1]) / 4)
        centered_point = (centered_x, centered_y)
        print("Centered Point:",centered_point)
        centered_x = int((centered_x / image_width) * ascii_representation_size[0])
        centered_y = int((centered_y / image_height) * ascii_representation_size[1])
        centered_point = (centered_x, centered_y)
        print("Centered Point:",centered_point)
        # overwrite ascii_representation to include text centered at centered_point offset by half the length of text
        text_length = len(text)
        text_start = centered_x - int(text_length / 2)
        text_end = text_start + text_length
        
        ascii_lines = ascii_representation.split("\n")
        ascii_lines[centered_y] = ascii_lines[centered_y][:text_start] + text + ascii_lines[centered_y][text_end:]
        ascii_representation = "\n".join(ascii_lines)

    new_ascii_representation = ""
    for line in ascii_representation.split("\n"):
        if line.strip() != "":
            new_ascii_representation += line + "\n"
    ascii_representation = new_ascii_representation
    # ascii_representation += "#" * (ascii_representation_size[0]+2)
    print("---BLOCK TOP")
    print(ascii_representation)
    print("---BLOCK BOTTOM")
    return ascii_representation
    
async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Iterator,
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                # print("chunk", chunk)
                await inner_send_chan.send(dict(data=json.dumps(chunk)))
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            print("disconnected")
            with anyio.move_on_after(1, shield=True):
                print(f"Disconnected from client (via refresh/close) {request.client}")
                raise e

def get_images_from_objects(objects):
    images = []
    for obj in objects:
        if obj["type"] == "image":
            try:
                if len(obj["base64"].split(",")) > 1:
                    images.append(bytes_to_PIL_image(base64.b64decode(obj["base64"].split(",")[1])))
                else:
                    images.append(bytes_to_PIL_image(base64.b64decode(obj["base64"])))
            except Exception as e:
                images.append(bytes_to_PIL_image(base64.b64decode(obj["base64"])))
        elif obj["type"] == "image_url":
            if "url" in obj:
                images.append(bytes_to_PIL_image(image_url_to_bytes(obj["url"])))
            elif "image_url" in obj:
                if "url" in obj["image_url"]:
                    images.append(bytes_to_PIL_image(image_url_to_bytes(obj["image_url"]["url"])))
                else:
                    images.append(bytes_to_PIL_image(image_url_to_bytes(obj["image_url"])))
    return images

class ResponseFormat(BaseModel):
    type: str
    json_schema: Union[dict, str]

class ChatCompletionsRequest(BaseModel):
    messages: list[Message]
    ocr_resolution: Optional[int] = 256
    image_resolution: Optional[int] = 1152
    resize_image: Optional[bool] = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 40
    min_p: Optional[float] = 0.05
    repeat_penalty: Optional[float] = 1.1
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[list[str]] = []
    stops: Optional[list[str]] = []
    stream: Optional[bool] = False
    response_type: Optional[str] = "assistant"
    prompt_style: Optional[PromptStyle] = None
    model: Optional[str] = "auto"
    priority: Optional[int] = 99
    grammar: Optional[Union[dict, str]] = None
    response_format: Optional[ResponseFormat] = None
    transform: Optional[list[str]] = ["middle-out"]
    prefill: Optional[str] = None

class CompletionsRequest(BaseModel):
    prompt: str
    images: Optional[list[str]] = [] # List of image URLs/base64 strings
    ocr_resolution: Optional[int] = 256
    image_resolution: Optional[int] = 1152
    resize_image: Optional[bool] = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 40
    min_p: Optional[float] = 0.05
    repeat_penalty: Optional[float] = 1.1
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stops: Optional[list[str]] = []
    stream: Optional[bool] = False
    model: Optional[str] = "auto"
    priority: Optional[int] = 99
    grammar: Optional[Union[dict, str]] = None
    response_format: Optional[ResponseFormat] = None

class EmbeddingsRequest(BaseModel):
    strings: list[str]
    model: Optional[str] = None

class TokenizeRequest(BaseModel):
    input: str
    model: Optional[str] = None

class DetokenizeRequest(BaseModel):
    input: list[int]
    model: Optional[str] = None

class ModelData(BaseModel):
    id: str
    object: str
    owned_by: str
    permissions: list[str]
    input_modalities: list[str]
    output_modalities: list[str]
    n_ctx: int
    priority: int
    currently_loaded: bool = False
    prompt_style: Optional[PromptStyle] = None

multiple_models = False
models_path = "./models/"
models = []
last_model = ""
llama = None
clip_model = None
processing = False
llama_options = {
    "n_gpu_layers": 99,
    "flash_attn": True,
    "n_ctx": 8192,
    "vision": {
        "enabled": False,
        "multi_image_enabled": False,
        "encoder_model": "",
        "ocr_resolution": 256,
        "image_resolution": 1152,
    },
    "priority": 0,
}

def get_model_options(model):
    global llama_options
    new_options = {}
    for key, value in llama_options.items():
        if type(value) == tuple: # wtf??
            value = value[0]
        new_options[key] = value
    # print("Default options:", new_options)
    if os.path.exists(models_path+model+".json"):
        print("Loading model options from", models_path+model+".json")
        for key, value in json.load(open(models_path+model+".json")).items():
            new_options[key] = value
    return new_options

def get_models_data():
    output = []
    auto_config = ModelData(
        id="auto",
        object="model",
        owned_by="",
        permissions=[],
        input_modalities=["text"],
        output_modalities=["text"],
        n_ctx=8192,
        priority=99,
        currently_loaded=True
    )
    for model in models:
        options = get_model_options(model)
        modalities = []
        modalities.append("text")
        if options["vision"]["enabled"]:
            modalities.append("image")
        model_data = ModelData(
            id=model,
            object="model",
            owned_by="",
            permissions=[],
            input_modalities=modalities,
            output_modalities=["text"], # Only text output for now, looking at Mini-Omni as a proof of concept for multimodal outputs -- Could also add comfyui support for text to image?
            n_ctx=options["n_ctx"],
            priority=options["priority"],
            currently_loaded=model == last_model
        )
        if options["n_ctx"] > auto_config.n_ctx:
            auto_config.n_ctx = options["n_ctx"]
        for modality in modalities:
            if modality not in auto_config.input_modalities:
                auto_config.input_modalities.append(modality)
        output.append(model_data)
    return output, auto_config

def get_all_model_options():
    global models
    options = {}
    for model in models:
        options[model] = get_model_options(model)
    return options

model_lock = asyncio.Lock()

async def load_model(model: str,input_modalities: list[str] = ["text"],output_modalities: list[str] = ["text"],priority: int = 99, token_length=8192):
    global llama
    global clip_model
    global llama_options
    global last_model
    global loaded_formatter
    async with model_lock:
        models_data, auto_config = get_models_data()
        print("Loading model", model, "with input modalities", input_modalities, "and output modalities", output_modalities, "and priority", priority, "and token length", token_length)
        if model == "auto":
            print("Models data:", models_data)
            # print("Auto config:", auto_config)
            best_model = None
            best_distance = 999999
            options = 0
            for model_data in models_data:
                print("Checking model", model_data)
                # check that it has all the required input and output modalities
                if not (all(modality in model_data.input_modalities for modality in input_modalities) and all(modality in model_data.output_modalities for modality in output_modalities)):
                    print("Model", model_data.id, "does not have the required input and output modalities")
                    continue
                # check that the model can fit the token length
                if model_data.n_ctx <= token_length:
                    continue
                priority_distance = abs(model_data.priority - priority)
                print("Model", model_data.id, "has priority distance", priority_distance)
                if priority_distance < best_distance:
                    options += 1
                    best_model = model_data
                    best_distance = priority_distance
            if best_model == None:
                raise ValueError(f"No model found with input modalities {input_modalities} and output modalities {output_modalities} with {token_length} tokens")
            model = best_model.id
            print(f"Auto model selected: {model} from {options} available options")
        
        # time.sleep(3)
        if model not in models:
            raise ValueError(f"Model '{model}' not found!")
        options = get_model_options(model)
        if "prompt_style" in options and options["prompt_style"] != None:
            prompt_style = PromptStyle(**options["prompt_style"])
            loaded_formatter = MessageFormatter(prompt_style=prompt_style)
            print("Loaded formatter:",loaded_formatter)
        else:
            print("No prompt style found")
            loaded_formatter = None
        # time.sleep(3)
        if model != last_model:
            print("Changing model from", last_model, "to", model)
            # time.sleep(3)
            last_model = model
            abs_model_path = models_path + model + ".gguf"
            abs_model_path = os.path.abspath(abs_model_path)
            print("Checking for model options at", models_path+model+".json")
            if clip_model != None:
                clip_free(clip_model)
            clip_model = None

            if llama is not None:
                llama.close()
                llama = None
            print("Loading model", model)
            print("Model path", abs_model_path)
            print("Options", options)
            # time.sleep(5)
            try:
                llama = Llama(
                    model_path=abs_model_path,
                    n_gpu_layers=options["n_gpu_layers"],
                    n_ctx=options["n_ctx"]
                )
                if options["vision"]["enabled"]:
                    print("Loading vision encoder", options["vision"]["encoder_model"])
                    clip_model = clip_model_load(options["vision"]["encoder_model"].encode(), 1)
            except Exception as e:
                print(e)
                print("Failed to load model", model)
                input("Press enter to continue...")
            print("Performing initial completions to warm up the model...")
            llama.create_completion("This is a test of the model", max_tokens=25)
            llama.create_completion("This is another test of the model", max_tokens=25)
            print("Model", model, "loaded")
        else:
            if token_length > options["n_ctx"]:
                raise ValueError(f"Token length {token_length} is greater than the model's context size {options['n_ctx']}")
            print("Model", model, "already loaded")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--first_model", type=str, required=False)
    args.add_argument("--n_gpu_layers", type=int, required=False, default=99)
    args.add_argument("--flash_attn", type=bool, required=False, default=True)
    args.add_argument("--n_ctx", type=int, required=False, default=8192)
    args.add_argument("--host", type=str, required=False, default=None)
    args.add_argument("--port", type=int, required=False, default=8000)
    args.add_argument("--root_path", type=str, required=False, default="")
    args.add_argument("--openapi_url", type=str, required=False, default="/openapi.json")
    args.add_argument("--docs_url", type=str, required=False, default="/docs")
    args.add_argument("--redoc_url", type=str, required=False, default="/redoc")

    kwargs = vars(args.parse_args())

    if kwargs["model_path"].endswith(".gguf"):
        models = [kwargs["model_path"]]
    else:
        multiple_models = True
        models = os.listdir(kwargs["model_path"])
        models_path = kwargs["model_path"]
        models = [model for model in models if model.endswith(".gguf")]
    models = [model.split("/")[-1].split("\\")[-1].replace(".gguf", "") for model in models]

    if "n_gpu_layers" in kwargs:
        llama_options["n_gpu_layers"] = kwargs["n_gpu_layers"],
    if "flash_attn" in kwargs:
        llama_options["flash_attn"] = kwargs["flash_attn"],
    if "n_ctx" in kwargs:
        llama_options["n_ctx"] = kwargs["n_ctx"]

    if not multiple_models:
        # load_model(kwargs["model_path"].split("/")[-1].split("\\")[-1].replace(".gguf", ""))
        asyncio.run(load_model(kwargs["model_path"].split("/")[-1].split("\\")[-1].replace(".gguf", "")))
    else:
        if kwargs["first_model"] in models:
            # load_model(kwargs["first_model"])
            asyncio.run(load_model(kwargs["first_model"]))
        else:
            # load_model(models[0])
            asyncio.run(load_model(models[0]))
    api = FastAPI(title="🦙 llama-cpp-python Customized API", root_path=kwargs["root_path"], openapi_url=kwargs["openapi_url"], docs_url=kwargs["docs_url"], redoc_url=kwargs["redoc_url"])
    from fastapi.middleware.cors import CORSMiddleware

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
@api.post("/v1/completions", tags=["OpenAI V1"])
@api.post("/v1/engines/copilot-codex/completions",tags=["Copilot Codex V1"])
async def completions(request: Request, body: CompletionsRequest) -> llama_cpp.Completion:
    global default_formatter
    global loaded_formatter
    global llama
    global llama_options
    global last_model
    if loaded_formatter:
        formatter = loaded_formatter
    else:
        formatter = default_formatter
    prompt = body.prompt
    images = body.images
    ocr_resolution = body.ocr_resolution
    image_resolution = body.image_resolution
    resize_image = body.resize_image
    input_modalities = ["text"]
    token_length = len(llama.tokenize(prompt.encode("utf-8")))
    if len(images) > 0:
        input_modalities.append("image")
    
    print("Body:",body)
    print("Requested Model:",body.model)

    if body.model and body.model != "" and (body.model in models or body.model == "auto") and body.model != last_model:
        print("Loading model", body.model)
        await load_model(body.model, input_modalities=input_modalities, priority=body.priority, token_length=token_length) # Load the model if it's not already loaded
    new_images = []
    for image in images:
        image_bytes = image_url_to_bytes(image) # Load the image from the URL/base64 string and get the bytes
        image = bytes_to_PIL_image(image_bytes) # Convert the bytes to a PIL image
        new_images.append(image)
    images = new_images
    if len(images) > 0:
        prompt, ascii_blocks = multimodal_prompt_format(prompt, images, ocr_resolution, image_resolution, resize_image)
        ocr_count = prompt.count("{ocr}")
        if ocr_count != len(ascii_blocks):
            print(f"Error: Number of OCR blocks ({len(ascii_blocks)}) does not match number of OCR placeholder strings in prompt ({ocr_count})")
        for i in range(ocr_count):
            prompt = prompt.replace("{ocr}", ascii_blocks[i], 1) # Replace the first instance of "{ocr}" with the first OCR block
    
    grammar = None
    if body.grammar and type(body.grammar) == dict: # original format 1
        print("Grammar:",body.grammar)
        grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(body.grammar))
    elif body.grammar and type(body.grammar) == str: # original format 2
        print("Grammar:",body.grammar)
        grammar = llama_cpp.LlamaGrammar.from_string(body.grammar)
    elif body.response_format and type(body.response_format) != None:
        print("Response format:",body.response_format)
        if body.response_format.type == "json_schema":
            if type(body.response_format.json_schema) == dict:
                grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(body.response_format.json_schema))
            elif type(body.response_format.json_schema) == str:
                grammar = llama_cpp.LlamaGrammar.from_string(body.response_format.json_schema)
            else:
                print("Error: Grammar is not a valid type")
    else:
        print("No grammar found")
    print("Grammar:",grammar)
    async with model_lock:
        iterator_or_completion = llama.create_completion(
            prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            min_p=body.min_p,
            repeat_penalty=body.repeat_penalty,
            frequency_penalty=body.frequency_penalty,
            presence_penalty=body.presence_penalty,
            stop=formatter.stop + body.stops,
            stream=body.stream,
            grammar=grammar
        )
        if isinstance(iterator_or_completion, Iterator):
            # EAFP: It's easier to ask for forgiveness than permission
            first_response = next(iterator_or_completion)
            # If no exception was raised from first_response, we can assume that
            # the iterator is valid and we can use it to stream the response.
            def iterator() -> Iterator[llama_cpp.CreateCompletionStreamResponse]:
                yield first_response
                yield from iterator_or_completion
            send_chan, recv_chan = anyio.create_memory_object_stream(10)
            return EventSourceResponse(
                recv_chan,
                data_sender_callable=partial(  # type: ignore
                    get_event_publisher,
                    request=request,
                    inner_send_chan=send_chan,
                    iterator=iterator(),
                ),
                sep="\n",
                ping_message_factory=None,
            )
        else:
            return iterator_or_completion

@api.post("/v1/embeddings", tags=["OpenAI V1"])
async def embeddings(body: EmbeddingsRequest) -> list[float]:
    global llama
    global llama_options
    global last_model
    max_length = 0
    for string in body.strings:
        token_length = len(llama.tokenize(string.encode("utf-8"))) # Tokenize with currently loaded model to figure out if we need to load a new model
        if token_length > max_length:
            max_length = token_length
    print("Requested Model:",body.model)
    if body.model and body.model != "" and (body.model in models or body.model == "auto") and body.model != last_model:
        print("Loading model", body.model)
        await load_model(body.model,token_length=token_length) # Load the model if it's not already loaded
    return llama.create_embedding(body.strings)

@api.post("/v1/chat/completions", tags=["OpenAI V1"])
async def chat_completions(
    request: Request,
    body: ChatCompletionsRequest):
    global formatter
    global llama
    global llama_options
    global last_model
    global default_formatter
    global loaded_formatter
    ocr_resolution = body.ocr_resolution
    image_resolution = body.image_resolution
    resize_image = body.resize_image
    modalities = ["text"]
    if loaded_formatter:
        formatter = loaded_formatter
        print(f"Loaded formatter for model '{last_model}':",formatter)
    else:
        formatter = default_formatter
        print("Default formatter:",formatter)
    if body.prompt_style:
        print("Body contained prompt style:",body.prompt_style)
        formatter = MessageFormatter(prompt_style=body.prompt_style)
    prompt, images_objects = formatter.get_string_from_messages(body.messages)
    # print("Prompt:",prompt)
    # print("Images:",images_objects)
    if len(images_objects) > 0:
        modalities.append("image")
    token_length = len(llama.tokenize(prompt.encode("utf-8"))) # Tokenize with currently loaded model to figure out if we need to load a new model
    print("Requested Model:",body.model)
    if body.model and body.model != "" and (body.model in models or body.model == "auto") and body.model != last_model:
        print("Loading model", body.model)
        await load_model(body.model,input_modalities=modalities,priority=body.priority,token_length=token_length) # Load the model if it's not already loaded
    model_options = get_model_options(last_model)
    response_type = body.response_type
    if response_type == "user" and "prompt_style" in model_options and model_options["prompt_style"] != None and model_options["prompt_style"]["user_name"] != None:
        response_type = model_options["prompt_style"]["user_name"]
    elif response_type == "assistant" and "prompt_style" in model_options and model_options["prompt_style"] != None and model_options["prompt_style"]["assistant_name"] != None:
        response_type = model_options["prompt_style"]["assistant_name"]
    elif response_type == "system" and "prompt_style" in model_options and model_options["prompt_style"] != None and model_options["prompt_style"]["system_name"] != None:
        response_type = model_options["prompt_style"]["system_name"]
    prompt += formatter.start_message(response_type)
    if body.prefill and body.prefill.strip() != "":
        prompt += body.prefill
    images = get_images_from_objects(images_objects)
    # new_images = []
    # for image in images:
    #     image_bytes = image_url_to_bytes(image) # Load the image from the URL/base64 string and get the bytes
    #     image = bytes_to_PIL_image(image_bytes) # Convert the bytes to a PIL image
    #     new_images.append(image)
    # images = new_images
    if len(images) > 0:
        prompt, ascii_blocks = multimodal_prompt_format(prompt, images, ocr_resolution, image_resolution, resize_image)
        ocr_count = prompt.count("{ocr}")
        if ocr_count != len(ascii_blocks):
            print(f"Error: Number of OCR blocks ({len(ascii_blocks)}) does not match number of OCR placeholder strings in prompt ({ocr_count})")
        for i in range(ocr_count):
            prompt = prompt.replace("{ocr}", ascii_blocks[i], 1) # Replace the first instance of "{ocr}" with the first OCR block
    if body.grammar and type(body.grammar) == dict:
        grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(body.grammar))
    elif body.grammar and type(body.grammar) == str:
        grammar = llama_cpp.LlamaGrammar.from_string(body.grammar)
    elif body.response_format and type(body.response_format) != None:
        print("Response format:",body.response_format)
        if type(body.response_format.json_schema) == dict:
            grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(body.response_format.json_schema))
        elif type(body.response_format.json_schema) == str:
            grammar = llama_cpp.LlamaGrammar.from_string(body.response_format.json_schema)
        else:
            print("Error: Grammar is not a valid type")
        # if body.response_format.type == "json_schema":
        #     grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(body.response_format.json_schema))
    else:
        grammar = None
    print("Grammar:",grammar)
    
    stops = formatter.stop + body.stop + body.stops
    stops = list(set(stops))
    if "prompt_style" in model_options and model_options["prompt_style"] != None and model_options["prompt_style"]["stop"] != None:
        stops += model_options["prompt_style"]["stop"]
    print("Stops:",stops)
    keyword_args = dict(
        prompt=prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        min_p=body.min_p,
        repeat_penalty=body.repeat_penalty,
        frequency_penalty=body.frequency_penalty,
        presence_penalty=body.presence_penalty,
        stop=formatter.stop + body.stops,
        stream=body.stream,
        grammar=grammar,
    )
    start_time = time.time()
    first_token_time = None

    async with model_lock:
        iterator_or_completion: Union[llama_cpp.ChatCompletion, Iterator[llama_cpp.ChatCompletionChunk]] = await run_in_threadpool(llama.create_completion, **keyword_args)
        if isinstance(iterator_or_completion, Iterator):
            def chat_wrapper(iterator: Iterator[llama_cpp.ChatCompletionChunk]):
                for chunk in iterator:
                    # print(chunk)
                    chat_completion = {
                        "id": chunk["id"],
                        "object": "chat.completion",
                        "created": chunk["created"],
                        "model": body.model,
                        "choices": [],
                        "usage": {
                            "time_taken": time.time() - start_time,
                            "time_to_first_token": first_token_time - start_time if first_token_time else "N/A",
                        }
                    }
                    # print(chunk)
                    for choice in chunk["choices"]:
                        chat_completion["choices"].append({
                            "index": choice["index"],
                            "message": {
                                "content": choice["text"],
                                "role": formatter.assistant_name,
                                "name": ""
                            },
                            "logprobs": choice["logprobs"],
                            "finish_reason": choice["finish_reason"],
                        })
                    yield chat_completion
            iterator_or_completion = chat_wrapper(iterator_or_completion)
            # EAFP: It's easier to ask for forgiveness than permission
            first_response = await run_in_threadpool(next, iterator_or_completion)
            first_token_time = time.time()

            print(iterator_or_completion)
            # If no exception was raised from first_response, we can assume that
            # the iterator is valid and we can use it to stream the response.
            def iterator() -> Iterator[llama_cpp.CreateCompletionStreamResponse]:
                yield first_response
                yield from iterator_or_completion

            send_chan, recv_chan = anyio.create_memory_object_stream(10)
            return EventSourceResponse(
                recv_chan,
                data_sender_callable=partial(  # type: ignore
                    get_event_publisher,
                    request=request,
                    inner_send_chan=send_chan,
                    iterator=iterator(),
                ),
                sep="\n",
                ping_message_factory=None,
            )
        else:
            chat_completion = {
                "id": iterator_or_completion["id"],
                "object": "chat.completion",
                "created": iterator_or_completion["created"],
                "model": body.model,
                "choices": [],
                "usage": {
                    "prompt_tokens": iterator_or_completion["usage"]["prompt_tokens"],
                    "completion_tokens": iterator_or_completion["usage"]["completion_tokens"],
                    "total_tokens": iterator_or_completion["usage"]["total_tokens"],
                    "time_taken": time.time() - start_time,
                    "time_to_first_token": first_token_time - start_time if first_token_time else "N/A",
                }
            }
            for choice in iterator_or_completion["choices"]:
                chat_completion["choices"].append({
                    "index": choice["index"],
                    "message": {
                        "content": choice["text"],
                        "role": response_type,
                    },
                    "logprobs": choice["logprobs"],
                    "finish_reason": choice["finish_reason"],
                })
            return chat_completion

@api.get("/v1/models", tags=["OpenAI V1"])
def get_models() -> list[ModelData]:
    # return [{
    #     "id": model,
    #     "object": "model",
    #     "owned_by": "",
    #     "permissions": [],
    #     "currently_loaded": model == last_model
    # } for model in models]
    models, auto_config = get_models_data()
    models.append(auto_config)
    return models

@api.post("/extras/tokenize", tags=["Extras"])
async def tokenize(body: TokenizeRequest) -> list[int]:
    global llama
    global last_model
    global llama_options
    token_length = len(llama.tokenize(body.input.encode("utf-8"))) # Tokenize with currently loaded model to figure out if we need to load a new model
    if body.model and body.model != "" and body.model in models and body.model != last_model:
        await load_model(body.model,token_length=token_length)
    return llama.tokenize(body.input)

@api.post("/extras/tokenize/count", tags=["Extras"])
async def count_tokens(body: TokenizeRequest) -> int:
    global llama
    global last_model
    global llama_options
    token_length = len(llama.tokenize(body.input.encode("utf-8"))) # Tokenize with currently loaded model to figure out if we need to load a new model
    if body.model and body.model != "" and body.model in models and body.model != last_model:
        await load_model(body.model,token_length=token_length)
    string_bytes = body.input.encode("utf-8")
    return len(llama.tokenize(string_bytes, add_bos=False))

@api.post("/extras/detokenize", tags=["Extras"])
async def detokenize(body: DetokenizeRequest) -> str:
    global llama
    global llama_options
    global last_model
    token_length = len(llama.tokenize(body.input.encode("utf-8"))) # Tokenize with currently loaded model to figure out if we need to load a new model
    if body.model and body.model != "" and body.model in models and body.model != last_model:
        await load_model(body.model,token_length=token_length)
    return llama.detokenize(body.input)

@api.post("/extras/chat/get_prompt", tags=["Extras"])
def get_prompt(body: CompletionsRequest) -> str:
    return formatter.get_string_from_messages(body.prompt)

@api.get("/heartbeat", tags=["Extras"])
def heartbeat() -> dict:
    return {"status": "ok"}

if __name__ == "__main__":
    while True:
        try:
            uvicorn.run(api, host=kwargs["host"], port=kwargs["port"])
        except Exception as e:
            print(e)
            time.sleep(2)
            continue