## How to Run

Run via `python3 .\worker.py --model_path {model_path}` Model path should be the path to the GGUF file you want to run or a directory containing multiple GGUF files and JSON configuration files.
`--first_model {model_slug}` is optional and specifies the first model to run. If not specified, the alphabetically first model in the directory will be run. If specified, the model must be in the directory. The `model_slug` is the name of the model file without the extension. For example, if the model is named `model.gguf`, the `model_slug` is `model`.

## Additional Arguments
- `--n_gpu_layers {int}` is optional and specifies the number of layers to run on the GPU. Default is 99.
- `--flash_attn {bool}` is optional and specifies whether to use the flash attention mechanism. Default is True.
- `--n_ctx {int}` is optional and specifies the context size. Default is 8192.
- `--host {str}` is optional and specifies the host to run the server on. Default is: 0.0.0.0
- `--port {int}` is optional and specifies the port to run the server on. Default is: 8000
- `--root_path {str}` is optional and specifies the root path for the server. Default is "".
- `--openapi_url {str}` is optional and specifies the URL for the OpenAPI JSON file. Default is "/openapi.json".
- `--docs_url {str}` is optional and specifies the URL for the Swagger UI. Default is "/docs".
- `--redoc_url {str}` is optional and specifies the URL for the ReDoc UI. Default is "/redoc".

## Model Configuration File
The model configuration file is a JSON file that is used to set the model's priority and paramters. The priority is used when using the `auto` model selection mode. The parameters are used to set the model's parameters. The parameters are the same as the command line arguments. The model configuration file should be named the same as the model with the extension `.json`. For example, if the model is named `model.gguf`, the configuration file should be named `model.json`.

An example that I personally use for auto model selection is as follows:

Vision Model Example:
```json
{
    "vision": {
        "enabled": true,
        "encoder_model": "./vision_encoder/Bunny-V-mmproj-model-f16.gguf"
    },
    "priority": 1
}
```

Long Context Model Example:
```json
{
    "n_ctx":131072,
    "priority": 8
}
```

Smart Short Context Model Example:
```json
{
    "priority": 10
}
```

This setup will switch to BunnyV when images are sent with the request, switch to a long context model when the context length of the request exceeds 8192 tokens, and switch to a smart short context model when the context length of the request is less than 8192 tokens. In my experience, this setup has worked well for a variety of use cases.

Originally I created this to use over the default llama-cpp-python OpenAI server because that was slower than this on my old hardware. I have added a bunch of stuff I personally wanted to it and now I just use it out of preference. I'm releasing it here for transparency for exactly how I run my models for those who are interested. I'm not sure if it will be useful to anyone else, but I hope it is.