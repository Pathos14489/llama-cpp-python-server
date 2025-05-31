from pydantic import BaseModel
import json
from typing import Optional, Union

class PromptStyle(BaseModel):
    stop: list[str] = ["<|eot_id|>","<|end_header_id|>"]
    banned_chars: list[str] = ["{", "}", "\"" ]
    end_of_sentence_chars: list[str] = [".", "?", "!"]
    BOS_token: str = "<|start_header_id|>"
    EOS_token: str = "<|eot_id|>"
    message_signifier: str = ": "
    role_seperator: str = "<|end_header_id|>\n\n"
    message_seperator: str = ""
    message_format: str = "[BOS_token][role][role_seperator][name][message_signifier][content][specific_EOS_token][EOS_token][message_seperator]"
    system_name: str = "system"
    user_name: str = "user"
    assistant_name: str = "assistant"
    system_EOS_token: str = ""
    user_EOS_token: str = ""
    assistant_EOS_token: str = ""

class MessageTextContent(BaseModel):
    type: str = "text"
    text: Optional[str]

class ImageUrl(BaseModel):
    url: str

class MessageImageURLContent(BaseModel):
    type: str = "image_url"
    url: Optional[str]
    image_url: Optional[ImageUrl]

class MessageImageBase64Content(BaseModel):
    type: str = "image"
    base64: Optional[str]

class Message(BaseModel):
    role: str
    content: Union[str, list[dict]] # Union[MessageTextContent, MessageImageURLContent, MessageImageBase64Content]
    name: Optional[str] = ""

class MessageFormatter(): # Tokenizes(only availble for counting the tokens in a string presently for local_models), and parses and formats messages for use with the language model
    def __init__(self, prompt_style: Optional[PromptStyle] = None): # Initializes the message formatter with a prompt style
        if prompt_style == None:
            ps = PromptStyle()
        else:
            ps = prompt_style
        self._prompt_style = ps
        self.stop = ps.stop
        self.banned_chars = ps.banned_chars
        self.end_of_sentence_chars = ps.end_of_sentence_chars
        self.BOS_token = ps.BOS_token
        self.EOS_token = ps.EOS_token
        self.message_signifier = ps.message_signifier
        self.role_seperator = ps.role_seperator
        self.message_seperator = ps.message_seperator
        self.message_format = ps.message_format
        self.system_name = ps.system_name
        self.user_name = ps.user_name
        self.assistant_name = ps.assistant_name
        self.system_EOS_token = ps.system_EOS_token
        self.user_EOS_token = ps.user_EOS_token
        self.assistant_EOS_token = ps.assistant_EOS_token

    def _version(self): # Returns the version of the message formatter
        """Returns the version of the message formatter"""
        return "0.0.4"
        
    def new_message(self, content, role, name=None): # Parses a string into a message format with the name of the speaker
        """Parses a string into a message format with the name of the speaker"""
        if type(name) == str: # If the name is a string, check if it's empty and set it to None if it is
            if name.strip() == "":
                name = None
        parsed_msg = self.start_message(role, name)
        if content.strip() == "":
            return ""
        parsed_msg += content
        parsed_msg += self.end_message(role, name)
        return parsed_msg

    def start_message(self, role="", name=None): # Returns the start of a message with the name of the speaker
        """Returns the start of a message with the name of the speaker"""
        parsed_msg_part = self.message_format
        msg_sig = self.message_signifier
        if not name:
            name = ""
            msg_sig = ""
        role_sep = self.role_seperator
        if role == "":
            role_sep = ""
            
        if name == "":
            parsed_msg_part = parsed_msg_part.split("[message_signifier]")[0]
        if role == "":
            parsed_msg_part = parsed_msg_part.split("[role_seperator]")[0]
        parsed_msg_part = parsed_msg_part.replace("[BOS_token]",self.BOS_token)
        formatted_roled = ""
        if role == "system":
            formatted_roled = self.system_name
        elif role == "user":
            formatted_roled = self.user_name
        elif role == "assistant":
            formatted_roled = self.assistant_name
        parsed_msg_part = parsed_msg_part.replace("[role]",formatted_roled)
        parsed_msg_part = parsed_msg_part.replace("[role_seperator]",role_sep)
        parsed_msg_part = parsed_msg_part.replace("[name]",name)
        parsed_msg_part = parsed_msg_part.replace("[message_signifier]",msg_sig)
        specific_EOS_token = ""
        if role == "system":
            specific_EOS_token = self.system_EOS_token
        elif role == "user":
            specific_EOS_token = self.user_EOS_token
        elif role == "assistant":
            specific_EOS_token = self.assistant_EOS_token
        parsed_msg_part = parsed_msg_part.replace("[specific_EOS_token]",specific_EOS_token)
        parsed_msg_part = parsed_msg_part.replace("[EOS_token]",self.EOS_token)
        parsed_msg_part = parsed_msg_part.replace("[message_seperator]",self.message_seperator)
        parsed_msg_part = parsed_msg_part.split("[content]")[0]
        print(f"Parsed message first part: {parsed_msg_part}")
        return parsed_msg_part

    def end_message(self, role="", name=None): # Returns the end of a message with the name of the speaker (Incase the message format chosen requires the name be on the end for some reason, but it's optional to include the name in the end message)
        """Returns the end of a message with the name of the speaker (Incase the message format chosen requires the name be on the end for some reason, but it's optional to include the name in the end message)"""
        parsed_msg_part = self.message_format
        msg_sig = self.message_signifier
        if not name:
            name = ""
            msg_sig = ""
        role_sep = self.role_seperator
        if role == "":
            role_sep = ""
        if name == "":
            parsed_msg_part = parsed_msg_part.split("[message_signifier]")[1]
        if role == "":
            parsed_msg_part = parsed_msg_part.split("[role_seperator]")[1]
        parsed_msg_part = parsed_msg_part.replace("[BOS_token]",self.BOS_token)
        formatted_roled = ""
        if role == "system":
            formatted_roled = self.system_name
        elif role == "user":
            formatted_roled = self.user_name
        elif role == "assistant":
            formatted_roled = self.assistant_name
        parsed_msg_part = parsed_msg_part.replace("[role]",formatted_roled)
        parsed_msg_part = parsed_msg_part.replace("[role_seperator]",role_sep)
        parsed_msg_part = parsed_msg_part.replace("[name]",name)
        parsed_msg_part = parsed_msg_part.replace("[message_signifier]",msg_sig)
        specific_EOS_token = ""
        if role == "system":
            specific_EOS_token = self.system_EOS_token
        elif role == "user":
            specific_EOS_token = self.user_EOS_token
        elif role == "assistant":
            specific_EOS_token = self.assistant_EOS_token
        parsed_msg_part = parsed_msg_part.replace("[specific_EOS_token]",specific_EOS_token)
        parsed_msg_part = parsed_msg_part.replace("[EOS_token]",self.EOS_token)
        parsed_msg_part = parsed_msg_part.replace("[message_seperator]",self.message_seperator)
        parsed_msg_part = parsed_msg_part.split("[content]")[1]
        print(f"Parsed message second part: {parsed_msg_part}")
        return parsed_msg_part

    def get_string_from_messages(self, messages: list[Message]): # Returns a formatted string from a list of messages
        """Returns a formatted string from a list of messages"""
        print(f"Using message format: {self.message_format}")
        context = ""
        images = []
        print(f"Creating string from messages: {len(messages)}")
        for message in messages:
            # print(f"Message:",message)
            # if "content" in message:
            #     content = message["content"]
            # else:
            #     try:
            #         content = message.content
            #     except:
            #         raise ValueError("Message does not have 'content' key!")
            if type(message.content) == str:
                content = message.content
            else:
                content = ""
                for content_item in message.content:
                    if content_item["type"] == "text":
                        content += content_item["text"]
                    elif content_item["type"] == "image_url":
                        content += "{image}"
                        images.append(content_item)
                    elif content_item["type"] == "image":
                        content += "{image}"
                        images.append(content_item)
            if "role" in message:
                role = message["role"]
            else:
                try:
                    role = message.role
                except:
                    raise ValueError("Message does not have 'role' key!")
            if "name" in message:
                name = message["name"]
            else:
                try:
                    name = message.name
                except:
                    name = None
            msg_string = self.new_message(content, role, name)
            context += msg_string
        print(f"Context:")
        print(context)
        return context, images
    
    def __str__(self):
        return json.dumps(self._prompt_style.model_dump(), indent=4)