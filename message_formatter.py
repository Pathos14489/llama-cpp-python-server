from pydantic import BaseModel
import json
from typing import Optional
class PromptStyle(BaseModel):
    stop: list[str] = ["<|eot_id|>","<|end_header_id|>"]
    banned_chars: list[str] = ["{", "}", "\"" ]
    end_of_sentence_chars: list[str] = [".", "?", "!"]
    BOS_token: str = "<|start_header_id|>"
    EOS_token: str = "<|eot_id|>"
    message_signifier: str = ": "
    role_seperator: str = "<|end_header_id|>\n\n"
    message_seperator: str = ""
    message_format: str = "[BOS_token][role][role_seperator][name][message_signifier][content][EOS_token][message_seperator]"
    system_name: str = "system"
    user_name: str = "user"
    assistant_name: str = "assistant"
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
        parsed_msg_part = parsed_msg_part.replace("[role]",role)
        parsed_msg_part = parsed_msg_part.replace("[role_seperator]",role_sep)
        parsed_msg_part = parsed_msg_part.replace("[name]",name)
        parsed_msg_part = parsed_msg_part.replace("[message_signifier]",msg_sig)
        parsed_msg_part = parsed_msg_part.replace("[EOS_token]",self.EOS_token)
        parsed_msg_part = parsed_msg_part.replace("[message_seperator]",self.message_seperator)
        parsed_msg_part = parsed_msg_part.split("[content]")[0]
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
        parsed_msg_part = parsed_msg_part.replace("[role]",role)
        parsed_msg_part = parsed_msg_part.replace("[role_seperator]",role_sep)
        parsed_msg_part = parsed_msg_part.replace("[name]",name)
        parsed_msg_part = parsed_msg_part.replace("[message_signifier]",msg_sig)
        parsed_msg_part = parsed_msg_part.replace("[EOS_token]",self.EOS_token)
        parsed_msg_part = parsed_msg_part.replace("[message_seperator]",self.message_seperator)
        parsed_msg_part = parsed_msg_part.split("[content]")[1]
        return parsed_msg_part

    def get_string_from_messages(self, messages): # Returns a formatted string from a list of messages
        """Returns a formatted string from a list of messages"""
        context = ""
        print(f"Creating string from messages: {len(messages)}")
        for message in messages:
            # print(f"Message:",message)
            if "content" in message:
                content = message["content"]
            else:
                try:
                    content = message.content
                except:
                    raise ValueError("Message does not have 'content' key!")
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
        return context
    
    def __str__(self):
        return json.dumps(self._prompt_style.model_dump(), indent=4)