

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any, Union
class SeparatorStyle(Enum):
    """Different separator style."""
    LLAMA3 = auto()
    QWEN2= auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.QWEN2
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    tokenizer_id: str = ""
    tokenizer: Any = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None
    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.LLAMA3:
            ret = ""
            if self.system:
                ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.QWEN2:
            ret = "" if self.system == "" else self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images, _ = message
                        message = "<image>" * len(images) + message
                    ret += role + message + self.sep
                else:
                    ret += role
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            tokenizer_id=self.tokenizer_id,
            tokenizer=self.tokenizer,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            skip_next=self.skip_next,
            )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_qwen2 = Conversation(
    system="",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.QWEN2,
    sep="<|im_end|>\n",
)


llama_3_chat = Conversation(
    # system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
    #        "You are able to understand the visual content that the user provides, "
    #        "and assist the user with a variety of tasks using natural language.",
    system = "",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n",
           "<|start_header_id|>system<|end_header_id|>\n\n"),
    version="llama_3_chat",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA3,
    sep="<|eot_id|>",
    # sep="<|end_of_text|>",
)


default_conversation = conv_qwen2
conv_templates = {
    "qwen2": conv_qwen2,
    "llama3":llama_3_chat
}