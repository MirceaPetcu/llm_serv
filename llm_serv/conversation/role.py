from enum import Enum


class Role(str, Enum):
    USER:str = "user"
    ASSISTANT:str = "assistant"
    SYSTEM:str = "system"
