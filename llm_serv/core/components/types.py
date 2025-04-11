from enum import Enum

class LLMRequestType(str, Enum):
    LLM = "LLM"
    OCR = "OCR"
    IMAGE = "IMAGE"
