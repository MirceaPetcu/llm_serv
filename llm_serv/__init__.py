from llm_serv.api import LLMService
from llm_serv.client import LLMServiceClient
from llm_serv.core.base import LLMRequest, LLMResponse
from llm_serv.core.base import LLMProvider
from llm_serv.conversation import Conversation, Message, Role, Image, Document
import importlib.metadata
import pathlib
import re
import os


try:
    pkg_dir = pathlib.Path(__file__).parent.parent.absolute()
    pyproject_path = pkg_dir / "pyproject.toml"
    
    if pyproject_path.exists():
        with open(pyproject_path, "r") as f:
            content = f.read()
            version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if version_match:
                __version__ = version_match.group(1)
            else:
                __version__ = "0.0.0"  # Fallback
    else:
        __version__ = "0.0.0"  # Fallback
except Exception:
    __version__ = "0.0.0"  # Fallback if anything goes wrong

# This ensures LLMService is initialized when the package is imported
_ = LLMService.list_models()

__all__ = ["LLMService", "LLMServiceClient", "LLMRequest", "LLMResponse", "LLMProvider", 
           "Conversation", "Message", "Role", "Image", "Document", "__version__"]