import json


def serialize(self) -> str:
    """Serialize a StructuredResponse to JSON string."""
    data = {
        "class_name": self.class_name,
        "definition": self.definition,
        "instance": self.instance,
        "native": self.native
    }
    return json.dumps(data)
