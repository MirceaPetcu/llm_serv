
    
@staticmethod
def deserialize(json_string: str) -> "StructuredResponse":
    data = json.loads(json_string)
    sr = StructuredResponse(
        class_name=data.get("class_name", "StructuredResponse"),
        definition=data.get("definition"),
        instance=data.get("instance"),
    )
    return sr