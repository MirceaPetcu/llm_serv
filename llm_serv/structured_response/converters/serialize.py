import json


def serialize(self) -> str:
        data = {
            "class_name": self.class_name,
            "definition": self.definition,
            "instance": self.instance,
        }
        return json.dumps(data)
