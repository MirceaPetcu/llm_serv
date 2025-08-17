from typing import Any
from xml.etree import ElementTree as ET

from llm_serv.structured_response.utils import camel_to_snake, coerce_text_to_type


def from_prompt(self, xml_string: str) -> None:
    """
    Parse an LLM answer string, extract the XML section for the root element, and
    populate the instance dictionary according to the definition schema.
    """
    if not self.definition:
        raise ValueError("Definition not initialized. Call from_basemodel first.")

    root_tag = camel_to_snake(self.class_name)
    # Extract substring between first <root_tag and last </root_tag>
    start_idx = xml_string.find(f"<{root_tag}")
    end_idx = xml_string.rfind(f"</{root_tag}>")
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Root XML tags not found in LLM output")
    # Move end to include closing tag
    end_idx += len(f"</{root_tag}>")
    xml_sub = xml_string[start_idx:end_idx]

    # Ensure the opening tag is well-formed (strip attributes if any are present)
    try:
        root_element = ET.fromstring(xml_sub)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML content: {exc}") from exc

    def parse_element(element: ET.Element, schema: Any) -> Any:
        # schema can be a dict (complex) or a type descriptor for simple
        if isinstance(schema, dict) and "type" in schema and schema["type"] == "list":
            # List parsing from <li> children
            items: list[Any] = []
            for li in element.findall("li"):
                elem_schema = schema["elements"]
                if isinstance(elem_schema, dict):
                    # complex item: children tags represent fields
                    item: dict[str, Any] = {}
                    for child in li:
                        if child.tag == "li":
                            continue
                        if child.tag is ET.Comment:
                            continue
                        field_schema = elem_schema.get(child.tag)
                        if field_schema is None:
                            continue
                        item[child.tag] = parse_element(child, field_schema)
                    items.append(item)
                else:
                    # simple item: li text
                    items.append(coerce_text_to_type(li.text or "", str(elem_schema)))
            return items

        if isinstance(schema, dict) and "type" not in schema:
            # Nested object: parse each child according to schema
            obj: dict[str, Any] = {}
            for field_name, field_schema in schema.items():
                child = element.find(field_name)
                if child is None:
                    obj[field_name] = None
                    continue
                obj[field_name] = parse_element(child, field_schema)
            return obj

        # Simple field
        assert isinstance(schema, dict)
        type_name = schema.get("type", "str")
        # For enum treat as str
        if type_name == "enum":
            return (element.text or "").strip()
        return coerce_text_to_type(element.text or "", type_name)

    self.instance = {}
    for field_name, schema in self.definition.items():
        child = root_element.find(field_name)
        if child is None:
            self.instance[field_name] = None
            continue
        self.instance[field_name] = parse_element(child, schema)
    assert self.instance is not None
