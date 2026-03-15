"""
Domain-specific actions (Tools) for custom use cases.
These tools extend the core AI capabilities for specific use cases.
"""

GEMINI_CUSTOM_ACTIONS = [
    {
        "name": "rotate_3d_model",
        "description": (
            "Rotate the 3D product model in the viewer to show a specific angle.\n"
            "**Invocation Condition:** Invoke this tool when the user asks to see the back, side, "
            "top, or bottom of the product, or explicitly asks to rotate it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "direction": {
                    "type": "STRING",
                    "enum": ["left", "right", "up", "down", "front", "back"],
                    "description": "The direction to rotate the 3D model."
                }
            },
            "required": ["direction"]
        }
    }
]

OPENAI_CUSTOM_ACTIONS = [
    {
        "type": "function",
        "name": "rotate_3d_model",
        "description": (
            "Rotate the 3D product model in the viewer to show a specific angle.\n"
            "**Invocation Condition:** Invoke this tool when the user asks to see the back, side, "
            "top, or bottom of the product, or explicitly asks to rotate it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "front", "back"],
                    "description": "The direction to rotate the 3D model."
                }
            },
            "required": ["direction"]
        }
    }
]
