"""
Definitions for Actions (Tools) available to the AI Agents.
These tools allow the AI to trigger functionality on the client's host website.

Best Practice (Gemini Live): Each tool includes an **Invocation Condition:** block
to tell the model precisely when to invoke it.
"""

# Gemini Function Declaration format using plain dictionaries for Live API config
GEMINI_NYX_ACTIONS = {
    "function_declarations": [
        {
            "name": "trigger_confetti",
            "description": (
                "Trigger a celebratory confetti blast on the user's screen.\n"
                "**Invocation Condition:** Invoke this tool when the user explicitly asks for confetti, "
                "is celebrating a success or achievement, or asks for something fun or festive."
            )
        },
        {
            "name": "navigate_to_section",
            "description": (
                "Scroll the user's page to a specific named section.\n"
                "**Invocation Condition:** Invoke this tool when the user asks to navigate to, see, "
                "or go to a specific section of the page. Use section IDs from the initialization context "
                "(e.g. 'pricing', 'features', 'contact_us'). If no sections were provided, "
                "you may attempt common IDs like 'top', 'bottom', 'contact'."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "section_id": {
                        "type": "STRING",
                        "description": "The section to navigate to (e.g. 'pricing', 'features', 'contact_us')"
                    }
                },
                "required": ["section_id"]
            }
        },
        {
            "name": "send_rich_content",
            "description": (
                "Send a rich interactive element (like a link card, data table, product carousel, or media) "
                "directly into the user's chat widget to enhance the conversation visually.\n"
                "**Invocation Condition:** Invoke this tool when you are describing a product, presenting "
                "structured data, sharing a link, or any situation where a visual element would enhance "
                "comprehension. Call this tool concurrently as you begin speaking about the content."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "item_id": {
                        "type": "STRING",
                        "description": "A unique identifier for this content block. If an item_id already exists, it will be updated."
                    },
                    "content_type": {
                        "type": "STRING",
                        "description": "The type of content to render (e.g., 'link_card', 'table', 'form'). Ensure the client widget has a renderer registered for this type."
                    },
                    "action": {
                        "type": "STRING",
                        "description": "The action to perform on the client. Typically 'replace' (upsert), 'append', or 'remove'."
                    },
                    "payload_json": {
                        "type": "STRING",
                        "description": "A fully serialized JSON string containing the data for the rich element. Must be valid JSON."
                    }
                },
                "required": ["item_id", "content_type", "payload_json"]
            }
        },
        {
            "name": "request_screen_context",
            "description": (
                "Request a screenshot of the user's current screen/viewport.\n"
                "**Invocation Condition:** Invoke this tool *immediately* when the user asks about something "
                "visually on their screen, such as 'What am I looking at?', 'Help me fill this form', "
                "or 'What does this page show?'. You MUST invoke this tool without ANY conversational filler. "
                "Do NOT say 'Let me look' or 'Sure'. Remain unmistakably silent and yield your turn. "
                "Only answer after you receive the image attachment."
            )
        }
    ]
}

# OpenAI Realtime format
OPENAI_NYX_ACTIONS = [
    {
        "type": "function",
        "name": "trigger_confetti",
        "description": (
            "Trigger a celebratory confetti blast on the user's screen.\n"
            "**Invocation Condition:** Invoke this tool when the user explicitly asks for confetti, "
            "is celebrating a success or achievement, or asks for something fun or festive."
        ),
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "type": "function",
        "name": "navigate_to_section",
        "description": (
            "Scroll the user's page to a specific named section.\n"
            "**Invocation Condition:** Invoke this tool when the user asks to navigate to, see, "
            "or go to a specific section of the page. Use section IDs from the initialization context "
            "(e.g. 'pricing', 'features', 'contact_us'). If no sections were provided, "
            "you may attempt common IDs like 'top', 'bottom', 'contact'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "section_id": {"type": "string", "description": "The section to navigate to"}
            },
            "required": ["section_id"]
        }
    },
    {
        "type": "function",
        "name": "send_rich_content",
        "description": (
            "Send a rich interactive element directly into the user's chat widget. "
            "Use this for complex data, link cards, forms, or media.\n"
            "**Invocation Condition:** Invoke this tool when describing a product, presenting "
            "structured data, sharing a link, or any situation where a visual element enhances comprehension. "
            "Call concurrently as you begin speaking. Ensure the payload is a valid JSON string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string", "description": "Unique ID for this content block"},
                "content_type": {"type": "string", "description": "Content type (e.g. 'link_card', 'table', 'form')"},
                "action": {"type": "string", "enum": ["replace", "append", "remove"]},
                "payload_json": {"type": "string", "description": "Serialized JSON string with the element data"}
            },
            "required": ["item_id", "content_type", "payload_json"]
        }
    },
    {
        "type": "function",
        "name": "request_screen_context",
        "description": (
            "Request a screenshot of the user's current screen/viewport.\n"
            "**Invocation Condition:** Invoke this tool *immediately* when the user asks about something "
            "visually on their screen, such as 'What am I looking at?', 'Help me fill this form', "
            "or 'What does this page show?'. You MUST invoke this tool without ANY conversational filler. "
            "Do NOT say 'Let me look' or 'Sure'. Remain unmistakably silent and yield your turn. "
            "Only answer after you receive the image attachment."
        ),
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]
