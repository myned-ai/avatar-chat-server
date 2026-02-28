"""
Definitions for Actions (Tools) available to the AI Agents.
These tools allow the AI to trigger functionality on the client's host website.
"""

# Gemini Function Declaration format using plain dictionaries for Live API config
GEMINI_NYX_ACTIONS = {
    "function_declarations": [
        {
            "name": "trigger_confetti",
            "description": "Trigger a celebratory confetti blast on the user's screen. Use this when the user asks for confetti, is celebrating a success, or asks for something fun."
        }
    ]
}

# OpenAI Realtime format
OPENAI_NYX_ACTIONS = [
    {
        "type": "function",
        "name": "trigger_confetti",
        "description": "Trigger a celebratory confetti blast on the user's screen. Use this when the user asks for confetti, is celebrating a success, or asks for something fun.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]
