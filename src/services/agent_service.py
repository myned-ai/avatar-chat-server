"""
Agent Service

Provides agent instances based on configuration.
Supports local sample agents and remote agents.
"""

from typing import Optional

from agents import BaseAgent, SampleOpenAIAgent, SampleGeminiAgent, RemoteAgent
from config import Settings, get_settings

# Global agent instance
_agent: Optional[BaseAgent] = None


def get_agent(settings: Optional[Settings] = None) -> BaseAgent:
    """
    Get or create the agent instance based on configuration.

    Args:
        settings: Application settings (uses defaults if not provided)

    Returns:
        Agent instance
    """
    global _agent

    if _agent is None:
        settings = settings or get_settings()
        agent_type = settings.agent_type

        if agent_type == "sample_openai":
            _agent = SampleOpenAIAgent(settings)
        elif agent_type == "sample_gemini":
            _agent = SampleGeminiAgent(settings)
        elif agent_type == "remote":
            _agent = RemoteAgent(settings)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}. Supported: sample_openai, sample_gemini, remote")

    return _agent