"""
Agent Service

Provides agent instances based on configuration.
Supports local sample agents and remote agents.
"""

from agents import BaseAgent, RemoteAgent, SampleGeminiAgent, SampleOpenAIAgent
from core.settings import get_settings


def create_agent_instance() -> BaseAgent:
    """
    Create a new agent instance based on configuration.
    Used for creating a fresh agent for each session.

    Returns:
        New Agent instance
    """
    settings = get_settings()
    agent_type = settings.agent_type

    if agent_type == "sample_openai":
        return SampleOpenAIAgent()
    elif agent_type == "sample_gemini":
        return SampleGeminiAgent()
    elif agent_type == "remote":
        return RemoteAgent()
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Supported: sample_openai, sample_gemini, remote")
