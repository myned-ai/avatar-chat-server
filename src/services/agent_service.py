"""
Agent Service

Provides agent instances based on configuration.
Supports local sample agents and remote agents.
"""

from agents import BaseAgent, RemoteAgent, SampleGeminiAgent, SampleOpenAIAgent
from core.config import get_settings

# Global agent instance
_agent: BaseAgent | None = None


def get_agent() -> BaseAgent:
    """
    Get or create the agent instance based on configuration.

    Returns:
        Agent instance
    """
    global _agent

    if _agent is None:
        settings = get_settings()
        agent_type = settings.agent_type

        if agent_type == "sample_openai":
            _agent = SampleOpenAIAgent()
        elif agent_type == "sample_gemini":
            _agent = SampleGeminiAgent()
        elif agent_type == "remote":
            _agent = RemoteAgent()
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}. Supported: sample_openai, sample_gemini, remote")

    return _agent
