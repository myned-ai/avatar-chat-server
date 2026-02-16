from pathlib import Path

import httpx

from core.logger import get_logger

logger = get_logger(__name__)


class KnowledgeService:
    """Service to handle loading and formatting knowledge base content."""

    @staticmethod
    async def load_knowledge_base(source: str | None) -> str:
        """
        Loads knowledge base content from a local file or URL.
        Returns an empty string if source is not set or loading fails.
        """
        if not source:
            return ""

        content = ""
        try:
            # Check if source is a URL
            if source.startswith(("http://", "https://")):
                logger.info(f"Fetching knowledge base from URL: {source}")
                async with httpx.AsyncClient() as client:
                    response = await client.get(source, timeout=5.0)
                    response.raise_for_status()
                    content = response.text
            else:
                # Assume local file path
                file_path = Path(source)
                
                # If path doesn't exist, try resolving relative to project root
                if not file_path.exists() and not file_path.is_absolute():
                    # Resolve project root: src/services/knowledge_service.py -> src/services -> src -> root
                    project_root = Path(__file__).parent.parent.parent
                    potential_path = project_root / source
                    if potential_path.exists():
                        logger.info(f"Resolved relative path to: {potential_path}")
                        file_path = potential_path

                if file_path.exists() and file_path.is_file():
                    logger.info(f"Loading knowledge base from file: {file_path}")
                    content = file_path.read_text(encoding="utf-8")
                else:
                    logger.warning(f"Knowledge base file not found: {source} (checked {file_path.absolute()})")
                    return ""

            if content:
                logger.info(f"Loaded {len(content)} characters from knowledge base")
                return content.strip()

        except Exception as e:
            logger.error(f"Failed to load knowledge base from {source}: {e}")

        return ""

    @staticmethod
    def format_instructions(base_instructions: str, knowledge_content: str) -> str:
        """Helper to clearly separate base instructions from knowledge base"""
        if not knowledge_content:
            return base_instructions

        logger.info("Injecting knowledge base content into system instructions")
        return (
            f"{base_instructions}\n\n"
            f"### KNOWLEDGE BASE ###\n"
            f"Use the following information to answer user questions:\n"
            f"{knowledge_content}\n"
            f"########################"
        )
