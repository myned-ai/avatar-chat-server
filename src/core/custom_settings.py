from .settings import CoreSettings


class CustomSettings(CoreSettings):
    """
    Specific settings for the Avatar custom domain demo.
    Replace these instructions with your own domain-specific logic.
    """
    @property
    def domain_instructions(self) -> str:
        return (
            ""
        )
