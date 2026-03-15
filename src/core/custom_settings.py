from .settings import CoreSettings

class CustomSettings(CoreSettings):
    """
    Specific settings for the Avatar custom domain demo.
    Replace these instructions with your own domain-specific logic.
    """
    @property
    def domain_instructions(self) -> str:
        return (
            "**Custom Domain Capabilities:**\n"
            "1. PRODUCT SELECTION: When you receive a 'viewing_product' SYSTEM EVENT, this means the user clicked on a specific product. "
            "You must silently update your internal context to prioritize this newly selected product. If the user then asks 'tell me more' or 'what is this?', "
            "assume they are referring to the MOST RECENT product from the 'viewing_product' event, completely ignoring previous selections. "
            "Do NOT call the `request_screen_context` tool for product inquiries if you already have the product data from this event.\n"
            "2. 3D PRODUCT VIEWER: The user has a 3D model viewer on their screen. If the user asks to see another angle, "
            "such as 'show me the back of it' or 'rotate it left', you must deduce the desired direction and call the `rotate_3d_model` action."
        )
