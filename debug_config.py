import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from core.settings import get_settings
from agents.gemini.gemini_settings import get_gemini_settings

def debug_settings():
    print("=== Configuration Debug ===")
    
    # Check .env file directly
    env_path = Path(".env")
    if env_path.exists():
        print(f".env file found at {env_path.absolute()}")
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if "GEMINI_VOICE" in line:
                print(f"RAW .env line: {line}")
    else:
        print(".env file NOT FOUND")

    # Check Pydantic settings loading
    try:
        settings = get_gemini_settings()
        print(f"\nLoaded GEMINI_VOICE from settings: '{settings.gemini_voice}'")
        print(f"Loaded GEMINI_MODEL from settings: '{settings.gemini_model}'")
    except Exception as e:
        print(f"Error loading settings: {e}")

if __name__ == "__main__":
    debug_settings()
