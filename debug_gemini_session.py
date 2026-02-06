import asyncio
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
API_VERSION = "v1beta"

client = genai.Client(api_key=API_KEY, http_options={"api_version": API_VERSION})
CONFIG = {"response_modalities": ["AUDIO"]}

async def main():
    print("Connecting...")
    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        print("Connected.")
        print("Session methods:")
        methods = [m for m in dir(session) if not m.startswith('_')]
        print(methods)
        import inspect
        try:
             # print(f"Signature of session.send: {inspect.signature(session.send)}")
             # print(f"Signature of session.send_realtime_input: {inspect.signature(session.send_realtime_input)}")
             help(session.send_realtime_input)
        except Exception as e:
             print(f"Could not get signature: {e}")

if __name__ == "__main__":
    asyncio.run(main())
