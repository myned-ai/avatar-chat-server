import asyncio
import os
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def test_connection():
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
    voice = os.getenv("GEMINI_VOICE", "Puck")
    
    if not api_key:
        logger.error("No API key found")
        return

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    
    instructions = "You are a helpful and friendly AI assistant. Always respond in English only, even if the user speaks another language. Keep your responses brief and conversational - no more than 2-3 sentences unless specifically asked for more detail.  Only respond to clear audio or text.  If the user's audio is not clear (e.g., ambiguous input/background noise/silent/unintelligible) or if you did not fully hear or understand the user, ask for clarification"

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        )
    )

    logger.info(f"Connecting to {model} with voice {voice}...")
    
    try:
        # Simulate the Agent's behavior: manual __aenter__
        logger.info("Attempting manual __aenter__...")
        ctx = client.aio.live.connect(model=model, config=config)
        session = await ctx.__aenter__()
        
        logger.info("Connected! Waiting for messages...")
        
        # Create a background task to send dummy audio
        async def send_dummy_audio():
            logger.info("Starting audio stream...")
            try:
                # Generate 1 second of silence at 16kHz
                # 16000 samples * 2 bytes = 32000 bytes
                chunk_size = 1024
                silence = b'\x00' * chunk_size
                
                for i in range(50): # Send for a bit
                    await session.send_realtime_input(
                        audio=types.Blob(data=silence, mime_type="audio/pcm")
                    )
                    await asyncio.sleep(0.05)
                logger.info("Finished sending dummy audio")
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

        asyncio.create_task(send_dummy_audio())

        try:
            async for response in session.receive():
                logger.info(f"Received: {response}")
        except Exception as e:
            logger.error(f"Receive loop ended: {e}")
        
        # Cleanup
        await ctx.__aexit__(None, None, None)
                
    except Exception as e:
        logger.error(f"Connection failed: {e}")
                
    except Exception as e:
        logger.error(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
