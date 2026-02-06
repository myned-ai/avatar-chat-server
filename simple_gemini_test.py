import asyncio
import os
import sys
import pyaudio
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

# Configuration
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
API_VERSION = "v1alpha"

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

client = genai.Client(
    api_key=API_KEY,
    http_options={"api_version": API_VERSION}
)

CONFIG = {
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {
                "voice_name": "Kore"
            }
        }
    },
    "input_audio_transcription": {},
    "output_audio_transcription": {},
    # [EXPERIMENTAL] Attempt to disable thinking process
    "generation_config": {
        "thinking_config": {
             "include_thoughts": False
        }
    }
}

class GeminiVoiceLoop:
    def __init__(self):
        self.pya = pyaudio.PyAudio()
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None

    async def listen_audio(self):
        """Reads from microphone and puts into out_queue."""
        mic_info = self.pya.get_default_input_device_info()
        print(f"[System] Opening microphone: {mic_info['name']}")
        
        audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        print("[System] Listening... (You can speak now)")
        while True:
            data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def send_realtime(self):
        """Sends audio from out_queue to Gemini session."""
        print("[System] Send task started.")
        while True:
            msg = await self.out_queue.get()
            try:
                # Log occasional sends to avoid spam, but confirm activity
                # print(f"[Send] {len(msg['data'])} bytes") 
                
                # await self.session.send(input=msg, end_of_turn=False) 
                # await self.session.send(msg, end_of_turn=False) # Try positional?
                await self.session.send_realtime_input(
                    audio=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                )
            except Exception as e:
                print(f"[Error] Send failed: {e}")
                await asyncio.sleep(1)

    async def receive_audio(self):
        """Receives audio from Gemini and puts into audio_in_queue."""
        print("[System] Receive task started.")
        while True:
            try:
                turn = self.session.receive()
                async for response in turn:
                    # Print raw response structure once to debug
                    # print(f"[DEBUG] Response keys: {[k for k in dir(response) if not k.startswith('_')]}")
                    
                    server_content = getattr(response, "server_content", None)
                    if server_content:
                        # Print Spoken Transcript
                        out_trans = getattr(server_content, "output_transcription", None)
                        if out_trans and out_trans.text:
                             print(f"\n[Gemini Spoken]: {out_trans.text}")
                        
                        # Print User Transcript
                        in_trans = getattr(server_content, "input_transcription", None)
                        if in_trans and in_trans.text:
                             print(f"\n[User]: {in_trans.text}")

                        # Print Thoughts (if any)
                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn:
                             parts = getattr(model_turn, "parts", [])
                             for part in parts:
                                 text = getattr(part, "text", None)
                                 if text:
                                     print(f"\n[Gemini Thought]: {text[:60]}...")

                        # Check for interruption
                        if getattr(server_content, "interrupted", False):
                            print("\n[RX] Interrupted")
                            # Clear playback queue on interrupt
                            while not self.audio_in_queue.empty():
                                self.audio_in_queue.get_nowait()

                    # Handle Audio
                    # Note: SDK yields response objects, check structure carefully
                    # Looking at debug_types, response usually has server_content
                    # But audio data might be in model_turn parts
                    
                    if server_content:
                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn:
                            parts = getattr(model_turn, "parts", [])
                            for part in parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and inline_data.data:
                                    self.audio_in_queue.put_nowait(inline_data.data)

            except Exception as e:
                print(f"\n[Error] Receive loop error: {e}")
                break

    async def play_audio(self):
        """Plays audio from audio_in_queue."""
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        print(f"[System] Connecting to Gemini ({MODEL})...")
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                print("[System] Connected!")
                print(f"[DEBUG] Session methods: {[m for m in dir(session) if not m.startswith('_')]}")

                # Use asyncio.gather instead of TaskGroup for Python <3.11 compatibility
                await asyncio.gather(
                    self.listen_audio(),
                    self.send_realtime(),
                    self.receive_audio(),
                    self.play_audio(),
                )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n[Error] Connection failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        main_loop = GeminiVoiceLoop()
        asyncio.run(main_loop.run())
    except KeyboardInterrupt:
        print("\n[System] Stopped by user.")
