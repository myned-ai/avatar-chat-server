import asyncio
import base64
import json
import sys

import websockets

# Try to import pyaudio
try:
    import pyaudio
except ImportError:
    print("Error: PyAudio is required.")
    print("pip install pyaudio websockets")
    sys.exit(1)

# Configuration
SERVER_URL = "ws://localhost:8080/ws"
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 2400  # 100ms at 24kHz
# [NEW] Audio Gate Threshold (Adjust based on mic sensitivity)
SILENCE_THRESHOLD = 200

# Global flag to stop threads
is_running = True

def is_silent(data_chunk):
    """Simple RMS amplitude check for 16-bit PCM."""
    # Convert bytes to integers (signed 16-bit little endian)
    count = len(data_chunk) // 2
    sum_squares = 0.0
    for i in range(0, len(data_chunk), 2):
        # Little-endian 16-bit signed integer
        sample = int.from_bytes(data_chunk[i:i+2], byteorder='little', signed=True)
        sum_squares += sample * sample
    
    rms = (sum_squares / count) ** 0.5
    return rms, rms < SILENCE_THRESHOLD

async def send_audio(websocket, input_stream, loop):
    """
    Reads audio from microphone and sends it to the server.
    Applies a simple Noise Gate to prevent accidental VAD interrupts.
    """
    print("[Mic] Started recording...")
    print(f"[Mic] Noise Gate Enabled (Threshold: {SILENCE_THRESHOLD})")
    print("[Mic] If you see 'Gated', speak louder.")
    
    # Notify server we are starting audio stream
    await websocket.send(json.dumps({
        "type": "audio_stream_start",
        "userId": "test_client"
    }))

    last_was_silent = True

    try:
        while is_running:
            # Read audio chunk in a separate thread
            audio_data = await loop.run_in_executor(
                None,
                lambda: input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            )
            
            # Check for silence
            rms, silent = await loop.run_in_executor(None, is_silent, audio_data)

            if silent:
                if not last_was_silent:
                    print(f"\n[Mic] Gated (RMS: {int(rms)}) ", end="", flush=True)
                else:
                    print(".", end="", flush=True)

                last_was_silent = True
                await asyncio.sleep(0.01)
                continue
            
            if last_was_silent:
                print(f"\n[Mic] Sending (RMS: {int(rms)}) > ", end="", flush=True)
            
            last_was_silent = False
            
            # Encode and send
            b64_data = base64.b64encode(audio_data).decode("utf-8")
            msg = {
                "type": "audio",
                "data": b64_data
            }
            await websocket.send(json.dumps(msg))
            await asyncio.sleep(0.001)
            
    except websockets.exceptions.ConnectionClosed:
        print("[Mic] Connection closed")
    except Exception as e:
        print(f"[Mic] Error: {e}")

async def receive_audio(websocket, output_stream):
    """
    Receives audio/text from server and plays it.
    """
    print("[Speaker] Listening for server audio...")
    try:
        while is_running:
            message = await websocket.recv()
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "sync_frame":
                # Play audio
                audio_b64 = data.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    output_stream.write(audio_bytes)
                    print(".", end="", flush=True)
            
            elif msg_type == "transcript_delta":
                print(f"\rAI: {data.get('text')}", end="", flush=True)
            
            elif msg_type == "transcript_done":
                print(f"\n[Transcript: {data.get('text')}]")

            elif msg_type == "audio_start":
                 print(f"\n[Audio Start] Turn ID: {data.get('turnId')}")
                
            elif msg_type == "interrupt":
                print("\n[Interrupted]")

            elif msg_type == "audio_end":
                 print("\n[Audio Finished]")

    except websockets.exceptions.ConnectionClosed:
        print("\n[Speaker] Connection closed")
    except Exception as e:
        print(f"\n[Speaker] Error: {e}")

async def run_client():
    global is_running
    
    p = pyaudio.PyAudio()
    
    # Input Stream (Microphone)
    try:
        input_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"Failed to open microphone: {e}")
        return

    # Output Stream (Speakers)
    try:
        output_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True
        )
    except Exception as e:
        print(f"Failed to open speakers: {e}")
        input_stream.close()
        return

    print(f"Connecting to {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("Connected! Start speaking (Ctrl+C to stop).")
            
            loop = asyncio.get_running_loop()
            
            # Run send and receive tasks concurrently
            sender_task = asyncio.create_task(send_audio(websocket, input_stream, loop))
            receiver_task = asyncio.create_task(receive_audio(websocket, output_stream))
            
            # Wait for either to finish (or error)
            _done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in pending:
                task.cancel()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nConnection Error: {e}")
    finally:
        is_running = False
        print("\nCleaning up...")
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        pass
