import asyncio
import base64
import json
import time

import websockets

# Try to import pyaudio for playback context
try:
    import pyaudio
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("PyAudio not found. Running in silent mode (logs only).")

# Configuration
SERVER_URL = "ws://localhost:8080/ws"
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16 if HAS_AUDIO else None

async def run_interruption_test():
    p = None
    stream = None
    
    if HAS_AUDIO:
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True
            )
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            return

    print(f"Connecting to {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("Connected!")
            
            # 1. Start a long generation
            prompt = "Please count from one to twenty very slowly."
            print(f"\n[Step 1] Sending prompt: '{prompt}'")
            
            msg = {
                "type": "text",
                "data": prompt
            }
            await websocket.send(json.dumps(msg))
            
            # State tracking
            is_playing = True
            frames_received = 0
            interruption_sent = False
            start_time = time.time()
            
            print("[Step 2] Listening for audio (will interrupt in 3 seconds)...")
            
            while is_playing:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    print("Timeout waiting for message!")
                    break

                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "audio_start":
                    print(f"\n[Server] Audio Started (Turn: {data.get('turnId')})")
                    
                elif msg_type == "sync_frame":
                    frames_received += 1
                    
                    # Play audio if available
                    if HAS_AUDIO and stream:
                        audio_b64 = data.get("audio")
                        if audio_b64:
                            stream.write(base64.b64decode(audio_b64))
                            
                    if frames_received % 10 == 0:
                        print(".", end="", flush=True)
                    
                    # TRIGGER INTERRUPTION after ~3 seconds (approx 90 frames at 30fps)
                    if frames_received > 90 and not interruption_sent:
                        print("\n\n[Step 3] >>> SENDING INTERRUPT COMMAND <<<")
                        await websocket.send(json.dumps({"type": "interrupt"}))
                        interruption_sent = True
                        
                elif msg_type == "interrupt":
                    print("\n\n[Server] Confirmed Interruption!")
                    print("Waiting to ensure no more audio frames arrive...")
                    
                elif msg_type == "transcript_done":
                    # Check if the server marked it as interrupted
                    was_interrupted = data.get("interrupted", False)
                    print(f"\n[Server] Transcript Done. Interrupted flag: {was_interrupted}")
                    print(f"Final Text: {data.get('text')}")
                    
                    if was_interrupted:
                        print("\n[SUCCESS] Server correctly handled interruption.")
                        is_playing = False # Test complete
                    else:
                        print("\n[WARNING] Transcript finished normally? Interruption might have failed or been too late.")
                        is_playing = False

                elif msg_type == "audio_end":
                    print("\n[Server] Audio End Event received.")
                    if not interruption_sent:
                        print("[FAILED] Audio finished before we could interrupt!")
                    is_playing = False

    except websockets.exceptions.ConnectionClosed:
        print("\nConnection closed by server")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()

if __name__ == "__main__":
    if not HAS_AUDIO:
        print("Install pyaudio to hear the test: pip install pyaudio")
        
    try:
        asyncio.run(run_interruption_test())
    except KeyboardInterrupt:
        pass
