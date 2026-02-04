import asyncio
import base64
import json
import queue
import sys
import threading

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
# OpenAI/Server expects 24kHz. Windows PyAudio usually handles resampling automatically.
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 2400  # 100ms at 24kHz
MIC_GAIN = 20.0    # Boost microphone input by 20x to fix low volume issues

# Global flags
is_running = True
is_ai_speaking = False

# Audio Playback Queue
playback_queue = queue.Queue()

# Debugging
debug_frames = []

def playback_worker(output_stream):
    """
    Background thread to play audio from queue.
    Ensures smooth playback and allows instant clearing on interruption.
    """
    print("[Playback] Worker thread started")
    while is_running:
        try:
            # Blocking get with timeout to allow checking is_running
            data = playback_queue.get(timeout=0.1)
            output_stream.write(data)
            playback_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"\n[Playback] Error: {e}")
            break
    print("[Playback] Worker thread stopped")

def get_rms(data_chunk):
    """Calculate RMS amplitude."""
    count = len(data_chunk) // 2
    if count == 0:
        return 0
    sum_squares = 0.0
    for i in range(0, len(data_chunk), 2):
        sample = int.from_bytes(data_chunk[i:i+2], byteorder='little', signed=True)
        sum_squares += sample * sample
    return (sum_squares / count) ** 0.5

def apply_gain(data_bytes, gain):
    """Simple digital gain for 16-bit PCM."""
    if gain == 1.0:
        return data_bytes
        
    samples = []
    for i in range(0, len(data_bytes), 2):
        sample = int.from_bytes(data_bytes[i:i+2], byteorder='little', signed=True)
        val = int(sample * gain)
        # Hard clipping to prevent overflow
        val = max(min(val, 32767), -32768)
        samples.append(val.to_bytes(2, byteorder='little', signed=True))
    
    return b"".join(samples)


def select_input_device(p):
    """List inputs and ask user to select one."""
    print("\n=== Audio Input Devices ===")
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    input_devices = []

    default_device_index = info.get('defaultInputDevice')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            is_default = " [DEFAULT]" if i == default_device_index else ""
            print(f"ID {i}: {name}{is_default}")
            input_devices.append(i)

    print("\n---------------------------")
    print(f"Default microphone ID is: {default_device_index}")
    
    selection = input(f"Enter microphone ID to use [{default_device_index}]: ").strip()
    
    if not selection:
        return default_device_index
    
    try:
        idx = int(selection)
        if idx in input_devices:
            return idx
        print(f"Invalid ID. Using default: {default_device_index}")
        return default_device_index
    except ValueError:
        print(f"Invalid input. Using default: {default_device_index}")
        return default_device_index

async def send_audio(websocket, input_stream, loop):
    """
    Reads audio from microphone and sends it to the server.
    """
    global debug_frames
    print(f"[Mic] Recording started (Gain: {MIC_GAIN}x)")
    print("[Mic] Use HEADPHONES to avoid echo/interruption.")
    
    await websocket.send(json.dumps({
        "type": "audio_stream_start",
        "userId": "test_client"
    }))

    chunk_counter = 0

    try:
        while is_running:
            # Non-blocking read (key fix for stuttering)
            audio_data = await loop.run_in_executor(
                None,
                lambda: input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            )
            
            # [DEBUG] Verify packet size (Should be 4800 bytes for 2400 samples * 2 bytes)
            if chunk_counter == 0:
                 print(f"[DEBUG] First Chunk Size: {len(audio_data)} bytes (Expected: {CHUNK_SIZE * 2})")

            # [DEBUG] Record first 5 seconds (50 chunks)
            if chunk_counter < 50:
                debug_frames.append(audio_data)
                chunk_counter += 1
            elif chunk_counter == 50:
                 chunk_counter += 1
            
            if MIC_GAIN != 1.0:
                audio_data = apply_gain(audio_data, MIC_GAIN)

            rms = await loop.run_in_executor(None, get_rms, audio_data)

            # Show RMS to debug "shouting but silence" issues
            status = "Listening" if is_ai_speaking else "Talking"
            bars = "|" * int(rms / 100) # Simple VU meter
            if len(bars) > 20:
                bars = bars[:20] + "+"
            
            # Clear line before printing to avoid mixing with other output
            print(f"\r[MIC] {status} RMS:{int(rms):<5} {bars:<22}         ", end="", flush=True)

            b64_data = base64.b64encode(audio_data).decode("utf-8")
            msg = {
                "type": "audio",
                "data": b64_data
            }
            await websocket.send(json.dumps(msg))
            
            await asyncio.sleep(0.001)
            
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"\n[!!!] CONNECTION CLOSED UNEXPECTEDLY: {e}")
        print("[!!!] Server may have crashed or connection was terminated")
    except websockets.exceptions.ConnectionClosed:
        print("\n[Mic] Connection closed normally")
    except Exception as e:
        print(f"\n[Mic] Error: {type(e).__name__}: {e}")

async def receive_audio(websocket):
    """
    Receives audio/text from server and queues it for playback.
    """
    global is_ai_speaking
    print("[Speaker] Ready for audio...")
    
    # Track if we received audio_start for current turn
    audio_start_received = False
    sync_frames_before_start = 0
    
    try:
        while is_running:
            message = await websocket.recv()
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "sync_frame":
                # Check if sync_frame arrived before audio_start
                if not audio_start_received:
                    sync_frames_before_start += 1
                    print(f"\n[WARNING] sync_frame received BEFORE audio_start! (count: {sync_frames_before_start})")
                
                if not is_ai_speaking:
                    continue  # Ignore audio frame if we are interrupted/not speaking

                audio_b64 = data.get("audio")
                if audio_b64:
                    print(".", end="", flush=True) # [DEBUG]
                    audio_bytes = base64.b64decode(audio_b64)
                    # Put to queue instead of writing directly
                    playback_queue.put(audio_bytes)
            
            elif msg_type == "transcript_delta":
                print(f"\r[AI] {data.get('text'):<60}", end="", flush=True)
            
            elif msg_type == "transcript_done":
                text = data.get('text', '')
                print(f"\n[AI COMPLETE] {text}")

            elif msg_type == "audio_start":
                 if sync_frames_before_start > 0:
                     print(f"\n[ERROR] Received {sync_frames_before_start} sync_frames BEFORE audio_start!")
                 print(f"\n[SERVER] Audio Start - Turn ID: {data.get('turnId')}")
                 is_ai_speaking = True
                 audio_start_received = True
                 sync_frames_before_start = 0  # Reset counter for next turn

            elif msg_type == "interrupt":
                print("\n[!!!] INTERRUPT RECEIVED - Clearing playback queue...")
                is_ai_speaking = False
                audio_start_received = False  # Reset for next turn
                # CLEAR the queue immediately (thread-safe way)
                with playback_queue.mutex:
                    playback_queue.queue.clear()

            elif msg_type == "audio_end":
                 print("\n[SERVER] Audio Finished")
                 is_ai_speaking = False
                 audio_start_received = False  # Reset for next turn

    except websockets.exceptions.ConnectionClosed:
        print("\n[Speaker] Connection closed")
    except Exception as e:
        print(f"\n[Speaker] Error: {e}")

async def run_client():
    global is_running
    
    p = pyaudio.PyAudio()
    
    # 1. Select Device
    device_index = select_input_device(p)
    print(f"Selected Device Index: {device_index}")

    try:
        input_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index, # [FIX] Use selected device
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Output uses default
        output_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True
        )
    except Exception as e:
        print(f"Audio Device Error: {e}")
        try:
            print("Try different sample rate or device.")
        except:  # noqa: E722
            pass
        return

    # Start Playback Worker Thread
    worker_thread = threading.Thread(target=playback_worker, args=(output_stream,), daemon=True)
    worker_thread.start()

    print(f"Connecting to {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("Connected! Start speaking (Ctrl+C to stop).")
            
            loop = asyncio.get_running_loop()
            
            # [NEW] Keyboard Interrupt Listener (Threaded to avoid blocking loop)
            def key_listener(ws_loop, ws):
                print(" [Controls] Press ENTER to send INTERRUPT signal")
                while is_running:
                    try:
                        sys.stdin.readline()
                        if not is_running:
                            break
                        print(" [User] Sending Interrupt...", flush=True)
                        future = asyncio.run_coroutine_threadsafe(
                             ws.send(json.dumps({"type": "interrupt"})),
                             ws_loop
                        )
                        # Wait for send to complete
                        future.result(timeout=1.0)
                        print(" [User] Interrupt sent successfully", flush=True)
                        # Also clear local queue immediately
                        with playback_queue.mutex:
                            playback_queue.queue.clear()
                    except Exception as e:
                        print(f" [User] Error sending interrupt: {e}", flush=True)
                        break
            
            key_thread = threading.Thread(target=key_listener, args=(loop, websocket), daemon=True)
            key_thread.start()

            sender_task = asyncio.create_task(send_audio(websocket, input_stream, loop))
            receiver_task = asyncio.create_task(receive_audio(websocket))
            
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
        # Give worker a moment to exit
        worker_thread.join(timeout=1.0)

        try:
            if 'input_stream' in locals():
                input_stream.close()
            if 'output_stream' in locals():
                output_stream.close()
        except:  # noqa: E722
            pass
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        pass
