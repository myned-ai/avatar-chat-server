import asyncio
import base64
import json
import queue
import sys
import threading
import time

import websockets

# Try to import pyaudio
try:
    import pyaudio
except ImportError:
    print("Error: PyAudio is required.")
    print("pip install pyaudio websockets")
    sys.exit(1)

# Configuration matched to simple_gemini_test.py (working)
SERVER_URL = "ws://localhost:8080/ws"
# Input: 16k (working mic setting)
INPUT_SAMPLE_RATE = 16000
# Output: 24k (Gemini native response)
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024 
MIC_GAIN = 1.0     

# Global flags
is_running = True
is_ai_speaking = False

# Queue for playback
playback_queue = asyncio.Queue()

# Queue for mic input (from blocking read thread -> send task)
mic_queue = asyncio.Queue()


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
        return default_device_index


async def capture_mic_input(p, device_index):
    """
    Captures audio using blocking reads in a separate thread (asyncio.to_thread).
    This matches simple_gemini_test.py pattern.
    """
    print(f"[Mic] Opening stream on device {device_index} at {INPUT_SAMPLE_RATE}Hz...")
    try:
        stream = await asyncio.to_thread(
            p.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("[Mic] Listening...")
        
        while is_running:
            try:
                # Blocking read in thread executor
                data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                await mic_queue.put(data)
            except Exception as e:
                print(f"[Mic] Error reading stream: {e}")
                break
                
    except Exception as e:
        print(f"[Mic] Fatal stream error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()


async def send_audio(websocket):
    """Reads from mic_queue and sends to WebSocket."""
    # 1. Start Stream
    # Note: We tell server we are sending 16k if needed, but since we updated server config to 16k default, 
    # and default matches, we are good.
    # Just in case, let's send explicit config in start message if the server supports it?
    # Actually, current server code ignores 'config' in start message unless we uncommented it.
    # But since loop config is updated to 16k, it matches.
    
    await websocket.send(json.dumps({
        "type": "audio_stream_start",
        "userId": "test_client_working_pattern",
        # "config": { "audio": { "inputSampleRate": 16000 } } 
    }))

    while is_running:
        data = await mic_queue.get()
        
        # Visualization
        rms = get_rms(data)
        status = "Listening" if is_ai_speaking else "Talking"
        bars = "|" * int(rms / 100)
        if len(bars) > 20: bars = bars[:20] + "+"
        print(f"\r[MIC] {status} RMS:{int(rms):<5} {bars:<22}         ", end="", flush=True)

        b64_data = base64.b64encode(data).decode("utf-8")
        await websocket.send(json.dumps({
            "type": "audio",
            "data": b64_data
        }))


async def play_audio_worker(p):
    """
    Plays audio from playback_queue.
    """
    print(f"[Speaker] Opening stream at {OUTPUT_SAMPLE_RATE}Hz...")
    try:
        stream = await asyncio.to_thread(
            p.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True
        )
        
        while is_running:
            data = await playback_queue.get()
            await asyncio.to_thread(stream.write, data)
            
    except Exception as e:
        print(f"[Speaker] Error: {e}")


async def receive_audio(websocket):
    global is_ai_speaking
    print("[Speaker] Ready for messages...")
    
    try:
        while is_running:
            message = await websocket.recv()
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "sync_frame":
                if not is_ai_speaking: continue

                audio_b64 = data.get("audio")
                if audio_b64:
                    # print(".", end="", flush=True)
                    audio_bytes = base64.b64decode(audio_b64)
                    await playback_queue.put(audio_bytes)
            
            elif msg_type == "transcript_delta":
                print(f"\r[AI] {data.get('text'):<60}", end="", flush=True)
            
            elif msg_type == "transcript_done":
                print(f"\n[AI COMPLETE] {data.get('text', '')}")

            elif msg_type == "audio_start":
                 print(f"\n[SERVER] Audio Start")
                 is_ai_speaking = True

            elif msg_type == "interrupt":
                print("\n[!!!] INTERRUPT RECEIVED - Clearing playback queue...")
                is_ai_speaking = False
                # Drain queue
                while not playback_queue.empty():
                    playback_queue.get_nowait()

            elif msg_type == "audio_end":
                 print("\n[SERVER] Audio Finished")
                 is_ai_speaking = False

    except websockets.exceptions.ConnectionClosed:
        print("\n[Receive] Connection closed")
    except Exception as e:
        print(f"\n[Receive] Error: {e}")


async def user_interrupt_monitor(websocket):
    """Monitors stdin for ENTER key to send interrupt."""
    loop = asyncio.get_running_loop()
    print(" [Controls] Press ENTER to Interrupt")
    while is_running:
        try:
            await loop.run_in_executor(None, sys.stdin.readline)
            if not is_running: break
            print(" [User] Interrupt!", flush=True)
            await websocket.send(json.dumps({"type": "interrupt"}))
            # Clear local
            while not playback_queue.empty():
                playback_queue.get_nowait()
        except:
            break


async def run_client():
    global is_running
    
    p = pyaudio.PyAudio()
    device_index = select_input_device(p)
    
    print(f"Connecting to {SERVER_URL}...")
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("Connected!")
            
            # Start Tasks
            tasks = [
                asyncio.create_task(capture_mic_input(p, device_index)),
                asyncio.create_task(send_audio(websocket)),
                asyncio.create_task(receive_audio(websocket)),
                asyncio.create_task(play_audio_worker(p)),
                asyncio.create_task(user_interrupt_monitor(websocket))
            ]
            
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for t in tasks: t.cancel()
            
    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        is_running = False
        p.terminate()

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(run_client())
    except KeyboardInterrupt:
        pass
