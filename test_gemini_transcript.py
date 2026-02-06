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
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 2400  # 100ms
MIC_GAIN = 10.0    # Boost mic for VAD testing

# Global flags
is_running = True
is_ai_speaking = False
playback_queue = queue.Queue()

# Trackers for validation
current_turn_id = None
last_offset_ms = 0

def playback_worker(output_stream):
    """Background thread to play audio from queue."""
    print("[System] Playback worker started")
    while is_running:
        try:
            data = playback_queue.get(timeout=0.1)
            output_stream.write(data)
            playback_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"\n[System] Playback Error: {e}")
            break

def apply_gain(data_bytes, gain):
    """Simple digital gain."""
    if gain == 1.0: 
        return data_bytes
    samples = []
    for i in range(0, len(data_bytes), 2):
        sample = int.from_bytes(data_bytes[i:i+2], byteorder='little', signed=True)
        val = int(sample * gain)
        val = max(min(val, 32767), -32768)
        samples.append(val.to_bytes(2, byteorder='little', signed=True))
    return b"".join(samples)

def get_input_device_index(p):
    """Select appropriate input device."""
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    # Use default if available, otherwise 0
    return info.get('defaultInputDevice')

async def send_audio(websocket, input_stream, loop):
    """Reads mic and sends to server."""
    print(f"[Mic] Streaming started (Gain: {MIC_GAIN}x)...")
    
    await websocket.send(json.dumps({
        "type": "audio_stream_start",
        "userId": "gemini_transcript_tester"
    }))

    while is_running:
        try:
            # Non-blocking read
            audio_data = await loop.run_in_executor(
                None, lambda: input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            )
            
            if MIC_GAIN != 1.0:
                audio_data = apply_gain(audio_data, MIC_GAIN)

            b64_data = base64.b64encode(audio_data).decode("utf-8")
            await websocket.send(json.dumps({
                "type": "audio",
                "data": b64_data
            }))
            
            await asyncio.sleep(0.005)
        except Exception as e:
            if is_running:
                print(f"[Mic] Error: {e}")
            break

async def receive_messages(websocket):
    """Handles incoming messages and focuses on Transcript Validation."""
    global is_ai_speaking, current_turn_id, last_offset_ms
    
    print("[System] Client connected. Waiting for events...")
    print("-" * 80)
    print(f"{'ROLE':<10} | {'TURN ID':<30} | {'OFFSET (ms)':<15} | TEXT")
    print("-" * 80)

    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "transcript_delta":
                role = data.get("role", "unknown")
                turn_id = data.get("turnId", "unknown")
                text = data.get("text", "")
                timestamp = data.get("timestamp")
                start = data.get("startOffset", 0)
                end = data.get("endOffset", 0)
                
                # Check for logic errors
                offset_str = f"{start}-{end}"
                continuity_check = ""
                
                if turn_id == current_turn_id:
                   if start < last_offset_ms:
                       continuity_check = " [Warning: Time Regression]"
                   last_offset_ms = end
                else:
                    # New turn detected implicitly via delta (should normally get audio_start first)
                    continuity_check = " [New Turn Stream]"
                    current_turn_id = turn_id
                    last_offset_ms = end

                print(f"{role:<10} | {turn_id[-8:]:<30} | {offset_str:<15} | {text}{continuity_check}")

            elif msg_type == "transcript_done":
                role = data.get("role")
                text = data.get("text")
                interrupted = data.get("interrupted", False)
                final_dur = data.get("finalAudioDurationMs", 0)
                
                status = "[INTERRUPTED]" if interrupted else "[COMPLETE]"
                print(f"{'SYSTEM':<10} | {'-'*30} | {final_dur:<15} | {status} {text}")
                print("-" * 80)

            elif msg_type == "audio_start":
                turn_id = data.get("turnId") or "unknown_turn"
                current_turn_id = turn_id
                last_offset_ms = 0
                is_ai_speaking = True
                print(f"{'SYSTEM':<10} | {turn_id[-8:]} (NEW TURN)      | {'0':<15} | Audio Started")

            elif msg_type == "interrupt":
                offset = data.get("offsetMs", "N/A")
                turn_id = data.get("turnId") or "Unknown"
                print(f"{'SYSTEM':<10} | {str(turn_id)[-8:]:<30} | {offset:<15} | *** INTERRUPTION SIGNAL RECEIVED ***")
                # Immediately clear playback
                with playback_queue.mutex:
                    playback_queue.queue.clear()
                is_ai_speaking = False

            elif msg_type == "sync_frame":
                # Only buffer if not interrupted
                if is_ai_speaking:
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        playback_queue.put(base64.b64decode(audio_b64))

            elif msg_type == "avatar_state":
                state = data.get("state")
                # print(f"[State] Avatar is now {state}")

    except websockets.exceptions.ConnectionClosed:
        print("\n[System] Connection closed by server.")
    except Exception as e:
        print(f"\n[System] Receive Error: {e}")

async def run():
    global is_running
    
    print("="*60)
    print("GEMINI AGENT TRANSCRIPT CLIENT TEST")
    print("Ensure AGENT_TYPE=sample_gemini is set in .env")
    print("="*60)

    # Audio Setup
    p = pyaudio.PyAudio()
    output_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        output=True
    )
    
    input_dev = get_input_device_index(p)
    input_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=input_dev,
        frames_per_buffer=CHUNK_SIZE
    )

    # Start Playback Thread
    t = threading.Thread(target=playback_worker, args=(output_stream,))
    t.start()

    # WebSocket Loop
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            loop = asyncio.get_running_loop()
            
            # Run Send/Recv in parallel
            send_task = asyncio.create_task(send_audio(websocket, input_stream, loop))
            recv_task = asyncio.create_task(receive_messages(websocket))
            
            await asyncio.gather(send_task, recv_task)
    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    finally:
        is_running = False
        t.join(timeout=1.0)
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
