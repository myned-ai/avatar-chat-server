import asyncio
import websockets
import json

async def test_confetti():
    uri = "ws://localhost:8080/ws"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Starting stream...")
            
            # 1. Start stream message
            await websocket.send(json.dumps({
                "type": "audio_stream_start",
                "userId": "test-user-script"
            }))
            
            # Wait a moment for config and initialization
            await asyncio.sleep(1)
            
            # 2. Send the specific text command
            print("Sending text prompt: 'Call the trigger_confetti function immediately. Do not talk, just call the function.'")
            await websocket.send(json.dumps({
                "type": "text",
                "data": "Call the trigger_confetti function immediately. Do not say anything else, just execute the tool."
            }))

            print("Waiting for agent response...")
            action_triggered = False
            
            # Listen for up to 15 seconds
            for _ in range(150):
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(msg)
                    
                    if data["type"] == "trigger_action":
                        print("\n========================================================")
                        print(f"✅ SUCCESS! Received action trigger from server:")
                        print(f"   Function:  {data['function_name']}")
                        print(f"   Arguments: {data['arguments']}")
                        print("========================================================\n")
                        action_triggered = True
                        break
                    elif data["type"] == "transcript_done":
                        print(f"[Agent Speaking] {data.get('text', '')}")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error reading message: {e}")
                    break

            if not action_triggered:
                print("\n❌ FAILED: The agent responded but never triggered the 'trigger_action' websocket event.")
                
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_confetti())
