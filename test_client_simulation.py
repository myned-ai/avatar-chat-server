import asyncio
import json
import time

import websockets

WS_URL = "ws://localhost:8080/ws"

class ChatClientSimulator:
    def __init__(self):
        self.current_turn_id = None
        self.turn_text = ""
        self.ws = None
        self.start_time = 0
        self.session_active = True

    async def run(self):
        try:
            async with websockets.connect(WS_URL) as websocket:
                self.ws = websocket
                print(f"[System] Connected to {WS_URL}")
                print("[System] This script simulates a client displaying text optimistically")
                print("[System] and then correcting it when an interruption occurs.\n")

                # 1. Send a request that generates a long response
                trigger_msg = {
                    "type": "text",
                    "data": "Tell me a long story about space exploration so I can interrupt you."
                }
                print(f"[User] Sent: \"{trigger_msg['data']}\"")
                await self.ws.send(json.dumps(trigger_msg))

                # 2. Schedule an interruption after 3 seconds
                asyncio.create_task(self.invoke_interruption(3.0))

                # 3. Handle events
                async for message in self.ws:
                    if not self.session_active:
                        break
                    
                    event = json.loads(message)
                    self.process_event(event)
        except Exception as e:
            print(f"\n[Error] {e}")

    async def invoke_interruption(self, delay):
        print(f"[System] Waiting {delay}s before interrupting...")
        await asyncio.sleep(delay)
        
        print(f"\n\n>>> [ACTION] TRIGGERING INTERRUPTION NOW <<<")
        print(">>> [ACTION] Sending 'interrupt' event")
        # Send explicit interrupt (simulating VAD trigger or Stop button)
        await self.ws.send(json.dumps({"type": "interrupt"}))
        
        await asyncio.sleep(0.5)
        print(">>> [ACTION] Sending new user test: 'Stop, that is enough.'")
        await self.ws.send(json.dumps({"type": "text", "data": "Stop, that is enough."}))
        
        # Let the new response play out briefly then exit
        await asyncio.sleep(5.0)
        self.session_active = False

    def process_event(self, event):
        evt_type = event.get("type")
        
        # Calculate relative time since audio start for logging
        current_time = time.time()
        rel_time_str = "0.00"
        if self.start_time > 0:
            rel_time_str = f"{(current_time - self.start_time):.2f}"

        if evt_type == "audio_start":
            self.current_turn_id = event.get("turnId")
            self.turn_text = ""
            self.start_time = time.time()
            print(f"\n[{rel_time_str}s] [AUDIO_START] New Turn ID: {self.current_turn_id}")

        elif evt_type == "transcript_delta":
            role = event.get("role")
            if role == "assistant":
                turn_id = event.get("turnId")
                
                # Only update if it matches current turn
                if turn_id == self.current_turn_id:
                    new_text = event.get("text", "")
                    self.turn_text += new_text
                    # Carriage return \r to update the line in place
                    print(f"\r[{rel_time_str}s] [DISPLAY] {self.turn_text}", end="", flush=True)

        elif evt_type == "interrupt":
            turn_id = event.get("turnId")
            offset = event.get("offsetMs")
            print(f"\n[{rel_time_str}s] [INTERRUPT] Server detected interruption on Turn {turn_id}")
            print(f"           └─ Cutoff Audio at: {offset}ms")
            print(f"           └─ Action: Stop audio player & prune buffer.")

        elif evt_type == "transcript_done":
            turn_id = event.get("turnId")
            final_text = event.get("text")
            is_interrupted = event.get("interrupted", False)
            role = event.get("role", "assistant")
            
            if role == "user":
                 print(f"\n[{rel_time_str}s] [USER] Transcribed: \"{final_text}\"")
                 return

            if is_interrupted:
                print(f"\n[{rel_time_str}s] [CORRECTION] Turn {turn_id} was interrupted!")
                print(f"           └─ Old Display: \"{self.turn_text}\"")
                print(f"           └─ New Display: \"{final_text}\" (Truncated)")
                # Update our local state to match truth
                self.turn_text = final_text
            else:
                if turn_id == self.current_turn_id:
                    print(f"\n[{rel_time_str}s] [DONE] Turn Complete. Final: \"{final_text}\"")

        elif evt_type == "control":
            # Ignore session updates for this view
            pass

if __name__ == "__main__":
    client = ChatClientSimulator()
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nDisconnected")
