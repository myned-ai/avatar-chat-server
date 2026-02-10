# Avatar Chat Client Implementation Specification

**Version:** 1.3
**Date:** January 31, 2026
**Target:** Client-Side Developers & AI Agents

## 1. Overview
This document specifies the communication protocol and implementation requirements for a client connecting to the Avatar Chat Server. The server wraps the OpenAI Realtime API and provides specific orchestration for 3D Avatars (audio, blendshapes, transcripts) and handles server-side Voice Activity Detection (VAD) for interruptions.

### 1.1 Transport
- **Protocol:** WebSocket (WS/WSS)
- **Endpoint:** `/ws`
- **Audio Format:** JSON (Base64 Encoded PCM16). Sample Rate is dictated by the server via `config` event.

---

## 2. Connection Lifecycle

1.  **Connect**: Client establishes WebSocket connection.
2.  **Config**: Server sends `config` event with expected sample rate.
3.  **Session Loop**:
    -   Client streams Base64 audio chunks (JSON).
    -   Server streams JSON events (text).
3.  **Interruption**: If user speaks while avatar is talking, Server sends `interrupt` control event. Client must prune audio buffer.

---

## 3. Server-to-Client Events (Downstream)

All server messages are JSON objects. The `type` field is discriminative.

### 3.1 `config` (NEW)
Sent immediately after connection to inform client of required audio settings.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"config"` |
| `audio` | object | Yes | Audio settings |
| `audio.inputSampleRate` | number | Yes | Required sample rate for client mic (e.g. 16000 or 24000) |

### 3.2 `audio_start`
Signals the beginning of a new audio response turn from the Avatar.
*Trigger:* Analysis start or VAD silence detected on user end.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"audio_start"` |
| `turnId` | string | Yes | Unique ID for this turn (e.g., `turn_1727393...`) |
| `sessionId` | string | Yes | Current Session ID |
| `sampleRate` | number | Yes | Audio sample rate (default `24000`) |
| `format` | string | Yes | Audio format (default `"audio/pcm16"`) |
| `timestamp` | number | Yes | Server timestamp (ms) |

### 3.2 `sync_frame`
Contains a chunk of audio and the corresponding blendshapes (visemes) for facial animation.
*Frequency:* high (e.g., 30fps or 60fps depending on server settings).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"sync_frame"` |
| `audio` | string | Yes | Base64 encoded PCM16 raw audio bytes |
| `weights` | number[] | Yes | Array of 52 ARKit blendshape values (0.0 - 1.0) |
| `frameIndex` | number | Yes | Sequential index of the frame in this turn |
| `turnId` | string | Yes | Correlates to `audio_start.turnId` |
| `timestamp` | number | Yes | Server timestamp (ms) |

### 3.3 `audio_end`
Signals that the current audio turn has finished streaming from the server.
*Trigger:* AI finishes response generation.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"audio_end"` |
| `turnId` | string | Yes | The ID of the turn that just finished |
| `sessionId` | string | Yes | Current Session ID |
| `timestamp` | number | Yes | Server timestamp (ms) |

### 3.4 `transcript_delta`
Streaming text updates for UI display (User or Assistant).

> **Important:** Transcript deltas often arrive **faster** than the audio playback. The client is responsible for synchronizing the display of text with the audio, or handling the "rollback" if the text is displayed ahead of an interruption.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"transcript_delta"` |
| `role` | string | Yes | `"assistant"` or `"user"` |
| `text` | string | Yes | The chunk of text (token) to append |
| `turnId` | string | Yes | Correlates to audio turn |
| `startOffset` | number | No | Estimated start time in ms relative to turn (assistant only) |
| `endOffset` | number | No | Estimated end time in ms relative to turn (assistant only) |
| `itemId` | string | No | Source Item ID (if available) |
| `previousItemId` | string | No | Previous Item ID (if available) |

### 3.5 `transcript_done`
Final confirmed text for a turn. Sent when silence is detected (user) or generation finishes (assistant).

> **Interruption Handling:** If `interrupted` is true, this event contains the **truncated** text representing what was actually spoken before the cut-off. The client **must** replace its current displayed text with this value to fix any "future text" that was displayed but never spoken.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"transcript_done"` |
| `role` | string | Yes | `"assistant"` or `"user"` |
| `text` | string | Yes | Full text of the turn |
| `turnId` | string | Yes | Turn ID |
| `interrupted`| boolean| No | `true` if this transcript was cut short by interruption |
| `itemId` | string | No | Item ID (if available) |

### 3.6 `interrupt` (CRITICAL)
Sent when Server VAD detects user speech while Avatar is outputting audio.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"interrupt"` |
| `turnId` | string | Yes | The turn ID that is being interrupted (ACTIVE turn) |
| `offsetMs` | number | Yes | The exact cut-off point in milliseconds relative to `audio_start` |
| `timestamp` | number | Yes | Server timestamp (ms) |

**Client Action (Immediate):**
1.  **Stop Playback**: Stop audio immediately if playback position > `offsetMs`.
2.  **Prune Buffer**: Discard any `sync_frames` in queue.
3.  **UI Feedback**: Visual indication that avatar stopped listening/speaking.

### 3.7 `avatar_state`
High-level state for UI status indicators.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"avatar_state"` |
| `state` | string | Yes | `"Listening"`, `"Responding"`, or `"Processing"` |

### 3.8 `pong`
Response to client ping.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"pong"` |
| `timestamp` | number | Yes | Server timestamp |

---

## 4. Client-to-Server Events (Upstream)

### 4.1 `audio_stream_start`
Sent before sending initial raw audio bytes.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"audio_stream_start"` |
| `userId` | string | No | Client user identifier |

### 4.2 `audio`
**IMPORTANT:** The server currently ONLY supports JSON audio. Binary messages are not supported.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"audio"` |
| `data` | string | Yes | Base64 encoded PCM16 audio |

### 4.3 `text`
Send text input (Simulates "User Transcribed" event).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"text"` |
| `data` | string | Yes | The user's message |

### 4.4 `interrupt`
Manually trigger interruption from client side (e.g. "Stop" button).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"interrupt"` |

### 4.5 `ping`
Keepalive.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Value: `"ping"` |

---

## 5. Client Logic & Best Practices

To implement a "Human-Like" conversational experience, the client **must** implement the following logic defined in python-like pseudocode.

### 5.1 Global State
```python
current_turn_id = None
audio_buffer = Queue()
is_playing = False
start_time_of_current_turn = 0
```

### 5.2 Handling Start & End
```python
def on_audio_start(event):
    current_turn_id = event.turnId
    audio_buffer.clear()
    
    # Reset tracking
    start_time_of_current_turn = now()
    
    # UI Update
    show_avatar_talking_state()

def on_audio_end(event):
    if event.turnId == current_turn_id:
        # Do NOT stop playback immediately! 
        # The buffer still has audio to play.
        # Just mark the stream as closed.
        mark_stream_complete() 
```

### 5.3 Handling Interruption (The "Magic" Logic)
This logic ensures the avatar stops speaking exactly where the user interrupted it, preventing "Zombie Audio" (avatar finishing a sentence after being interrupted).

```python
def on_interrupt(event):
    interrupted_turn = event.turnId
    cutoff_offset = event.offsetMs
    
    # 1. Verification
    if current_turn_id != interrupted_turn:
        return # Ignore outdated interrupts
    
    # 2. Stop processing new frames
    ignore_future_frames(interrupted_turn)
    
    # 3. Calculate local playback position
    ms_played = audio_player.get_playback_position_ms()
    
    # 4. Handle immediate cut vs fast-forward
    if ms_played >= cutoff_offset:
        # We played too much (latency), stop NOW
        audio_player.stop()
        audio_buffer.clear()
    else:
        # We are slightly behind, schedule stop
        remaining_ms = cutoff_offset - ms_played
        schedule_stop(in_ms=remaining_ms)
        # Discard any audio in buffer that comes AFTER cutoff_offset
        prune_buffer_after(cutoff_offset)
```

### 5.4 Transcript Synchronization & Rollback
Text generation is faster than audio generation. 

1.  **Advance Arrival**: You will receive `transcript_delta` events for words that the avatar has not spoken yet.
2.  **Display Strategy**:
    *   **Option A (Karaoke):** Use `startOffset` to schedule the text display to match the audio playback position.
    *   **Option B (Optimistic):** Display text immediately as it arrives.
3.  **Interruption Rollback**:
    *   If you use **Option B**, an interruption will cause a discrepancy: the screen shows a full sentence, but the avatar stops halfway.
    *   **Correction:** When `transcript_done` arrives with `interrupted: true` and the same `turnId`, you **must** replace the displayed text bubble with the `text` from this event. It represents the truncated version of what was actually spoken.
    *   **Ignore Future Deltas:** Once an `interrupt` or `transcript_done` is received for a `turnId`, ignore any subsequent `transcript_delta` messages for that same `turnId` (they may arrive out of order due to network/processing latency).

## 6. Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **"Zombie Audio"** | Client buffers too much audio and finishes playing queue after interrupt. | Implement `offsetMs` logic. Hard stop playback if `interrupt` received. |
| **Echo** | System audio feeding back into mic. | Use headphones or implement Acoustic Echo Cancellation (AEC) on client. |
| **Stuttering** | Network jitter. | Implement a small jitter buffer (100-200ms) before starting playback of a new turn. |
| **Lip Sync Drift** | Audio clock vs Rendering clock drift. | Sync animation frame to audio timestamp. Use `sync_frame` provided `audio` chunk for timing source. |

---
**End of Specification**
