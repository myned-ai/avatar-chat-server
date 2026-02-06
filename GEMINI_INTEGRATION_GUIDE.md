# Gemini Live API Integration Guide

This document outlines the specific configuration, constraints, and workarounds required to successfully integrate the Google Gemini Live API (`gemini-2.5-flash-native-audio-preview`) with the Avatar Chat Server.

## 1. API Version & SDK
*   **Version**: `v1alpha` (Required by the current `google-genai` SDK for `send_realtime_input`).
*   **Method**: Use `session.send_realtime_input(audio=types.Blob(...))` for audio transmission.
    *   *Note*: The official cookbook uses `v1beta` and `session.send()`, but this is incompatible with some installed SDK versions.

## 2. Audio Configuration
### Output (Agent -> User)
*   **Sample Rate**: **24,000 Hz** (Native Gemini Output).
*   **Do NOT Resample**: The server's `Wav2Arkit` and the Client (OpenAI-compatible) both expect or handle 24kHz.
    *   *Regression Risk*: Downsampling to 16kHz results in "chipmunk" audio (fast playback) because the client clock remains at 24kHz.
*   **Format**: PCM 16-bit, Little Endian, Mono.

### Input (User -> Agent)
*   User input (from `test_gemini_audio.py` or web client) is received at 24kHz (or varies).
*   We currently send it raw. Gemini handles commonly supported PCM formats.

## 3. Transcription & "Thinking" Process
The `gemini-2.5-flash-native-audio-preview` model has a unique behavior where it outputs its "Chain of Thought" (CoT) as text before speaking.

### The Problem
*   Internal thoughts (e.g., `**Crafting Response**`) are sent via `part.text`.
*   Spoken text is sent via `server_content.output_transcription.text`.

### The Solution (Filtering)
1.  **Ignore `part.text`**: In `SampleGeminiAgent.py`, we explicitly ignore `part.text` to prevent internal thoughts from reaching the user.
2.  **Enable Output Transcription**: You **MUST** specifically enable transcription in the config to receive the spoken text.

### Configuration Quirk (CRITICAL)
To enable transcription without causing a **Connection Timeout** or crash:
*   **DO**: Pass empty configuration objects.
    ```python
    output_audio_transcription=types.AudioTranscriptionConfig() 
    # NO arguments! Passing model="..." causes 1000/timeout errors.
    ```
*   **DON'T**: Pass `model="gemini-..."` inside the config object.

## 4. Interruption Handling
*   **Server-Side**: The server must send an explicit `{"type": "interrupt"}` JSON message to the client when it detects an interruption.
*   **Client-Side**: The client must listen for this message and **immediately clear** its audio playback queue to stop the previous response.
*   **Text Marker**: The `interrupted=True` flag in `transcript_done` is a secondary signal, but the explicit message is faster.

## 5. Known Issues
*   **Thinking Process**: Disabling the "Thinking" budget (`tokens=0`) caused instability in previous tests. We rely on filtering the text output instead.
