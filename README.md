# Avatar Chat Server

**Sample backend server for the [Avatar Chat Widget](https://github.com/myned-ai/avatar-chat-widget)**

> **See the Avatar Chat Widget in action -> [Try Nyx](https://myned.ai)**

Real-time voice-to-avatar interaction server combining AI agents (OpenAI Realtime API, Google Gemini Live API or your custom agent) with the Wav2Arkit model for synchronized avatar facial animation. This is an example server that powers the [Avatar Chat Widget](https://github.com/myned-ai/avatar-chat-widget) by processing audio streams and generating ARKit blendshapes for realistic facial animations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

## Features

- **Real-time Voice-to-Voice AI**: OpenAI Realtime API or Google Gemini Live API integration for natural conversation
- **Facial Animation Sync**: Wav2Arkit model for ARKit-compatible blendshapes
- **Modular Agent System**: Pluggable agents (sample OpenAI/Gemini or custom implementations)
- **WebSocket Communication**: Low-latency bidirectional streaming
- **CPU Acceleration**: ONNX-optimized inference for real-time performance without GPU
- **Production Ready**: Docker support, health checks, logging, authentication

## Use Cases

- **AI Customer Support Agents**: Replace chatbots with branded, empathetic 3D avatars
- **Interactive Kiosks**: Voice-enabled avatars for retail or information desks
- **Virtual Concierges**: Hospitality assistants that can see (via multimodal input) and speak
- **Education & Training**: Roleplay scenarios with responsive virtual characters

## Architecture

This server acts as the central brain between the client (Widget) and the AI (LLM):

1.  **Audio In**: Receives microphone audio from the user (Client)
2.  **Agent Processing**: Forwards audio to OpenAI/Gemini/Custom Agent
3.  **Response Generation**: Receives audio response from Agent
4.  **Facial Animation**: Runs Wav2Arkit model (ONNX) to generate lip-sync frames
5.  **Sync Out**: Streams synchronized Audio + Blendshape packets to Client at 30 FPS

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [Local Development](#local-development)
  - [Docker](#docker)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Docker Usage](#docker-usage)
- [Authentication](#authentication)
- [Performance Best Practices](#performance-best-practices)
- [Agent Modularity](#agent-modularity)
- [Development](#development)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Prerequisites

### Local Development

- Python 3.10.x (exact version required)
- [uv](https://github.com/astral-sh/uv) package manager
- **One of the following API keys** (depending on your chosen agent):
  - OpenAI API key with Realtime API access (for `sample_openai` agent)
  - Google Gemini API key (for `sample_gemini` agent)
- ONNX Runtime (CPU-optimized, included in dependencies)

### Docker

- Docker 20.10+
- Docker Compose 2.0+
- **One of the following API keys** (depending on your chosen agent):
  - OpenAI API key with Realtime API access (for `sample_openai` agent)
  - Google Gemini API key (for `sample_gemini` agent)

## Quick Start

### Local Development

```bash
# 1. Install uv (if not already installed)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/myned-ai/avatar-chat-server.git
cd avatar-chat-server

# 3. Install dependencies
uv sync

# 4. Download the Wav2Arkit model
pip install -U "huggingface_hub[cli]"
mkdir -p src/pretrained_models
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir src/pretrained_models

# 5. Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and other settings

# 6. Run server
uv run python src/main.py
```

Server will start at `http://localhost:8080`

**Test the server:** Open `test.html` in your browser to test the [Avatar Chat Widget](https://github.com/myned-ai/avatar-chat-widget) with your local server. Make sure `AUTH_ENABLED=false` in your `.env` file for testing.

### Docker

```bash
# 1. Clone and configure
git clone https://github.com/myned-ai/avatar-chat-server.git
cd avatar-chat-server
cp .env.example .env
# Edit .env with your settings

# 2. Download the ONNX model
pip install -U "huggingface_hub[cli]"
mkdir -p src/pretrained_models
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir src/pretrained_models

# 3. Build and run (production)
docker-compose up -d

# 4. View logs
docker-compose logs -f

# 5. Stop server
docker-compose down

# Development mode (with hot reload)
docker-compose --profile dev up
```

## Configuration

All settings can be configured via environment variables or `.env` file.

### Required Settings

At least one API key is required, depending on your chosen `AGENT_TYPE`:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (required if using `sample_openai` agent) |
| `GEMINI_API_KEY` | Your Google Gemini API key (required if using `sample_gemini` agent) |

### Optional Settings

| Variable | Default | Description |
|----------|---------|-------------|
| **Agent Configuration** |
| `AGENT_TYPE` | `sample_openai` | Agent type: `sample_openai`, `sample_gemini`, `remote` |
| `AGENT_URL` | *(none)* | WebSocket URL for remote agent (e.g., `ws://agent-service:8080/ws`) |
| **OpenAI Configuration** |
| `OPENAI_MODEL` | `gpt-realtime` | Realtime API model |
| `OPENAI_VOICE` | `alloy` | Voice: `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar` |
| `OPENAI_VOICE_SPEED` | `1.2` | Voice speed (0.25–1.5, where 1.0 is normal) |
| `OPENAI_TRANSCRIPT_SPEED` | `15.0` | Transcript character emission speed (chars/sec) for client sync |
| `OPENAI_TRANSCRIPTION_MODEL` | `gpt-4o-transcribe` | Transcription model for user speech |
| `OPENAI_TRANSCRIPTION_LANGUAGE` | `en` | Language for transcription (ISO-639-1 code) |
| `OPENAI_VAD_TYPE` | `semantic_vad` | Voice Activity Detection type: `server_vad`, `semantic_vad` |
| `OPENAI_VAD_THRESHOLD` | `0.9` | VAD threshold (0.0–1.0, higher = less sensitive) |
| `OPENAI_VAD_SILENCE_DURATION_MS` | `500` | Silence duration before turn ends (ms) |
| `OPENAI_VAD_PREFIX_PADDING_MS` | `300` | Audio to include before detected speech (ms) |
| `OPENAI_NOISE_REDUCTION` | `near_field` | Noise reduction: `near_field`, `far_field`, or empty to disable |
| **Gemini Configuration** |
| `GEMINI_MODEL` | `gemini-2.5-flash-native-audio-preview-12-2025` | Live API model |
| `GEMINI_VOICE` | `Leda` | Voice: `Puck`, `Charon`, `Kore`, `Fenrir`, `Aoede`, `Leda`, `Orus`, `Zephyr` |
| `GEMINI_API_VERSION` | `v1alpha` | Gemini API version |
| `GEMINI_THINKING_BUDGET` | `-1` | Thinking budget: `0`=disabled, `-1`=dynamic, `1`–`32768`=fixed tokens |
| `GEMINI_GOOGLE_SEARCH_GROUNDING` | `false` | Enable Google Search grounding for real-time information |
| `GEMINI_PROACTIVE_AUDIO` | `false` | Allow model to decide not to respond if content is not relevant |
| `GEMINI_CONTEXT_WINDOW_COMPRESSION` | `true` | Enable context window compression for longer sessions |
| `GEMINI_INPUT_SAMPLE_RATE` | `16000` | Audio sample rate sent to Gemini |
| `GEMINI_OUTPUT_SAMPLE_RATE` | `24000` | Audio sample rate received from Gemini |
| `GEMINI_TRANSCRIPT_SPEED` | `15.0` | Transcript character emission speed (chars/sec) |
| `GEMINI_VAD_START_SENSITIVITY` | `START_SENSITIVITY_LOW` | VAD start sensitivity |
| `GEMINI_VAD_END_SENSITIVITY` | `END_SENSITIVITY_LOW` | VAD end sensitivity |
| `GEMINI_TURN_COVERAGE` | `TURN_INCLUDES_ALL_INPUT` | Turn coverage mode |
| **Assistant Configuration** |
| `ASSISTANT_INSTRUCTIONS` | *see .env.example* | System prompt for the AI assistant |
| **Model Configuration** |
| `ONNX_MODEL_PATH` | `./pretrained_models/wav2arkit_cpu.onnx` | Path to ONNX model weights (CPU-only) |
| **Server Configuration** |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Server port |
| `USE_SSL` | `false` | Enable SSL (WebSocket endpoint becomes `wss://`) |
| `DEBUG` | `false` | Enable debug logging (verbose output) |
| `DEBUG_AUDIO_CAPTURE` | `false` | Save incoming audio to files for debugging |
| **Audio Configuration** |
| `INPUT_SAMPLE_RATE` | `24000` | Input audio sample rate (widget format) |
| `OUTPUT_SAMPLE_RATE` | `24000` | Output audio sample rate (for playback and lip-sync) |
| `WAV2ARKIT_SAMPLE_RATE` | `16000` | Wav2Arkit model expected sample rate |
| `BLENDSHAPE_FPS` | `30` | Output blendshape frame rate |
| `AUDIO_CHUNK_DURATION` | `0.5` | Audio chunk duration in seconds for Wav2Arkit processing |
| `TRANSCRIPT_CHARS_PER_SECOND` | `16.0` | Transcript timing estimation (chars/sec) |
| **Authentication** |
| `AUTH_ENABLED` | `false` | Enable HMAC token authentication |
| `AUTH_SECRET_KEY` | *(auto-generated)* | Secret key for HMAC signing (generate with `openssl rand -hex 32`) |
| `AUTH_TOKEN_TTL` | `3600` | Token time-to-live in seconds |
| `AUTH_ALLOWED_ORIGINS` | `localhost:5173,...` | CORS allowed origins (comma-separated) |
| `AUTH_ENABLE_RATE_LIMITING` | `true` | Enable rate limiting |

See [.env.example](.env.example) for complete configuration template.

## API Documentation

### REST Endpoints

- `GET /inf` - Server info and status
- `GET /health` - Health check endpoint
- `GET /docs` - OpenAPI documentation (Swagger UI)
- `GET /redoc` - ReDoc documentation
- `POST /api/auth/token` - Generate HMAC auth token (if authentication enabled)

### WebSocket Endpoint

Connect to `ws://localhost:8080/ws` (or `wss://` for production with TLS)

With authentication:
```
ws://localhost:8080/ws?token=YOUR_AUTH_TOKEN
```

#### Client → Server Messages

| Type | Payload | Description |
|------|---------|-------------|
| `text` | `{"type": "text", "data": "Hello"}` | Send text message to AI |
| `audio_stream_start` | `{"type": "audio_stream_start", "userId": "user123"}` | Start audio streaming session |
| `audio` | `{"type": "audio", "data": "<base64>"}` | Audio chunk (e.g. PCM16, 24kHz mono, base64-encoded) |
| `interrupt` | `{"type": "interrupt"}` | Explicitly interrupt AI response |
| `ping` | `{"type": "ping"}` | Heartbeat to keep connection alive |

#### Server → Client Messages

| Type | Description |
|------|-------------|
| `config` | Sent on connection. Contains negotiated audio settings (see below) |
| `audio_start` | AI started responding. Includes `sessionId`, `turnId`, `sampleRate`, `format` |
| `sync_frame` | Synchronized audio + blendshape frame at 30 FPS (see below) |
| `audio_chunk` | Fallback audio-only chunk when Wav2Arkit model is unavailable |
| `audio_end` | AI finished responding. Includes `sessionId`, `turnId` |
| `transcript_delta` | Streaming text fragment with timing offsets (see below) |
| `transcript_done` | Complete transcript for a turn (`role`: `"user"` or `"assistant"`) |
| `avatar_state` | Avatar state change (`"Listening"` or `"Responding"`) |
| `interrupt` | User interrupted AI response. Includes `turnId`, `offsetMs` |
| `error` | Error message |
| `pong` | Heartbeat response with `timestamp` |

#### Server Message Payloads

**`config`** — Sent immediately on WebSocket connection:
```json
{
  "type": "config",
  "audio": {
    "inputSampleRate": 24000
  }
}
```

**`audio_start`** — AI started a new response turn:
```json
{
  "type": "audio_start",
  "sessionId": "...",
  "turnId": "turn_1234567890_abcdef01",
  "sampleRate": 24000,
  "format": "audio/pcm16",
  "timestamp": 1234567890123
}
```

**`sync_frame`** — Synchronized audio + blendshape data (30 FPS):
```json
{
  "type": "sync_frame",
  "weights": { "jawOpen": 0.3, "mouthSmile_L": 0.5, ... },
  "audio": "<base64-pcm16>",
  "sessionId": "...",
  "turnId": "...",
  "timestamp": 1234567890123,
  "frameIndex": 0
}
```

**`audio_chunk`** — Fallback when Wav2Arkit model is not available (no blendshapes):
```json
{
  "type": "audio_chunk",
  "data": "<base64-pcm16>",
  "sessionId": "...",
  "timestamp": 1234567890123
}
```

**`transcript_delta`** — Streaming text with timing offsets for client sync:
```json
{
  "type": "transcript_delta",
  "text": "Hello",
  "role": "assistant",
  "turnId": "...",
  "sessionId": "...",
  "timestamp": 1234567890123,
  "startOffset": 0,
  "endOffset": 312
}
```

**`interrupt`** — Sent when user interrupts an AI response:
```json
{
  "type": "interrupt",
  "timestamp": 1234567890123,
  "turnId": "...",
  "offsetMs": 1500
}
```

#### Blendshape Weights Format

The `weights` object in `sync_frame` messages contains 52 ARKit-compatible blendshape coefficients (0.0-1.0):

```json
{
  "browInnerUp": 0.0,
  "browDown_L": 0.0,
  "browDown_R": 0.0,
  "jawOpen": 0.3,
  "mouthSmile_L": 0.5,
  "mouthSmile_R": 0.5,
  ...
}
```

See [ARKit Blendshape Documentation](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation) for complete list.

## Docker Usage

### Multi-Stage Build

The Dockerfile uses a multi-stage build optimized for CPU-only production:

1. **Base Stage**: Python 3.10-slim (Debian-based) with system dependencies
2. **Dependencies Stage**: Fast dependency installation with uv
3. **Production Stage**: Minimal image with non-root user, health checks
4. **Development Stage**: Hot reload support for development

### Production Deployment

```bash
# Build image
docker build -t avatar-chat-server .

# Run (CPU-only)
docker run -d \
  --name avatar-chat-server \
  -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/src/pretrained_models:/app/pretrained_models:ro \
  --restart unless-stopped \
  avatar-chat-server

# View logs
docker logs -f avatar-chat-server

# Health check
curl http://localhost:8080/health
```

### Docker Compose Profiles

- **Default (Production)**: `docker-compose up -d`
  - Optimized production build
  - Runs as non-root user
  - Health checks enabled
  - Auto-restart on failure

- **Development**: `docker-compose --profile dev up`
  - Hot reload enabled
  - Source code mounted as volume
  - Debug logging enabled

## Authentication

The server uses **HMAC-SHA256 signed tokens** for authentication. Tokens are origin-bound and time-limited.

### Enabling Authentication

Set in `.env`:
```bash
AUTH_ENABLED=true
AUTH_SECRET_KEY=your-secret-key-here-generate-with-openssl-rand-hex-32
AUTH_ALLOWED_ORIGINS=https://yourwebsite.com,https://www.yourwebsite.com
```

If `AUTH_SECRET_KEY` is not set, the server will auto-generate one at startup (not suitable for production — the key will change on each restart).

### Generate Secret Key

```bash
openssl rand -hex 32
```

### Getting a Token

The token endpoint validates the `Origin` header and returns a signed token for that origin:

```bash
curl -X POST http://localhost:8080/api/auth/token \
  -H "Origin: https://yourwebsite.com"
```

Response:
```json
{
  "token": "base64-encoded-hmac-token",
  "ttl": 3600,
  "origin": "https://yourwebsite.com"
}
```

### Using the Token

Include in WebSocket URL:
```
ws://localhost:8080/ws?token=YOUR_AUTH_TOKEN
```

### Security Layers

The authentication middleware implements a 4-layer security model:

1. **Origin Validation** — Whitelist check against `AUTH_ALLOWED_ORIGINS`
2. **HMAC Token Verification** — Signature and expiry validation
3. **Rate Limiting** — Token bucket algorithm (if enabled)
4. **Monitoring** — Audit logging

### Rate Limiting

When enabled, uses a **token bucket algorithm** with two levels:

- **Per-domain**: 100 token capacity, refills at 10 tokens/sec
- **Per-session**: 30 token capacity, refills at 5 tokens/sec

Exceeding limits results in WebSocket-level rejection (not HTTP 429).

## Performance Best Practices

### Hardware Recommendations

- **CPU**: 4+ cores recommended for real-time ONNX inference (8+ cores optimal)
- **RAM**: 8GB+ (16GB recommended)
- **Network**: Low-latency connection to AI API (< 100ms recommended)

### CPU Optimization Notes

The server uses ONNX Runtime for CPU-optimized inference. For best performance:
- ONNX Runtime automatically uses the best available SIMD instructions (SSE, AVX2, AVX-512)
- Ensure sufficient RAM (16GB+ recommended)
- Monitor RTF (Real-Time Factor) in debug logs; aim for <1.0
- If audio drops occur, consider reducing `audio_chunk_duration` in config

### Optimizations

The server implements several performance optimizations for real-time operation:

1. **Model Warmup**: Eliminates first-inference delay
2. **orjson**: 3-5x faster JSON serialization
3. **Base64 Pre-encoding**: Moves encoding off critical 30 FPS broadcast path
4. **Bounded Queues**: Prevents memory exhaustion under load
5. **Fire-and-forget Tasks**: Non-blocking broadcast to WebSocket clients

### Monitoring

Check server logs for performance indicators:
```bash
# Debug mode shows timing information
DEBUG=true uv run python src/main.py
```

Key metrics:
- WebSocket connection count
- Audio processing latency
- Model inference time
- Queue depths

## Agent Modularity

The server uses a modular agent system allowing different conversational AI backends:

### Sample Agents

- **sample_openai**: Uses OpenAI Realtime API (default)
- **sample_gemini**: Uses Google Gemini Live API

### Custom Agents

Implement the `BaseAgent` interface for custom AI services:

```python
from agents import BaseAgent, ConversationState

class MyCustomAgent(BaseAgent):
    @property
    def is_connected(self) -> bool:
        # Return connection status
        ...

    @property
    def state(self) -> ConversationState:
        # Return current conversation state
        ...

    @property
    def transcript_speed(self) -> float:
        # Transcript chars/sec for client sync (e.g., 15.0)
        ...

    def set_event_handlers(self, **kwargs) -> None:
        # Store callbacks: on_audio_delta, on_transcript_delta,
        # on_response_start, on_response_end, on_user_transcript,
        # on_interrupted, on_error
        ...

    async def connect(self) -> None:
        # Connect to your AI service
        ...

    def send_text_message(self, text: str) -> None:
        # Send text to AI
        ...

    def append_audio(self, audio_bytes: bytes) -> None:
        # Send audio to AI (PCM16 bytes)
        ...

    async def disconnect(self) -> None:
        # Disconnect from your AI service
        ...
```

Set `AGENT_TYPE=remote` and `AGENT_URL=ws://your-agent-service/ws` for remote agents.

### Switching Agents

1. Set `AGENT_TYPE` in `.env`
2. Provide required API keys
3. Restart the server

The core chat-server (WebSocket handling, blendshape generation) remains unchanged.

## Development

### Install Dev Dependencies

```bash
uv sync --group dev
```

### Code Quality Tools

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [ty](https://github.com/astral-sh/ty) for static type checking.

#### Linting

```bash
# Check for linting issues
uv run ruff check src/

# Auto-fix linting issues
uv run ruff check src/ --fix

# Fix with unsafe fixes (use with caution)
uv run ruff check src/ --fix --unsafe-fixes
```

#### Formatting

```bash
# Check formatting
uv run ruff format src/ --check

# Format code
uv run ruff format src/
```

#### Type Checking

```bash
# Run static type analysis
uv run ty check src/
```

#### Run All Checks

```bash
# Lint, format, and type check
uv run ruff check src/ --fix && uv run ruff format src/ && uv run ty check src/
```

### Running Tests

```bash
uv run pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Apple ARKit](https://developer.apple.com/augmented-reality/arkit/) for the blendshape specification standard
- [LAM Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) for facial animation model
- [Wav2Vec 2.0](https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) for speech representation learning
- [OpenAI](https://openai.com/) for Realtime API
- [Google](https://ai.google.dev/) for Gemini Live API
