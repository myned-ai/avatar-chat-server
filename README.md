# Avatar Chat Server

**Sample backend server for the [Avatar Chat Widget](https://github.com/myned-ai/avatar-chat-widget)**

Real-time voice-to-avatar interaction server combining AI agents (OpenAI Realtime API or Google Gemini Live API) with the Wav2Arkit model for synchronized avatar facial animation. This is an example server that powers the [Avatar Chat Widget](https://github.com/myned-ai/avatar-chat-widget) by processing audio streams and generating ARKit blendshapes for realistic facial animations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Real-time Voice-to-Voice AI**: OpenAI Realtime API or Google Gemini Live API integration for natural conversation
- **Facial Animation Sync**: Wav2Arkit model for ARKit-compatible blendshapes
- **Modular Agent System**: Pluggable agents (sample OpenAI/Gemini or custom implementations)
- **WebSocket Communication**: Low-latency bidirectional streaming
- **CPU Acceleration**: ONNX-optimized inference for real-time performance without GPU
- **Production Ready**: Docker support, health checks, logging, authentication
- **Optimized Performance**: Model warmup, orjson serialization, bounded queues

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
- [Development](#development)
- [License](#license)


## Prerequisites

### Local Development

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key with Realtime API access (for `sample_openai` agent)
- Google Gemini API key (for `sample_gemini` agent)
- ONNX Runtime (CPU-optimized, included in dependencies)

### Docker

- Docker 20.10+
- Docker Compose 2.0+
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
cd avatar_chat_server

# 3. Install dependencies
uv sync

# 4. Download the Wav2Arkit model
pip install -U "huggingface_hub[cli]"
mkdir -p pretrained_models
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir pretrained_models
mv pretrained_models/ src/

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
cd avatar_chat_server
cp .env.example .env
# Edit .env with your settings

# 2. Download the ONNX model
pip install -U "huggingface_hub[cli]"
mkdir -p pretrained_models
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir pretrained_models
mv pretrained_models/ src/

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

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (required for `sample_openai` agent) |
| `GEMINI_API_KEY` | Your Google Gemini API key (required for `sample_gemini` agent) |

### Optional Settings

| Variable | Default | Description |
|----------|---------|-------------|
| **Agent Configuration** |
| `AGENT_TYPE` | `sample_openai` | Agent type: `sample_openai`, `sample_gemini`, `remote` |
| `AGENT_URL` | *(none)* | WebSocket URL for remote agent (e.g., `ws://agent-service:8080/ws`) |
| **OpenAI Configuration** |
| `OPENAI_MODEL` | `gpt-4o-realtime-preview` | Realtime API model |
| `OPENAI_VOICE` | `alloy` | Voice for audio output (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) |
| **Gemini Configuration** |
| `GEMINI_MODEL` | `gemini-2.0-flash-exp` | Live API model |
| `GEMINI_VOICE` | `Puck` | Voice for audio output (`Puck`, `Kore`, `Fenrir`, `Aoede`) |
| **Assistant Configuration** |
| `ASSISTANT_INSTRUCTIONS` | *see .env.example* | System prompt for the AI assistant |
| **Model Configuration** |
| `ONNX_MODEL_PATH` | `./pretrained_models/wav2arkit_cpu.onnx` | Path to ONNX model weights (CPU-only) |
| **Server Configuration** |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Server port |
| `DEBUG` | `false` | Enable debug logging (verbose output) |
| **Authentication** |
| `AUTH_ENABLED` | `true` | Enable JWT authentication |
| `AUTH_SECRET_KEY` | *(generated)* | Secret key for JWT signing (generate with `openssl rand -hex 32`) |
| `AUTH_TOKEN_TTL` | `3600` | Token time-to-live in seconds |
| `AUTH_ALLOWED_ORIGINS` | *(comma-separated)* | CORS allowed origins |
| `AUTH_ENABLE_RATE_LIMITING` | `true` | Enable rate limiting |

See [.env.example](.env.example) for complete configuration template.

## API Documentation

### REST Endpoints

- `GET /` - Server info and status
- `GET /health` - Health check endpoint
- `GET /docs` - OpenAPI documentation (Swagger UI)
- `GET /redoc` - ReDoc documentation
- `POST /auth/token` - Generate JWT token (if authentication enabled)

### WebSocket Endpoint

Connect to `ws://localhost:8080/ws` (or `wss://` for production with TLS)

With authentication:
```
ws://localhost:8080/ws?token=YOUR_JWT_TOKEN
```

#### Client → Server Messages

| Type | Payload | Description |
|------|---------|-------------|
| `text` | `{"type": "text", "data": "Hello"}` | Send text message to AI |
| `audio_stream_start` | `{"type": "audio_stream_start", "userId": "user123"}` | Start audio streaming session |
| `audio` | `{"type": "audio", "data": "<base64>"}` | Audio chunk (PCM16, 24kHz mono) |
| `audio_stream_end` | `{"type": "audio_stream_end"}` | End audio streaming session |
| `ping` | `{"type": "ping"}` | Heartbeat to keep connection alive |

#### Server → Client Messages

| Type | Payload | Description |
|------|---------|-------------|
| `audio_start` | `{"type": "audio_start", "sessionId": "..."}` | AI started responding |
| `sync_frame` | `{"type": "sync_frame", "sessionId": "...", "audio": "<base64>", "weights": {...}}` | Synchronized audio + blendshape frame (30 FPS) |
| `audio_end` | `{"type": "audio_end", "sessionId": "..."}` | AI finished responding |
| `transcript_delta` | `{"type": "transcript_delta", "delta": "text"}` | Streaming assistant text (real-time) |
| `transcript_done` | `{"type": "transcript_done", "transcript": "...", "role": "user/assistant"}` | Complete transcript |
| `avatar_state` | `{"type": "avatar_state", "state": "Listening/Responding"}` | Avatar state change |
| `interrupt` | `{"type": "interrupt"}` | User interrupted AI response |
| `error` | `{"type": "error", "error": "..."}` | Error message |
| `pong` | `{"type": "pong"}` | Heartbeat response |

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

1. **Base Stage**: Ubuntu 22.04 with Python 3.10
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

### Enabling Authentication

Set in `.env`:
```bash
AUTH_ENABLED=true
AUTH_SECRET_KEY=your-secret-key-here-generate-with-openssl-rand-hex-32
AUTH_ALLOWED_ORIGINS=https://yourwebsite.com,https://www.yourwebsite.com
```

### Generate Secret Key

```bash
openssl rand -hex 32
```

### Getting a Token

```bash
curl -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token

Include in WebSocket URL:
```
ws://localhost:8080/ws?token=YOUR_JWT_TOKEN
```

### Rate Limiting

When enabled, limits requests per IP:
- 100 requests per minute per IP
- 429 status code when exceeded
- Automatic reset after 1 minute

## Performance Best Practices

### Hardware Recommendations

- **CPU**: 4+ cores recommended for real-time ONNX inference (8+ cores optimal)
- **RAM**: 8GB+ (16GB recommended)
- **Network**: Low-latency connection to AI API (< 100ms recommended)

### CPU Optimization Notes

The server uses ONNX Runtime for CPU-optimized inference. For best performance:
- Use a CPU with AVX-512 support if available
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
from agents import BaseAgent

class MyCustomAgent(BaseAgent):
    async def connect(self) -> None:
        # Connect to your AI service
        pass
    
    def send_text_message(self, text: str) -> None:
        # Send text to AI
        pass
    
    def append_audio(self, audio_bytes: bytes) -> None:
        # Send audio to AI
        pass
    
    # ... implement other methods
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
