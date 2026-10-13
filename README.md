# Almus Voice AI

Voice-generation API. **Phase 2:** optional real engine using the pretrained
**XTTS-v2** model, conditioned on a reference recording of an authorised speaker.

> **Not yet evaluated:** nobody has listened to output from a real reference recording
> yet. Do not claim the generated voice sounds like the owner until it has been tested
> and judged by listening. Passing tests proves valid audio is produced, not likeness.

## Model choice and verification

| Item | XTTS-v2 (via maintained `coqui-tts` fork) |
|---|---|
| Voice cloning | Yes, from a short (~6 s+) reference clip, cross-language |
| Languages | en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi |
| **Hausa (`ha`)** | **Not supported.** The API rejects it with 422 when XTTS is active |
| Package | `coqui-tts` (idiap fork; original `TTS` package is unmaintained). Python 3.10-3.14, PyTorch 2.2+ |
| Model licence | **Coqui Public Model License: non-commercial only** (model and its output). Coqui the company shut down in 2024, so no commercial licence is available |
| Download | ~2 GB; expect roughly 4-6 GB RAM in use |
| Hardware | CPU works but is slow (seconds to minutes per sentence). A CUDA GPU is much faster |

These were checked against the model card, the PyPI/GitHub pages and licence text when
Phase 2 was written. Re-check before relying on them (versions change). The code was
**not executed against the real model** in the environment that produced it: run the
integration test below on your own machine.

**Licence consequence:** personal and research use is fine. If Almus becomes a paid or
revenue-generating product, XTTS-v2 cannot be used.

**Hausa consequence:** Phase 2 can speak English (and the other languages above) in the
cloned voice. Real Hausa speech needs a different or fine-tuned model, which is outside Phase 2.

## Installation

Use Python 3.11 or 3.12.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
# 1. PyTorch first. CPU:
pip install "torch>=2.2,<2.9" "torchaudio>=2.2,<2.9" --index-url https://download.pytorch.org/whl/cpu
#    GPU: use the selector at https://pytorch.org/get-started/locally/ for your CUDA version.
#    (Staying below 2.9 avoids the extra torchcodec/FFmpeg requirement.)
# 2. App + engine:
pip install -r requirements-xtts.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env`: set `API_KEYS`, `ADMIN_API_KEYS` (different values), `VOICE_ENGINE=xtts_v2`.

### Download the model (explicit licence acceptance)

```bash
python scripts/download_xtts.py --accept-license --dest models/XTTS-v2
```

`models/` is git-ignored. The app never downloads models by itself.

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The model loads on the first generation request (slow once), then stays in memory. Set
`XTTS_PRELOAD=true` to load at startup instead. `XTTS_DEVICE=cuda` fails clearly if no
GPU exists; it never silently falls back to CPU. Requests are processed one at a time.

## Reference voices

Store your own consented recording. It must be an **uncompressed PCM WAV**, mono or
stereo, >= 16 kHz, 5-60 s (6-30 s of clean speech is ideal), <= 10 MB. Convert other formats first,
e.g. `ffmpeg -i in.m4a -ac 1 -ar 22050 ref.wav`.

```bash
curl -X PUT "http://127.0.0.1:8000/v1/voices/almus_owner" \
  -H "X-Admin-Key: YOUR_ADMIN_KEY" -H "Content-Type: audio/wav" \
  --data-binary @ref.wav
curl -H "X-Admin-Key: YOUR_ADMIN_KEY" http://127.0.0.1:8000/v1/voices
curl -X DELETE -H "X-Admin-Key: YOUR_ADMIN_KEY" http://127.0.0.1:8000/v1/voices/almus_owner
```

Voice ids: letters, digits, `_`, `-`, max 64 characters. Files are saved to
`REFERENCE_VOICE_DIR` (default `data/voices/`, git-ignored, owner-only permissions).
Remote URLs are never fetched. Add `?overwrite=true` to replace an existing voice.

**Never commit or upload recordings to GitHub.** `.gitignore` does not stop uploads made
through the GitHub web interface, and Git history keeps deleted files.

## Generate speech

```bash
curl -X POST http://127.0.0.1:8000/v1/voice/generate \
  -H "X-API-Key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"text":"Hello, welcome to Almus AI.","language":"en","voice_id":"almus_owner"}' \
  --output speech.wav
```

| Status | Meaning |
|---|---|
| 200 | `audio/wav` body (16-bit mono, 24 kHz) |
| 401 / 503 | bad key / auth not configured |
| 404 | reference voice not found |
| 422 | invalid text, language, voice id, or language unsupported by the engine |
| 501 | no engine configured (`VOICE_ENGINE=none`) |
| 502 | inference failed (no audio returned) |
| 503 | model could not load |

## Tests

```bash
pytest -v                          # mocked tests, no torch, no model, runs in CI
pytest -m integration -v           # real model; skipped unless the variables below are set
```

```bash
export ALMUS_XTTS_MODEL_DIR=models/XTTS-v2
export ALMUS_REFERENCE_WAV=/path/to/consented_reference.wav
export ALMUS_XTTS_DEVICE=cpu       # or cuda
pytest -m integration -v
```

CI (`.github/workflows/ci.yml`) installs only `requirements-dev.txt`: no PyTorch, no model
download, no GPU. The real engine is never exercised on free runners.

## Docker

Phase 1 image (no engine): `docker compose up --build`.
Real engine on CPU: `docker compose -f docker-compose.xtts.yml up --build`, with the model in
`./models/XTTS-v2` and voices in `./data/voices` (mounted, never baked in).
GPU in Docker needs the NVIDIA container toolkit and a CUDA PyTorch image; not provided here.

## Troubleshooting

- **503 "voice engine is unavailable":** check `XTTS_MODEL_DIR` contains `config.json`, `model.pth`, `vocab.json`, and that `torch` and `coqui-tts` import. The server log shows the error type only.
- **Errors loading the checkpoint (e.g. unpickling):** upgrade `coqui-tts` and PyTorch to compatible current versions; mismatches are the usual cause. Send the log line to the developer.
- **Missing torchcodec / FFmpeg:** appears with PyTorch >= 2.9; install `torchcodec` and FFmpeg, or stay on torch < 2.9.
- **422 language unsupported:** XTTS-v2 has no Hausa. Use a supported language.
- **Garbled or cut-off audio on long text:** keep sentences short; lower `MAX_TEXT_LENGTH`.
- **Very slow:** CPU inference is slow; use a GPU.

## Security and consent

Only record and use voices with the speaker's explicit consent. Generation needs `X-API-Key`;
voice management needs a separate `X-Admin-Key`; both fail closed if unset. Put TLS and
rate limiting in front before any public exposure. Logs hold metadata only (language,
length), never text, audio or keys.

## Current limitations and unresolved issues

- Real-model inference **not executed** by the author; the first real run may expose compatibility issues.
- Voice likeness unevaluated.
- Hausa unsupported by XTTS-v2; non-commercial licence.
- Single-process, one request at a time; no request timeout or rate limiting; no streaming.
- Reference upload accepts PCM WAV only.
- Cloning one clip is zero-shot; fine-tuning on a larger dataset is not part of Phase 2.
