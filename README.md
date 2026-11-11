# Chatterbox TTS Android

On-device multilingual text-to-speech using [Chatterbox](https://github.com/resemble-ai/chatterbox) with ExecuTorch.

## Features

- 🗣️ On-device TTS — no cloud API needed
- 🌍 23 languages supported
- 📱 Material 3 UI with Jetpack Compose
- 📥 Auto-downloads models from HuggingFace on first run (~2.6GB)
- 💾 Save generated audio as WAV

## Architecture

The app runs the full Chatterbox TTS pipeline on-device using ExecuTorch:

1. **Text Tokenization** — BPE tokenizer (MTLTokenizer port) with multilingual support
2. **Conditioning** — Default voice embedding applied via conditioning model
3. **T3 Decode** — Autoregressive speech token generation (prefill + decode loop)
4. **Vocoder** — S3Gen encoder → CFM (2-step) → HiFiGAN for waveform synthesis

## Models

9 ExecuTorch `.pte` models downloaded from [acul3/chatterbox-executorch](https://huggingface.co/acul3/chatterbox-executorch):

| Model | Purpose |
|-------|---------|
| `cond_model.pte` | Text + speaker conditioning |
| `t3_prefill.pte` | T3 prefill (KV cache init) |
| `t3_decode.pte` | T3 autoregressive decode |
| `s3gen_encoder.pte` | Speech tokens → mel spectrogram |
| `cfm_step1.pte` | CFM denoising step 1 |
| `cfm_step2.pte` | CFM denoising step 2 |
| `hifigan.pte` | Mel → waveform |
| `tokenizer_encoder.pte` | Neural tokenizer encoder |
| `ve_model.pte` | Voice encoder |

## Building

### With GitHub Actions (recommended)

Push to the repo and the CI workflow will build the APK automatically.

### Locally

```bash
# Requires Android SDK and JDK 17
./gradlew assembleDebug
```

## Setup

### Tokenizer Vocab

Replace the placeholder vocab in `app/src/main/assets/grapheme_mtl_merged_expanded_v1.json` with the real one from HuggingFace:

```bash
./scripts/download_assets.sh
```

### Voice Conditioning

For the default voice, extract conditioning tensors from `conds.pt` and place as binary files in assets:

```bash
python scripts/extract_conds.py
```

## Supported Languages

English, French, German, Spanish, Portuguese, Italian, Dutch, Polish, Czech, Romanian, Hungarian, Finnish, Danish, Swedish, Turkish, Russian, Chinese, Japanese, Korean, Malay, Indonesian, Thai, Vietnamese

## Requirements

- Android 8.0+ (API 26)
- ~3GB storage for models
- arm64-v8a or x86_64 device

## License

Apache 2.0
Build Android APK.
