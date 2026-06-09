# Reproducing the environment on a fresh RTX 5090 box

1. Base: NVIDIA PyTorch container (torch with sm_120 support), CUDA 12.8+.
2. `pip install qwen-tts soundfile`  (pulls transformers, accelerate, etc.)
3. torchaudio gotcha: if the container's torch is a custom NVIDIA build, stock
   torchaudio wheels fail with `undefined symbol`. qwen_tts only imports
   `torchaudio.compliance.kaldi` (unused by the 12Hz path). Either install a
   matching torchaudio or drop in the 3-file stub described in README.md
   ("Note" section) at site-packages/torchaudio/.
4. Model downloads automatically on first run:
   Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (public, no HF token needed).
5. Kernel JIT-compiles on first import (-arch=sm_120a), ~1-2 min.
6. Smoke test: `python3 synthesize.py "Hello." --out hello.wav`
