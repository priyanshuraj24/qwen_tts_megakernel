# Reproducing the environment on a fresh RTX 5090 box

1. Base: NVIDIA PyTorch container (torch with sm_120 support), CUDA 12.8+.
2. `pip install qwen-tts soundfile`  (pulls transformers, accelerate, etc.)
3. torchaudio gotcha: if the container's torch is a custom NVIDIA build, stock
   torchaudio wheels fail at import with
   `OSError: libtorchaudio.abi3.so: undefined symbol: torch_dtype_float4_e2m1fn_x2`
   (symbol name may vary by version). qwen_tts only imports
   `torchaudio.compliance.kaldi` (unused by the 12Hz path), so a stub suffices.
   Fix with one command:

       bash scripts/fix_torchaudio_stub.sh

   **This can recur**: any `pip install` that pulls torchaudio back in as a
   dependency overwrites the stub and the app stops starting with the error
   above. Just re-run the script. (Happened once on 2026-06-10 — app.py
   crashed on startup after a stray torchaudio 2.11.0 install.)
4. Model downloads automatically on first run:
   Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (public, no HF token needed).
5. Kernel JIT-compiles on first import (-arch=sm_120a), ~1-2 min.
6. Smoke test: `python3 synthesize.py "Hello." --out hello.wav`
