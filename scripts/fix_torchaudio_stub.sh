#!/usr/bin/env bash
# Fix the torchaudio ABI-mismatch crash.
#
# Symptom (on `python3 app.py` / `synthesize.py` / anything importing qwen_tts):
#   OSError: .../torchaudio/lib/libtorchaudio.abi3.so:
#   undefined symbol: torch_dtype_float4_e2m1fn_x2
#
# Cause: this container ships a custom NVIDIA torch build; stock PyPI torchaudio
# wheels are compiled against standard PyTorch and fail at dlopen. pip may
# silently (re)install torchaudio as a dependency of something else, overwriting
# the stub. qwen_tts only needs `import torchaudio.compliance.kaldi` to succeed
# (the kaldi path is used by the 25Hz tokenizer only — the 12Hz models we use
# never call it), so a 3-file stub is sufficient.
#
# Usage: bash scripts/fix_torchaudio_stub.sh
set -euo pipefail

SITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
TA="$SITE/torchaudio"

pip uninstall -y torchaudio 2>/dev/null || true
rm -rf "$TA"
mkdir -p "$TA/compliance"

cat > "$TA/__init__.py" << 'EOF'
"""Stub torchaudio — the real wheel is ABI-incompatible with this container's torch build.
qwen_tts only needs torchaudio.compliance.kaldi for the 25Hz tokenizer path, which the
12Hz models don't use. This stub satisfies the import without the broken C extension."""
__version__ = "0.0.0-stub"
from . import compliance  # noqa: F401
EOF

cat > "$TA/compliance/__init__.py" << 'EOF'
from . import kaldi  # noqa: F401
EOF

cat > "$TA/compliance/kaldi.py" << 'EOF'
"""Stub of torchaudio.compliance.kaldi — raises if actually called."""


def fbank(*args, **kwargs):
    raise NotImplementedError(
        "torchaudio is stubbed in this environment (ABI mismatch with the "
        "container's torch build). kaldi.fbank is only needed for the 25Hz "
        "tokenizer's x-vector extraction, which this project does not use."
    )
EOF

python3 -c "import torchaudio; print('torchaudio stub OK:', torchaudio.__version__)"
