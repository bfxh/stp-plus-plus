"""STP++ — geometric regularizers for LLM hidden-state trajectories.

Public surface:
    stpp.geometry : the losses (official STP + curvature-budgeted variants)
    stpp.data     : chat-jsonl -> tokenised spans
    stpp.train    : single-GPU LoRA trainer
    stpp.pretrain : full-token-supervision harness (perplexity + gap test + MTP)
    stpp.evaluate : exact-match + collision diagnostics
    stpp.probe    : straight-line vs smooth-curve test
"""

__version__ = "0.1.0"

import os as _os
from pathlib import Path as _Path

# All machine-specific locations are env-overridable; these defaults only keep the
# original box working. Never hard-code an absolute path outside this block.
_HF_HOME = _os.environ.get("STPP_HF_HOME") or str(_Path.home() / ".cache" / "hf")
_os.environ.setdefault("HF_HOME", _HF_HOME)
if _os.environ.get("STPP_HF_ENDPOINT"):          # e.g. https://hf-mirror.com in CN
    _os.environ.setdefault("HF_ENDPOINT", _os.environ["STPP_HF_ENDPOINT"])
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
