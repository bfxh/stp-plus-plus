"""STP++ — geometric regularizers for LLM hidden-state trajectories.

Public surface:
    stpp.geometry : the losses (official STP + curvature-budgeted variants)
    stpp.data     : chat-jsonl -> tokenised spans
    stpp.train    : single-GPU LoRA trainer
    stpp.evaluate : exact-match + collision diagnostics
    stpp.probe    : straight-line vs smooth-curve test
"""

__version__ = "0.1.0"

import os as _os

# Model weights are multi-GB and the C: drive is nearly full on this box; these are
# only *defaults*, anything already exported wins.
_os.environ.setdefault("HF_HOME", r"F:\aiwork\hf")
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
