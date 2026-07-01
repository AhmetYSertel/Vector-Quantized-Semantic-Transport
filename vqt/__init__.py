"""VQT — Vector-Quantized Semantic Transport.

A discrete, text-API-compatible control-plane channel for LLM agents.

Sender:   message --embed--> vector --VQ--> integer code (37-byte frame)
Receiver: code --cache?--> HIT: cached call (LLM skipped)
                        --MISS: dictionary[code] -> NL label -> text-API LLM
"""
from .encoders import Encoder, RealEncoder, FakeEncoder
from .codebook import Codebook
from .dictionary import Dictionary
from .cache import SemanticCache
from .frame import pack_frame, unpack_frame, FRAME_BYTES
from .experiment import run, ExperimentResult

__all__ = [
    "Encoder", "RealEncoder", "FakeEncoder",
    "Codebook", "Dictionary", "SemanticCache",
    "pack_frame", "unpack_frame", "FRAME_BYTES",
    "run", "ExperimentResult",
]
__version__ = "0.1.0"
