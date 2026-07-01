"""Fixed 37-byte wire frame. Fixed size enables pre-allocated receive
buffers and direct indexing. The version field is what makes silent
codebook-mismatch corruption detectable (see paper Sec. on versioning)."""
from __future__ import annotations
import struct

# <I code | f qerr | f margin | H version | H source | I ts | I meta>
_LAYOUT = "<IffHHII"
_BODY = struct.calcsize(_LAYOUT)      # 24 bytes
FRAME_BYTES = 37
_PAD = FRAME_BYTES - _BODY


def pack_frame(code, qerr, margin, version, source_id=0, ts=0, meta=0) -> bytes:
    body = struct.pack(
        _LAYOUT,
        int(code) & 0xFFFFFFFF, float(qerr), float(margin),
        int(version) & 0xFFFF, int(source_id) & 0xFFFF,
        int(ts) & 0xFFFFFFFF, int(meta) & 0xFFFFFFFF,
    )
    return body + b"\x00" * _PAD


def unpack_frame(frame: bytes) -> dict:
    if len(frame) != FRAME_BYTES:
        raise ValueError(f"expected {FRAME_BYTES} bytes, got {len(frame)}")
    code, qerr, margin, version, source_id, ts, meta = struct.unpack(
        _LAYOUT, frame[:_BODY])
    return {"code": code, "qerr": qerr, "margin": margin, "version": version,
            "source_id": source_id, "ts": ts, "meta": meta}
