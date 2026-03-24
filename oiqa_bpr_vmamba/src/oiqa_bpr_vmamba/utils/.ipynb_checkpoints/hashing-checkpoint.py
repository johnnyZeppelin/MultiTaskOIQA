from __future__ import annotations

import hashlib


def stable_int_hash(*parts: object, modulo: int = 2**31 - 1) -> int:
    text = '||'.join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    value = int.from_bytes(digest[:8], 'big', signed=False)
    return value % modulo
