from dataclasses import dataclass


@dataclass
class Transaction:
    txn_id: str
    user_id: str
    amount: float
    ip_addr: str
    timestamp: float


def zigzag_encode(n: int, bits: int = 64) -> int:
    return (n << 1) ^ (n >> (bits - 1))


def varint_encode(x: int) -> bytes:
    """Encodes unsigned int x into ceil((floor(log2(x))+1)/7) Base-128 bytes."""
    out = bytearray()
    while x > 0x7F:
        out.append((x & 0x7F) | 0x80)
        x >>= 7
    out.append(x & 0x7F)
    return bytes(out)
