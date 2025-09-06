import re

def is_valid_md5(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{32}", s or ""))

def is_valid_dice(d: int) -> bool:
    return isinstance(d, int) and 1 <= d <= 6