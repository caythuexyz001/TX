from collections import Counter

# Minimal, fast MD5 feature extractor (no crypto meaning; just structure)
# You can extend with your own features as needed.

def md5_features(md5_hash: str) -> dict:
    md5_hash = md5_hash.lower()
    counts = Counter(md5_hash)
    digits = sum(counts[c] for c in '0123456789')
    letters = 32 - digits
    even_hex = sum(counts[c] for c in '02468ace')
    odd_hex = 32 - even_hex
    hex_sum = sum(int(md5_hash[i:i+2], 16) for i in range(0, 32, 2))
    return {
        'digits': digits,
        'letters': letters,
        'even_hex': even_hex,
        'odd_hex': odd_hex,
        'byte_sum_mod_100': hex_sum % 100,
    }