from typing import Iterable

def runs(labels: Iterable[str], k: int = 3):
    labels = list(labels)
    out = []
    if not labels:
        return out
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] == cur:
            continue
        # close segment
        seg_len = i - start
        if seg_len >= k:
            out.append((start, i-1, cur, seg_len))
        cur = labels[i]
        start = i
    # tail
    seg_len = len(labels) - start
    if seg_len >= k:
        out.append((start, len(labels)-1, cur, seg_len))
    return out

def alternations(labels: Iterable[str], L: int = 4):
    labels = list(labels)
    out = []
    if len(labels) < 2:
        return out
    start = None
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            if start is None:
                start = i-1
        else:
            if start is not None and i - start + 1 >= L:
                out.append((start, i-1))
            start = None
    if start is not None and len(labels) - start >= L:
        out.append((start, len(labels)-1))
    return out

def blocks(labels: Iterable[str]):
    # find (TTXX)+ or (XXTT)+ segments
    labels = ''.join(labels)
    out = []
    i = 0
    while i + 3 < len(labels):
        chunk = labels[i:i+4]
        if chunk in ("TTXX", "XXTT"):
            j = i + 4
            while j + 3 < len(labels) and labels[j:j+4] == chunk:
                j += 4
            out.append((i, j-1, chunk, (j-i)//4))
            i = j
        else:
            i += 1
    return out