from math import comb

def binom_cdf(k: int, n: int, p: float) -> float:
    # inclusive CDF: P(X <= k)
    if n <= 0:
        return 1.0
    s = 0.0
    for i in range(0, k+1):
        s += comb(n, i) * (p**i) * ((1-p)**(n-i))
    return min(max(s, 0.0), 1.0)