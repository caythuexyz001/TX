from collections import deque
last_label: str | None
p_value_row: dict[str, float]
entropy: float


class MarkovEngine:
def __init__(self, window: int = 50, alpha: float = 1.0):
self.W = window
self.a = alpha
self.buf = deque(maxlen=window) # 'T'/'X'
self.C = {'T': {'T': 0, 'X': 0}, 'X': {'T': 0, 'X': 0}}


def reset(self):
self.buf.clear()
self.C = {'T': {'T': 0, 'X': 0}, 'X': {'T': 0, 'X': 0}}


def push(self, y: str):
if self.buf:
prev = self.buf[-1]
self.C[prev][y] += 1
self.buf.append(y)


def build_from(self, labels: list[str]):
self.reset()
for y in labels:
self.push(y)


def probs(self, last: str | None) -> tuple[float, float]:
def row(i: str):
nT = self.C[i]['T']; nX = self.C[i]['X']
pT = (nT + self.a) / (nT + nX + 2*self.a)
return pT, 1 - pT
if last in ('T', 'X'):
pT, pX = row(last)
else:
t = sum(self.C['T'].values()) + sum(self.C['X'].values())
if t == 0:
return 0.5, 0.5
mT = self.C['T']['T'] + self.C['X']['T']
pT = (mT + self.a) / (t + 2*self.a)
pX = 1 - pT
return pT, pX


def stats(self, last: str | None) -> MarkovStats:
# probs
pTT = (self.C['T']['T'] + self.a) / (self.C['T']['T'] + self.C['T']['X'] + 2*self.a)
pTX = 1 - pTT
pXT = (self.C['X']['T'] + self.a) / (self.C['X']['T'] + self.C['X']['X'] + 2*self.a)
pXX = 1 - pXT
# entropy on marginal within window
nT = sum(1 for v in self.buf if v == 'T')
nX = len(self.buf) - nT
import math
H = 0.0
for c in (nT, nX):
if c == 0:
continue
p = c / max(len(self.buf), 1)
H -= p * math.log(p, 2)
# p-values per row vs 0.5
rows = {
'T': (self.C['T']['T'], self.C['T']['X']),
'X': (self.C['X']['T'], self.C['X']['X'])
}
pvals = {}
for i, (t_cnt, x_cnt) in rows.items():
n = t_cnt + x_cnt
k = max(t_cnt, x_cnt)
if n == 0:
pvals[i] = 1.0
else:
pv = 2 * min(binom_cdf(k, n, 0.5), 1 - binom_cdf(k-1, n, 0.5))
pvals[i] = max(min(pv, 1.0), 0.0)
return MarkovStats(
transition=[[pTT, pTX], [pXT, pXX]],
counts=[[self.C['T']['T'], self.C['T']['X']], [self.C['X']['T'], self.C['X']['X']]],
last_label=last,
p_value_row=pvals,
entropy=H,
)
