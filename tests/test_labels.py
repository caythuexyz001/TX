---

## tests/test_labels.py
```python
from app.db.models import Round

def test_label_mapping():
    r = Round(d1=6, d2=5, d3=1); r.compute(); assert r.total == 12 and r.label == 'TAI'  # 12 → TAI
    r = Round(d1=4, d2=3, d3=3); r.compute(); assert r.total == 10 and r.label == 'XIU'  # 10 → XIU
    r = Round(d1=5, d2=5, d3=1); r.compute(); assert r.total == 11 and r.label == 'TAI'  # 11 → TAI
    r = Round(d1=1, d2=1, d3=1); r.compute(); assert r.total == 3  and r.label == 'XIU'  # 3 → XIU
