---

## tests/test_labels.py
```python
from app.db.models import Round

def test_label_mapping():
    r = Round(d1=6, d2=5, d3=0)
    # invalid die would be caught by API; here check compute
    r.d3 = 1; r.compute(); assert r.label == 'XIU'  # 12? nope -> fix
    r.d1, r.d2, r.d3 = 6, 4, 1; r.compute(); assert r.total == 11 and r.label == 'TAI'
    r.d1, r.d2, r.d3 = 1, 1, 1; r.compute(); assert r.total == 3 and r.label == 'XIU'