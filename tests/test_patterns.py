from app.analytics.patterns import runs, alternations, blocks

def test_runs():
    assert runs("TTTXX", k=3) == [(0,2,'T',3)]

def test_alt():
    out = alternations("TXTXTX", L=4)
    assert out and out[0][0] == 0

def test_blocks():
    out = blocks("TTXXTTXX")
    assert out and out[0][2] in ("TTXX","XXTT")