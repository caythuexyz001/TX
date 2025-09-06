from app.analytics.markov import MarkovEngine

def test_markov_probs():
    mk = MarkovEngine(window=10)
    for y in "TTTXXXTTTX":
        mk.push(y)
    pT, pX = mk.probs('T')
    assert 0.0 <= pT <= 1.0 and 0.0 <= pX <= 1.0 and abs((pT+pX)-1.0) < 1e-9