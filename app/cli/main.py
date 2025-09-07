import typer
import requests
import os


app = typer.Typer()
BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY")




def _headers():
h = {}
if API_KEY:
h["X-API-Key"] = API_KEY
return h


@app.command()
def ingest(d1: int, d2: int, d3: int, md5: str = typer.Option(None)):
r = requests.post(f"{BASE}/ingest", json={"d1": d1, "d2": d2, "d3": d3, "md5": md5}, headers=_headers())
typer.echo(r.json())


@app.command()
def stats(window: int = 50):
r = requests.get(f"{BASE}/stats", headers=_headers())
typer.echo(r.json())


@app.command()
def predict(md5: str = typer.Option(None)):
r = requests.get(f"{BASE}/predict", params={"md5": md5} if md5 else {}, headers=_headers())
typer.echo(r.json())


if __name__ == "__main__":
app()
