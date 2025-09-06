from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.db.base import init_db
from app.api.routes import router

app = FastAPI(title="TaiXiu Analyzer")
app.include_router(router)

# UI assets
app.mount("/static", StaticFiles(directory="app/ui/static"), name="static")
templates = Jinja2Templates(directory="app/ui/templates")

@app.on_event("startup")
async def _startup():
    init_db()

@app.get("/")
def home():
    return {"ok": True, "app": "TaiXiu Analyzer"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})