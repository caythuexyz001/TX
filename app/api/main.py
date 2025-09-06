from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from app.db.base import init_db
from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="TaiXiu Analyzer", lifespan=lifespan)
app.include_router(router)

app.mount("/static", StaticFiles(directory="app/ui/static"), name="static")
templates = Jinja2Templates(directory="app/ui/templates")

@app.get("/")
def home():
    return {"ok": True, "app": "TaiXiu Analyzer"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/md5", response_class=HTMLResponse)
async def md5_page(request: Request):
    return templates.TemplateResponse("md5.html", {"request": request})
