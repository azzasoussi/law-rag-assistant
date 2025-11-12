from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from rag_law import answer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "response": None})

@app.post("/", response_class=HTMLResponse)
def get_message(request: Request, user_message: str = Form(...)):
    response = answer(user_message)
    return templates.TemplateResponse("app.html", {"request": request, "response": response})

