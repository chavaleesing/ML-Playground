from fastapi import FastAPI
from routers import classification, regression

app = FastAPI()

app.include_router(classification.router, prefix="/classification")
app.include_router(regression.router, prefix="/regression")
