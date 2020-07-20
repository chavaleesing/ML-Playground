import os
import sys

import uvicorn
from fastapi import FastAPI

from routers import classification, regression

ROOT_DIR = os.path.abspath(os.curdir)
sys.path.append(ROOT_DIR)
app = FastAPI()

app.include_router(classification.router, prefix="/classification")
app.include_router(regression.router, prefix="/regression")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
