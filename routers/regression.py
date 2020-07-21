import os

from fastapi import APIRouter, Request
from starlette.responses import HTMLResponse

from services.regression import knn as knn_service
from services.regression.linear_regression import least_square as least_square_service
from services.regression.linear_regression import ridge_regression as ridge_regression_service

ROOT_DIR = os.path.abspath(os.curdir)
router = APIRouter()

@router.get("/")
async def test():
    return [{"regression": "200 OK"}]


@router.get("/knn")
async def knn(params: Request = {}):
    response = knn_service.demonstrate(params.query_params)
    if params.query_params.get("plot"):
        response = HTMLResponse(
        f'<html> \
            <h2><pre>{response}</pre></h2> \
            <img src="{ROOT_DIR}/temp_plt.png" alt="plt"/> \
        </html>'
        )
    return response

@router.get("/leastsquare")
async def knn(params: Request = {}):
    response = least_square_service.demonstrate(params.query_params)
    if params.query_params.get("plot"):
        response = HTMLResponse(
        f'<html> \
            <h2><pre>{response}</pre></h2> \
            <img src="{ROOT_DIR}/temp_plt.png" alt="plt"/> \
        </html>'
        )
    return response


@router.get("/ridge")
async def knn(params: Request = {}):
    response = ridge_regression_service.demonstrate(params.query_params)
    if params.query_params.get("plot"):
        response = HTMLResponse(
        f'<html> \
            <h2><pre>{response}</pre></h2> \
            <img src="{ROOT_DIR}/temp_plt.png" alt="plt"/> \
        </html>'
        )
    return response

