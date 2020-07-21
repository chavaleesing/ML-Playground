import io
import os

from fastapi import APIRouter, Query, Response, Request
from starlette.responses import HTMLResponse

from services.classification import decision_tree as decision_tree_service
from services.classification import logistic_regression as logistic_regression_service
from services.classification import knn as knn_service


ROOT_DIR = os.path.abspath(os.curdir)
router = APIRouter()


@router.get("/")
async def test():
    return {"classification": "200 OK"}


@router.get("/decisiontree")
async def decision_tree(params: Request={}):
    response = decision_tree_service.demonstrate(params.query_params)
    if params.query_params.get("plot"):
        response = HTMLResponse(
        f'<html> \
            <h2><pre>{response}</pre></h2> \
            <img src="{ROOT_DIR}/temp_plt.png" alt="plt"/> \
        </html>'
        )
    return response

@router.get("/logisticregression")
async def logistic(params: Request={}):
    response = logistic_regression_service.demonstrate(params.query_params)
    if params.query_params.get("plot"):
        response = HTMLResponse(
        f'<html> \
            <h2><pre>{response}</pre></h2> \
            <img src="{ROOT_DIR}/temp_plt.png" alt="plt"/> \
        </html>'
        )
    return response

@router.get("/knn")
async def knn(params: Request={}):
    response = knn_service.demonstrate(params.query_params)
    if params.query_params.get("plot"):
        response = HTMLResponse(
        f'<html> \
            <h2><pre>{response}</pre></h2> \
            <img src="{ROOT_DIR}/temp_plt.png" alt="plt"/> \
        </html>'
        )
    return response
