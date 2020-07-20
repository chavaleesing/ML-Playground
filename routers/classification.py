from fastapi import APIRouter, Response
import io
from starlette.responses import HTMLResponse

from services.classification import decision_tree as decision_tree_service
from services.classification import logistic_regression as logistic_regression_service

router = APIRouter()


@router.get("/")
async def test():
    return [{"classification": "200 OK"}]


@router.get("/decisiontree")
async def decision_tree():
    result = decision_tree_service.classify()
    return {"decisiontree": "decision tree", "result": result}

@router.get("/logistic")
async def logistic():
    import os
    ROOT_DIR = os.path.abspath(os.curdir)
    cv = logistic_regression_service.classify()
    cv.print_png('temp_plt.png')
    response = HTMLResponse(
        f'<html> \
            <p>show plt .....</p> \
            <body><img src="{ROOT_DIR}/temp_plt.png" alt="plt"></body> \
        </html>'
        )
    return response