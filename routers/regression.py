from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def test():
    return [{"classification": "200 OK"}]


@router.get("/linear")
async def decision_tree():
    return {"decisiontree": "decision tree"}
