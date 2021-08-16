from fastapi import FastAPI

app = FastAPI()

from dt import load_model, test_CLF

@app.get("/")
def read_root():
    PATH = "./test.pickle"
    clf_load = load_model(PATH)
    test_CLF(clf_load)
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}