from fastapi import FastAPI

from crud import app as crud_app

def creat_app():
    app = FastAPI()
    app.include_router(crud_app.router, prefix="/crud")
    return app

"""
解决跨域/官方demo姑且放在这儿
from starlette.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
