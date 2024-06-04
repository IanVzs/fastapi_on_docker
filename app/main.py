import aiohttp
from fastapi import FastAPI, Request, Response

import app_setting
#from lib import wx_api

app = app_setting.creat_app()

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

@app.get("/")
async def read_main():
    async with aiohttp.ClientSession() as session:
        a = await fetch(session, 'http://127.0.0.1:8099/favicon.ico')
    return {"msg": "app v0.10"}

@app.get("/favicon.ico")
async def favicon_ico():
    return "app v0.10"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8099, reload=True)
