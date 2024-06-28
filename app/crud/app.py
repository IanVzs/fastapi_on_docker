
from loguru import logger
from route_class import TimedRoute, APIRouter

from typing import Dict, Any
from fastapi import Depends, HTTPException, Request

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound

from .database import async_session_install
from .models import CRUD

router = APIRouter(route_class=TimedRoute)


@router.get("/gaid/{gaid}/")
async def get_crud_by_gaid(gaid: str):
    logger.info(f"fetch by gaid: {gaid}")    
    rst = {}
    async with async_session_install() as session:
        try:
            exec = await session.execute(select(CRUD).where(CRUD.gaid == gaid))
            rst = exec.scalar_one()
        except NoResultFound:
            logger.warning(f"None fetched by gaid: {gaid}")
    return rst

@router.get("/")
async def read_crud(request: Request):
    filters = request.query_params
    results = list()
    logger.info(f"filters: {filters}")
    async with async_session_install() as session:        
        # 使用 CRUD 模型直接构建查询
        sql = select(CRUD)
        for key, value in filters.items():
            if not hasattr(CRUD, key):
                continue
            sql = sql.where(getattr(CRUD, key) == value)
        logger.warning(f"最终: {sql}")
        exec = await session.execute(sql)
        results = exec.scalars().all()

    return results
