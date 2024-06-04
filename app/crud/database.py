import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine


SQLALCHEMY_DATABASE_URL = f'mysql+aiomysql://{os.environ.get("MYSQL_CRUD_USER")}:{os.environ.get("MYSQL_CRUD_PASSWD")}@{os.environ.get("MYSQL_CRUD_HOST")}:{os.environ.get("MYSQL_CRUD_PORT")}/{os.environ.get("MYSQL_CRUD_DB")}'
async_engine = create_async_engine(SQLALCHEMY_DATABASE_URL, pool_recycle=1500)
async_session_install = sessionmaker(async_engine, class_=AsyncSession)

Base = declarative_base()
