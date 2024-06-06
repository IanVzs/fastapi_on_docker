"""
MYSQL_USER=root MYSQL_PASSWD=passwd MYSQL_HOST=localhost MYSQL_PORT=3316 MYSQL_DB=crud MYSQL_TABLE=crud_current python app/scripts/database_test.py 
"""
import os
from loguru import logger
from sqlalchemy import create_engine, text

recoder_fmt = "{time:HH:mm:ss.SSS} | {level} - {message}"

logger.remove()
logger.add(
    "./logs/sqltest_{time:YYYY-MM-DD}.log",
    format=recoder_fmt,
    level="INFO",
    rotation="00:00",
    encoding="utf-8",
    compression="zip",
)

SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{os.environ.get("MYSQL_USER")}:{os.environ.get("MYSQL_PASSWD")}@{os.environ.get("MYSQL_HOST")}:{os.environ.get("MYSQL_PORT")}/{os.environ.get("MYSQL_DB")}'
logger.info(f"sql info: {SQLALCHEMY_DATABASE_URI}")
engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_recycle=1500)

def select():
    with engine.connect() as conn:
        result_proxy = conn.execute(text(f'select * from {os.environ.get("MYSQL_TABLE")} LIMIT 2'))
        result = result_proxy.fetchall()
        print(f"result: len({len(result)})")
        print(f"result: {result}")

if __name__ == "__main__":
    select()
