"""
MYSQL_USER=root MYSQL_PASSWD=nova#2021 MYSQL_HOST=localhost MYSQL_PORT=3316 MYSQL_DB=crud MYSQL_TABLE="crud_current" python app/scripts/sql_tools.py 
"""
import os
from loguru import logger
from sqlalchemy import create_engine, text

recoder_fmt = "{time:HH:mm:ss.SSS} | {level} - {message}"

logger.remove()
logger.add(
    "./logs/app_{time:YYYY-MM-DD}.log",
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
        # 返回值为ResultProxy类型
        _min = 104763520
        real_max = 132000000
        step = 3000
        list_datas = []
        all_data = {}
        aaa = 0
        while 1:
            _max = _min + step
            print(f"min: {_min}, max: {_max}")
            aaa += 1
            result_proxy = conn.execute(text(f'select offer_id, executes, updated_at from {os.environ.get("MYSQL_TABLE")} where pkg_name = "com.zhiliaoapp.musically" and id > {_min} and id < {_max}'))
            result = result_proxy.fetchall()
            print(f"result: len({len(result)})")
            for i in result:
                from analyze_tools import analyze, draw
                data = analyze(i)
                for k, v in data.items():
                    data[k] = all_data.get(k, 0) + v
                all_data.update(data)
                if aaa > 5:
                    draw(all_data)
                    aaa = 0
            _min = _max
            if _max >= real_max:
                break

if __name__ == "__main__":
    select()
