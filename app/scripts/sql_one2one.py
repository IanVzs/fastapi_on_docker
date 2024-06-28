"""
USER=root PASSWD=passwd SHOST=localhost THOST=localhost SPORT=3306 TPORT=3316 DB=curd TABLE=crud_current python app/scripts/sql_one2one.py 
"""
import os
import re
import time
from tqdm import tqdm
from loguru import logger

from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.exc import IntegrityError, OperationalError

# 定义数据库连接字符串
USER = os.environ['USER']
PASSWD = os.environ['PASSWD']
source_host = os.environ["SHOST"]
target_host = os.environ["THOST"]
SPORT = os.environ['SPORT']
TPORT = os.environ['TPORT']
DB = os.environ['DB']
TABLE = os.environ['TABLE']
LIMIT = os.environ.get('LIMIT', None)
OFFSET = os.environ.get('OFFSET')

source_connection_string = f"mysql+pymysql://{USER}:{PASSWD}@{source_host}:{SPORT}/{DB}"
target_connection_string = f"mysql+pymysql://{USER}:{PASSWD}@{target_host}:{TPORT}/{DB}"



# 创建源数据库连接
source_engine = create_engine(source_connection_string)
# 创建目标数据库连接
target_engine = create_engine(target_connection_string)

logger.warning(f"由{source_connection_string} 向 {target_connection_string}写入数据!")
time.sleep(1)


# 创建源数据库的元数据
source_metadata = MetaData()
source_metadata.reflect(bind=source_engine)

# 创建目标数据库的元数据
target_metadata = MetaData()
target_metadata.reflect(bind=target_engine)

# 获取源数据库中的表
source_table = source_metadata.tables[TABLE]

# 获取目标数据库中的表
target_table = target_metadata.tables[TABLE]

# 建立数据库连接
with source_engine.connect() as source_conn, target_engine.connect() as target_conn:
    # 从源数据库中选择数据
    if LIMIT is None:
        source_query = source_table.select()
    else:
        if 1:
            source_query = source_table.select().where(source_table.c.install_at > "2024-06-12 00:00:00", source_table.c.offer_id == 18723).limit(LIMIT).offset(OFFSET)
        else:
            source_query = source_table.select().where().limit(LIMIT).offset(OFFSET)
        logger.info(f"sql: {source_query}")
    result = source_conn.execute(source_query)
    
    # 逐行插入到目标数据库
    for row in tqdm(result):
        try:
            target_conn.execute(target_table.insert().values(row))
        except IntegrityError as err:
            if re.search('Duplicate entry.*key.*PRIMARY', str(err)):
                pass
            else:
                logger.info(f"插入错误|{err}, {row[0]}")
        except OperationalError as err:
            _continue = False
            for i in row:
                if i == "0000-00-00 00:00:00":
                    logger.warning(f"数据错误|{err}, {row[0]}")
                    _continue = True
                    break
            if _continue:
                continue
            logger.info(f"插入错误|{err}, {row[0]}")

    target_conn.commit()
