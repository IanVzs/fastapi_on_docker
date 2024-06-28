from loguru import logger

# logger.add('activity_{time}.log',
#            level="INFO",
#            rotation='5 MB',
#            encoding='utf-8',
#            compression="zip")
recoder_fmt = "{time:HH:mm:ss.SSS} | {level} - {message}"
# logger.add(sys.stdout, level="DEBUG")
logger.add(
    "./logs/app_{time:YYYY-MM-DD}.log",
    format=recoder_fmt,
    level="INFO",
    rotation="00:00",
    encoding="utf-8",
    compression="zip",
)
