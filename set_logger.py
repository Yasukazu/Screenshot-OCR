import logging
import logging.handlers
import os


def set_logger():
    logger = logging.getLogger()

    streamHandler = logging.StreamHandler()
    if os.environ.get("DEBUG") == "1":
        streamHandler.setLevel(logging.DEBUG)
    if not os.listdir("./logs"):
        os.makedirs("./logs")
    fileHandler = logging.handlers.RotatingFileHandler(
        "./logs/app.log", maxBytes=1000000, backupCount=5, encoding="utf-8"
    )

    formatter = logging.Formatter(
        '%(asctime)s| %(levelname)-5s | %(name)s.%(funcName)s.%(lineno)d | %(message)s'
    )

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    streamHandler.setLevel(logging.INFO)
    fileHandler.setLevel(logging.DEBUG)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    return logger

