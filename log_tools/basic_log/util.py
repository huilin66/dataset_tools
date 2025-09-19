import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logging(log_file, show_level=logging.DEBUG, save_level=logging.INFO):
    """配置日志记录：控制台显示DEBUG，文件保存INFO"""
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s'
    )


    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=7,
        utc=False,
        encoding = 'utf-8'
    )
    file_handler.setLevel(save_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(show_level)
    console_handler.setFormatter(formatter)


    root_logger = logging.getLogger()
    root_logger.setLevel(show_level)
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)