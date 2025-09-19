import logging
from logging.handlers import TimedRotatingFileHandler
import inspect
from functools import wraps
from pathlib import Path

class SourceAwareFormatter(logging.Formatter):
    """支持动态覆盖文件/函数名的格式化器"""

    def format(self, record):
        # 自动捕获调用位置（如果不手动指定）
        if not hasattr(record, 'file'):
            frame = inspect.currentframe()
            while frame:
                if frame.f_code.co_name == record.funcName:
                    record.file = frame.f_code.co_filename.split('/')[-1]  # 取短文件名
                    break
                frame = frame.f_back

        # 保留手动设置的函数名或使用默认
        record.function = getattr(record, 'function', record.funcName)
        return super().format(record)


def setup_logging(log_file, show_level=logging.DEBUG, save_level=logging.INFO):
    """配置日志：控制台DEBUG+，文件INFO+，每日轮转"""
    formatter = SourceAwareFormatter(
        '[%(asctime)s] [%(levelname)s] [%(file)s:%(function)s] - %(message)s'
    )

    # 文件处理器（每日轮转，保留7天）
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setLevel(save_level)  # 文件保存INFO+
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(show_level)  # 控制台显示DEBUG+
    console_handler.setFormatter(formatter)

    # 配置根Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 必须设为最低级别

    # 避免重复配置
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)


def log_source(file=None, function=None):
    """装饰器动态修改日志来源"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger()

            # 临时修改日志记录工厂
            old_factory = logger.makeRecord

            def custom_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                if file:
                    record.file = file
                if function:
                    record.function = function
                return record

            logger.makeRecord = custom_factory
            result = func(*args, **kwargs)
            logger.makeRecord = old_factory  # 恢复原工厂

            return result

        return wrapper

    return decorator


def auto_log_source(func=None, *, file=None, function=None):
    """
    自动捕获调用位置的装饰器
    用法：
    1. @auto_log_source  # 全自动
    2. @auto_log_source(file="custom.py")  # 半自动（只覆盖文件）
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # 获取调用栈信息
            frame = inspect.currentframe().f_back
            file_name = file or frame.f_code.co_filename.split('/')[-1]
            file_name = Path(file_name).name
            func_name = function or f.__name__

            # 动态标记日志来源
            logger = logging.getLogger()
            old_factory = logger.makeRecord

            def custom_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.file = file_name
                record.function = func_name
                return record

            logger.makeRecord = custom_factory
            result = f(*args, **kwargs)
            logger.makeRecord = old_factory  # 恢复

            return result

        return wrapper

    # 处理直接 @auto_log_source 和 @auto_log_source() 两种情况
    if func is None:
        return decorator
    else:
        return decorator(func)