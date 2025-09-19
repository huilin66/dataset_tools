import logging
import infer2
from util import auto_log_source

@auto_log_source()
def process_data():
    logging.info("data processing started")  # 日志自动标记模块名 [cdu_core]
    infer2.infer()
    logging.info("data processing end")  # 日志自动标记模块名 [cdu_core]