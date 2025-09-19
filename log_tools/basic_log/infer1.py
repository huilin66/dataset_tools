import logging
import infer2

def process_data():
    logging.info("data processing started")  # 日志自动标记模块名 [cdu_core]
    infer2.infer()
    logging.info("data processing end")  # 日志自动标记模块名 [cdu_core]