import time
import logging
from util import log_source

@log_source(file="api_service.py", function="handle_request")
def infer():
    logging.debug("ONNX model loaded")  # 日志自动标记模块名 [onnx_deployment]
    time.sleep(1)
    logging.debug("ONNX model infer finished")  # 日志自动标记模块名 [onnx_deployment]