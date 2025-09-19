from util import setup_logging, auto_log_source
import infer1
import logging
# 初始化全局日志
setup_logging(log_file='0910.log')
@auto_log_source()
def main():
    logging.info("main start")  # 日志自动标记模块名 [onnx_deployment]

    infer1.process_data()

if __name__ == '__main__':
    main()