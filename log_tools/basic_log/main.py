from util import setup_logging
import infer1
'''
级别

数值

用途

适用场景

DEBUG

10

最详细的调试信息

变量值、函数调用细节、算法执行步骤

INFO

20

确认程序正常运行

程序启动、重要操作完成、业务关键节点

WARNING

30

可能的问题警告

磁盘空间不足、网络延迟、配置默认值

ERROR

40

严重的功能错误

数据库连接失败、文件读取错误、API调用失败

CRITICAL

50

最严重的系统错误

系统资源耗尽、关键组件失败
'''
# 初始化全局日志
setup_logging(log_file='0910.log')

# 调用其他模块
infer1.process_data()