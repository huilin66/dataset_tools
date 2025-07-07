import time
import serial
import minimalmodbus
from datetime import datetime

'''
1: noise,   0.1dB
2: HCHO,    0.01ppm
3: TVOC,    1ppb(*1000)
4: NH3,     1ppm(*10000)
5: smell
'''

# 配置串口
SERIAL_PORT = 'COM4'  # 根据你的实际串口设备名称
BAUD_RATE = 4800  # 根据你的设备配置

# 初始化Modbus设备
instrument1 = minimalmodbus.Instrument(SERIAL_PORT, 1)  # 第一个参数是串口名称，第二个参数是设备地址
instrument1.serial.baudrate = BAUD_RATE
instrument1.serial.bytesize = 8
instrument1.serial.parity = serial.PARITY_NONE
instrument1.serial.stopbits = 1
instrument1.serial.timeout = 1  # 超时时间

instrument2 = minimalmodbus.Instrument(SERIAL_PORT, 2)  # 第一个参数是串口名称，第二个参数是设备地址
instrument2.serial.baudrate = BAUD_RATE
instrument2.serial.bytesize = 8
instrument2.serial.parity = serial.PARITY_NONE
instrument2.serial.stopbits = 1
instrument2.serial.timeout = 1  # 超时时间

instrument3 = minimalmodbus.Instrument(SERIAL_PORT, 3)  # 第一个参数是串口名称，第二个参数是设备地址
instrument3.serial.baudrate = BAUD_RATE
instrument3.serial.bytesize = 8
instrument3.serial.parity = serial.PARITY_NONE
instrument3.serial.stopbits = 1
instrument3.serial.timeout = 1  # 超时时间

instrument4 = minimalmodbus.Instrument(SERIAL_PORT, 4)  # 第一个参数是串口名称，第二个参数是设备地址
instrument4.serial.baudrate = BAUD_RATE
instrument4.serial.bytesize = 8
instrument4.serial.parity = serial.PARITY_NONE
instrument4.serial.stopbits = 1
instrument4.serial.timeout = 1  # 超时时间


while True:
    try:
        # 使用功能码03读取保持寄存器（以寄存器地址40001为例）
        register_address = 0  # 40001寄存器的地址在Modbus协议中是0
        register_value1 = instrument1.read_register(register_address, 2)  # 读取一个寄存器
        register_value2 = instrument2.read_register(register_address, 2)  # 读取一个寄存器
        register_value3 = instrument3.read_register(register_address, 2)  # 读取一个寄存器
        register_value4 = instrument4.read_register(register_address, 2)  # 读取一个寄存器

        # register_value1 *= 1
        # register_value2 *= 0.1
        # register_value3 *= 10
        # register_value4 *= 1

        register_value1 *= 10
        register_value2 *= 1
        register_value3 *= 100
        register_value4 *= 10

        # 读取时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print('%s, detection result: Noise %g dB, HCHO %g ppm, TVOC %g ppm, NH3 %g ppm' %
              (current_time, register_value1, register_value2, register_value3, register_value4))

        # 写入日志文件
        log_file = open('log1_Noise.txt', 'a')
        log_file.write('%s %s\n' % (current_time, str(register_value1)))
        log_file = open('log2_HCHO.txt', 'a')
        log_file.write('%s %s\n' % (current_time, str(register_value2)))
        log_file = open('log3_TVOC.txt', 'a')
        log_file.write('%s %s\n' % (current_time, str(register_value3)))
        log_file = open('log4_NH3.txt', 'a')
        log_file.write('%s %s\n' % (current_time, str(register_value4)))

        # 等待1秒
        time.sleep(1)

    except Exception as e:
        print(f"读取失败: {e}")
