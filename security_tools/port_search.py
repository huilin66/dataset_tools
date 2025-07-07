import socket
import socket
from concurrent.futures import ThreadPoolExecutor

def scan_port_single(ip, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((ip, port))
        if result == 0:
            print(f"Port {port} is open")
        sock.close()
    except Exception as e:
        print(f"Error scanning port {port}: {e}")

def scan_ports(ip, port_list, max_threads=100):
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for port in port_list:
            executor.submit(scan_port_single, ip, port)


def scan_port(ip, port):
    try:
        # 创建 socket 对象
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # 设置超时时间
        # 尝试连接
        result = sock.connect_ex((ip, port))
        if result == 0:
            print(f"Port {port} is open")
        else:
            print(f"Port {port} is closed")
        sock.close()
    except Exception as e:
        print(f"Error scanning port {port}: {e}")

# 目标 IP 和端口范围
target_ip = "18.166.73.93"
port_list = [
    22,     # 默认 SSH 端口
    2222,   # 常见的非标准 SSH 端口
    22222,  # 高位端口，常用于避免扫描
    2022,   # 另一种常见的非标准端口
    222,    # 简化的非标准端口
    2223,   # 类似 2222 的变体
    2224,   # 类似 2222 的变体
    2225,   # 类似 2222 的变体
    2226,   # 类似 2222 的变体
    2227,   # 类似 2222 的变体
    2228,   # 类似 2222 的变体
    2229,   # 类似 2222 的变体
    2200,   # 高位端口
]

# 扫描端口
for port in port_list:
    scan_port(target_ip, port)