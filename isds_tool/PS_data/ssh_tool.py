from pyputty.key import PPKKey
from paramiko import RSAKey

# 读取 PPK 文件
ppk = PPKKey.from_file(r"E:\data\202502_signboard\data_annotation\docs\dsds_key.ppk")

# 转换为 Paramiko 可用的 RSAKey
key = RSAKey.from_private_key_file(ppk.to_openssh_private_key())