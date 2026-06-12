import pandas as pd
from isds_tool.PS_data.att_tools import find_common_list
input_path1 = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1030/val_test.txt'
input_path2 = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1030/train_test.txt'
input1_df = pd.read_csv(input_path1, header=None, names=['file'])
input2_df = pd.read_csv(input_path2, header=None, names=['file'])
input1_list = input1_df['file'].to_list()
input2_list = input2_df['file'].to_list()

common_list = find_common_list(input1_list, input2_list)
print(common_list)