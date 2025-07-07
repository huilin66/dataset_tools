import os
import shutil
from tqdm import tqdm

def file_num_show(input_dir):
    print(len(os.listdir(input_dir)))

def copy_files(input_dir, output_dir):
    input_list = os.listdir(input_dir)
    for file_name in tqdm(input_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        if os.path.exists(output_path):
            print(output_path)
        shutil.copyfile(input_path, output_path)

if __name__ == '__main__':
    # for i in range(10):
    #     file_num_show(r'G:\alos_dem\30m_alos_dem%02d'%i)
    #
    # print('-'*10)
    # file_num_show(r'G:\alos_dem\30m_alos_dem008')
    # file_num_show(r'G:\alos_dem\30m_alos_dem009')
    # file_num_show(r'G:\alos_dem\30m_alos_dem010')
    # file_num_show(r'G:\alos_dem\30m_alos_dem_8')
    # file_num_show(r'G:\alos_dem\30m_alos_dem_888')

    # copy_files(r'G:\alos_dem\30m_alos_dem008', r'G:\alos_dem\30m_alos_dem00')
    # copy_files(r'G:\alos_dem\30m_alos_dem009', r'G:\alos_dem\30m_alos_dem00')

    # shutil.rmtree(r'G:\alos_dem\30m_alos_dem_8')
    # print(len(os.listdir(r'H:\also_demo\04')))
    # print(len(os.listdir(r'H:\also_demo\05')))
    # print(len(os.listdir(r'H:\also_demo\06')))
    # print(len(os.listdir(r'H:\also_demo\08')))
    # print(len(os.listdir(r'H:\also_demo\09')))
    print(len(os.listdir(r'G:\alos_dem\02')))
    print(len(os.listdir(r'G:\alos_dem\03')))
    print(len(os.listdir(r'G:\alos_dem\07')))