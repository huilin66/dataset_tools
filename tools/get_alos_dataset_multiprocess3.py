from get_alos_dataset_multiprocess import *

if __name__ == '__main__':
    print('cpu cores:', os.cpu_count())

    dem_unsorted_list = os.listdir(DATASET_PATH)
    dem_list = sorted(dem_unsorted_list, key=lambda path: int(os.path.basename(path).split('.')[0]))
    dem_lists = np.array_split(dem_list, 20)
    for i in range(15, 21):
        multi_process_alos(dem_lists[i])