import os
from skimage import io


# root_dir = r'/scrinvme/huilin/bdd/cp_data/C2Seg/src/C2Seg_BW/train'
root_dir = r'/scrinvme/huilin/bdd/cp_data/C2Seg/src/C2Seg_AB/train'
hsi_dir = os.path.join(root_dir, 'hsi')
msi_dir = os.path.join(root_dir, 'msi')
sar_dir = os.path.join(root_dir, 'sar')
label_dir = os.path.join(root_dir, 'label')

file_name = os.listdir(hsi_dir)[0]

hsi_path = os.path.join(hsi_dir, file_name)
msi_path = os.path.join(msi_dir, file_name)
sar_path = os.path.join(sar_dir, file_name)
label_path = os.path.join(label_dir, file_name)
hsi = io.imread(hsi_path)
msi = io.imread(msi_path)
sar = io.imread(sar_path)
label = io.imread(label_path)
print('hsi shape: ', hsi.shape)
print('msi shape: ', msi.shape)
print('sar shape: ', sar.shape)
print('label shape: ', label.shape)
