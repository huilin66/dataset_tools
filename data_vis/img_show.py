from skimage import io
import matplotlib.pyplot as plt

def img_show(img_path):
    img = io.imread(img_path)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    img_show(r'E:\data\tp\sar_det\images\18-2_030.png')