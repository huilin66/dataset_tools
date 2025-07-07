import numpy as np
import cv2
import tifffile as tff
import os
# import tensorflow as tf
# from network_architecture import unet
import mpmath
import decimal

NPY_FILE_PATH = r"G:\Albedo\Lunar_albedo\NAC_Images\NAC_albedo_dataset_img\data_record.npy"
def binary_float(num, length):
    """
    Converts a decimal number to binary string representation with given precision
    """
    decimal.getcontext().prec = length + 1  # add one bit to ensure correct rounding
    bits = bin(int(num))[2:] + "."
    num = num - int(num)
    while num > 0:
        num *= 2
        if num >= 1:
            bits += "1"
            num -= 1
        else:
            bits += "0"
        if len(bits) >= length:
            break
    if len(bits) < length:
        bits += "0" * (length - len(bits))
    return bits

def encoding(num, length):
    mpmath.mp.dps = length / 2

    a = mpmath.mpf(str(num))

    sin_a = mpmath.sin(mpmath.radians(a) / (a + 1e-5))
    cos_a = mpmath.cos(mpmath.radians(a) / (a + 1e-5))

    bin_sin_a = binary_float(sin_a, int(length / 2)).replace(".", "2")
    bin_cos_a = binary_float(cos_a, int(length / 2)).replace(".", "2").replace("b", "3")

    a_vec = bin_sin_a + bin_cos_a

    a_vec = np.float32(np.fromstring(a_vec, dtype="S1"))
    return a_vec

def image_augmentation(image, dem, albedo, sun_azimuth):
    rotation_angle = np.random.randint(0, 4, 1)
    image = np.rot90(image, rotation_angle)
    dem = np.rot90(dem, rotation_angle)
    albedo = np.rot90(albedo, rotation_angle)
    sun_azimuth = float(sun_azimuth) + 90 * rotation_angle
    if sun_azimuth > 360.:
        sun_azimuth = sun_azimuth - 360
    sun_azimuth = float(sun_azimuth)
    return image, dem, albedo, sun_azimuth

def get_illumination_vector_real(azimuth,elevation): #get illumination/viewing vector from azimuth and zenith angles in degrees
    theta = np.radians(azimuth); phi = np.radians(elevation)
    c_azi, s_azi = np.cos(theta), np.sin(theta)
    c_ele, s_ele = np.cos(phi), np.sin(phi)
    R_illu = np.array(((c_azi, -s_azi, 0), (s_azi, c_azi, 0),(0, 0, 1.0)))
    illumination_vector = np.matmul(R_illu,np.array([[0],[-c_ele],[s_ele]]))
    return illumination_vector

def get_normal(dem, resolution):
    #compute surface normal based on 2x2 grid
    p = dem[:,1:]-dem[:,0:-1] ; p = (0.5*p[0:-1,:]+0.5*p[1:,:])/ resolution
    q = dem[1:,:]-dem[0:-1,:]; q = (0.5*q[:,0:-1]+0.5*q[:,1:])/ resolution
    return p, q

def lambertian_surface(dem, resolution, azimuth, elevation_angle):
    illumination_vec =  np.matrix(get_illumination_vector_real(azimuth, elevation_angle))

    zero = np.zeros((dem.shape[0], dem.shape[1], 1))
    one = np.ones((dem.shape[0], dem.shape[1], 1))

    gx, gy = get_normal(dem, resolution)
    gx = np.pad(gx, ((0, 1), (0, 1), (0, 0)), 'constant')
    gy = np.pad(gy, ((0, 1), (0, 1), (0, 0)), 'constant')

    # gx =  cv2.Sobel(dem,-1,1,0,ksize=3)
    gx = np.reshape(gx, [gx.shape[0], gx.shape[1], 1])
    vec_x = np.concatenate((one, zero, gx), axis=2)

    # gy =  cv2.Sobel(dem,-1,0,1,ksize=3)
    # shape_gy = np.array(np.shape(gy))
    gy = np.reshape(gy, [gy.shape[0], gy.shape[1], 1])
    vec_y = np.concatenate((zero, one, gy), axis=2)

    surface_normal = np.cross(vec_x,vec_y)
    # angle_difference = np.zeros([dem.shape[0], dem.shape[1]])
    angle_difference = np.array(np.matmul(surface_normal, illumination_vec)/ (np.linalg.norm(illumination_vec, ord = 2) * np.linalg.norm(surface_normal, ord = 2, axis = 2)))
    ls = angle_difference
    return ls

def lambertian_surface2(dem, resolution, azimuth1, elevation_angle1, azimuth2, elevation_angle2):
    illumination_vec1 =  np.matrix(get_illumination_vector_real(azimuth1, elevation_angle1))
    illumination_vec2 =  np.matrix(get_illumination_vector_real(azimuth2, elevation_angle2))

    zero = np.zeros((dem.shape[0], dem.shape[1], 1))
    one = np.ones((dem.shape[0], dem.shape[1], 1))

    gx, gy = get_normal(dem, resolution)
    gx = np.pad(gx, ((0, 1), (0, 1), (0, 0)), 'constant')
    gy = np.pad(gy, ((0, 1), (0, 1), (0, 0)), 'constant')

    # gx =  cv2.Sobel(dem,-1,1,0,ksize=3)
    gx = np.reshape(gx, [gx.shape[0], gx.shape[1], 1])
    vec_x = np.concatenate((one, zero, gx), axis=2)

    # gy =  cv2.Sobel(dem,-1,0,1,ksize=3)
    # shape_gy = np.array(np.shape(gy))
    gy = np.reshape(gy, [gy.shape[0], gy.shape[1], 1])
    vec_y = np.concatenate((zero, one, gy), axis=2)

    surface_normal = np.cross(vec_x,vec_y)
    # angle_difference = np.zeros([dem.shape[0], dem.shape[1]])
    angle_difference1 = np.array(np.matmul(surface_normal, illumination_vec1)/ (np.linalg.norm(illumination_vec1, ord = 2) * np.linalg.norm(surface_normal, ord = 2, axis = 2)))
    angle_difference2 = np.array(np.matmul(surface_normal, illumination_vec2)/ (np.linalg.norm(illumination_vec2, ord = 2) * np.linalg.norm(surface_normal, ord = 2, axis = 2)))
    return angle_difference1, angle_difference2

def cone_function(radius = 112, height = 112, center_coor = (112, 112)):
    img = np.zeros((224, 224))
    circle_zero = np.zeros((224,224))
    boundary = cv2.circle(img=circle_zero, center=(center_coor[1], center_coor[0]), radius=radius, color=(1), thickness=-1)
    column_index = np.tile(range(0, 224), [224, 1])
    row_index = np.tile(np.reshape(range(0, 224), [224, 1]), [1, 224])
    img = np.where(boundary == 0, img, height * np.nan_to_num(np.sqrt(1 - (np.square(row_index - center_coor[0]) / radius**2) - (np.square(column_index - center_coor[1]) / radius**2))))
    img = np.reshape(img, (224, 224, 1))
    img = lambertian_surface(img, 1, 90, 45)
    img = np.where(img<0, 0, img)
    return img

def seeliger_surface(dem, resolution, azimuth, elevation_angle):
    illumination_vec = get_illumination_vector_real(azimuth, elevation_angle)
    camera_vec =  np.matrix([[0],
                             [0],
                             [1]])

    zero = np.zeros((dem.shape[0], dem.shape[1], 1))
    one = np.ones((dem.shape[0], dem.shape[1], 1))

    gx =  cv2.Sobel(dem,-1,1,0,ksize=3)
    gx = np.reshape(gx,[gx.shape[0], gx.shape[1], 1])
    gx = gx / resolution
    vec_x = np.concatenate((one,zero,gx), axis = 2)

    gy =  cv2.Sobel(dem,-1,0,1,ksize=3)
    shape_gy = np.array(np.shape(gy))
    gy = np.reshape(gy,[gy.shape[0], gy.shape[1], 1])
    gy = gy / resolution
    vec_y = np.concatenate((zero, one, gy), axis = 2)

    surface_normal = np.cross(vec_x,vec_y)
    incidence_angle = np.maximum(np.matmul(surface_normal,illumination_vec) / (np.linalg.norm(surface_normal) * np.linalg.norm(illumination_vec)), 0)
    incidence_angle = np.reshape(incidence_angle, [416,416])
    emission_angle = np.array(np.matmul(surface_normal,camera_vec) / (np.linalg.norm(surface_normal) * np.linalg.norm(camera_vec)))
    lommel_seeliger =  np.array(incidence_angle / (incidence_angle + emission_angle))
    return lommel_seeliger

def validation_set():
    label = []
    sample = []
    for i in range(1, 1000000):
        elevation_angle = 20 + i * 70 / 1000000

        dem = shape_generator((416, 416), 5, 5)
        image = np.stack((seeliger_surface(dem, 1, elevation_angle)) * 3, axis=-1)
        zero_image = np.zeros((416, 416, 3), dtype=np.float32)
        zero_dem = np.zeros((416, 416, 1), dtype=np.float32)
        zero_image = np.uint8((image - np.min(image)) / (np.max(image) - np.min(image)) * 255)
        zero_dem = np.expand_dims(dem, axis = 2)
        zero_dem = np.uint8((zero_dem - np.min(zero_dem)) / (np.max(zero_dem) - np.min(zero_dem)) * 255)
        cv2.imwrite("G:/cone_dataset/sample/" + str(i) + ".jpg", zero_image)
        cv2.imwrite("G:/cone_dataset/label/" + str(i) + ".jpg", zero_dem)

        with open("G:\cone_dataset/training_sample.txt", "a") as f1:
            f1.write("G:/cone_dataset/sample/" + str(i) + ".jpg" + "\n")
        with open("G:\cone_dataset/training_label.txt", "a") as f2:
            f2.write("G:/cone_dataset/label/" + str(i) + ".jpg" + "\n")
        with open("G:\cone_dataset/illumination_angle.txt", "a") as f3:
            f3.write(str(elevation_angle) + "\n")
        #sample.append(zero_image)
        #label.append(zero_dem)
        print(i)

    #sample = (sample - np.mean(sample)) / np.var(sample)
    #label = (label - np.mean(label)) / np.var(label)
    #np.save("E:/validation_sample.npy", sample)
    #np.save("E:/validation_label.npy", label)

def generate_coordinates(percentage):
    # 计算图片总像素数
    total_pixels = 50176

    # 计算要生成的坐标点数
    num_coordinates = int(total_pixels * percentage)

    # 创建网格
    grid_x, grid_y = np.meshgrid(range(224), range(224))

    # 将网格坐标转为一维数组
    coordinates = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    # 随机洗牌
    np.random.shuffle(coordinates)

    # 选择指定数量的坐标
    selected_coordinates = coordinates[:num_coordinates]

    return selected_coordinates

def training_data_loader_pretrain(batch_size):
    data_list = ["E:/alos_dem/pretrain_dataset/30m",
                "E:/alos_dem/pretrain_dataset/40m",
                "E:/alos_dem/pretrain_dataset/50m",
                "E:/alos_dem/pretrain_dataset/60m",
                "E:/alos_dem/pretrain_dataset/70m",
                "E:/alos_dem/pretrain_dataset/80m",
                "E:/alos_dem/pretrain_dataset/90m",
                "E:/alos_dem/pretrain_dataset/100m",
                "E:/alos_dem/pretrain_dataset/110m",
                "E:/alos_dem/pretrain_dataset/120m",
                "E:/alos_dem/pretrain_dataset/130m",
                "E:/alos_dem/pretrain_dataset/140m",
                "E:/alos_dem/pretrain_dataset/150m",
                "E:/alos_dem/pretrain_dataset/160m",
                "E:/alos_dem/pretrain_dataset/170m",
                "E:/alos_dem/pretrain_dataset/180m",
                "E:/alos_dem/pretrain_dataset/190m",
                "E:/alos_dem/pretrain_dataset/200m",
                "E:/alos_dem/pretrain_dataset/210m",
                "E:/alos_dem/pretrain_dataset/220m",
                "E:/alos_dem/pretrain_dataset/230m",
                "E:/alos_dem/pretrain_dataset/240m",
                "E:/alos_dem/pretrain_dataset/250m",
                "E:/alos_dem/pretrain_dataset/260m",
                "E:/alos_dem/pretrain_dataset/270m",
                "E:/alos_dem/pretrain_dataset/280m",
                "E:/alos_dem/pretrain_dataset/290m",
                "E:/alos_dem/pretrain_dataset/300m",
                ]
    dem_list = []
    for z in range(len(data_list)):
        img_list = os.listdir(data_list[z])
        dem_list.append(img_list)
    prob = [0.40, 0.25, 0.10, 0.01, 0.01, 0.01,
             0.01, 0.01, 0.01,  0.01, 0.01, 0.01,
             0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
             0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
             0.01, 0.01, 0.01, 0.01]

    while True:

        lamb_batch1 = []
        light_batch1 = []

        lamb_batch2 = []
        light_batch2 = []
        albedo_batch1 = []
        albedo_batch2 = []
        dem_batch = []


        for i in range(batch_size):
            random_path = np.random.choice(data_list, 1, False, prob)[0]
            resolution = int(random_path.split("/")[-1][:-1])
            index = int((resolution - 30) / 10)
            shuff_number = int(np.random.randint(0, len(dem_list[index]), 1))
            dem = tff.imread(os.path.join(random_path, dem_list[index][shuff_number]))
            dem = np.expand_dims(dem, axis=2)

            fake_azimuth = np.random.randint(0, 360, 2)
            elevation_angle = np.random.randint(10, 80, 2)

            sun_azimuth1 = encoding(int(fake_azimuth[0]), 256)
            sun_elevation1 = encoding(int(elevation_angle[0]), 256)
            light1 = np.concatenate((sun_azimuth1, sun_elevation1), axis = 0)
            lamb1 = lambertian_surface(dem, resolution, int(fake_azimuth[0]), int(elevation_angle[0]))

            sun_azimuth2 = encoding(fake_azimuth[1], 256)
            sun_elevation2 = encoding(elevation_angle[1], 256)
            light2 = np.concatenate((sun_azimuth2, sun_elevation2), axis = 0)
            lamb2 = lambertian_surface(dem, resolution, int(fake_azimuth[1]), int(elevation_angle[1]))

            dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem) + np.random.uniform(0.2, 0.5))
            dem = np.squeeze(dem)
            # 计算要设置为0的像素数量
            percentage = np.random.uniform(0.95, 0.999)
            coor = generate_coordinates(percentage)
            # 将选定的像素值设置为0
            dem[coor[:, 0], coor[:, 1]] = 0.

            lamb_int = np.uint8(lamb1 * 255)
            slic = cv2.ximgproc.createSuperpixelSLIC(lamb_int, region_size=70, ruler=50)
            slic.iterate(10)
            mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
            label_slic = slic.getLabels()
            num_region = np.max(label_slic) - np.min(label_slic) + 1

            albedo = np.ones((224, 224))
            for k in range(num_region):
                random_albedo = np.random.uniform(0.3, 0.9, 1)
                index = list(np.where(label_slic == k))
                lamb1[list(index[0]),list(index[1])] = lamb1[list(index[0]),list(index[1])] * random_albedo
                lamb2[list(index[0]), list(index[1])] = lamb2[list(index[0]), list(index[1])] * random_albedo
                albedo[list(index[0]),list(index[1])] = albedo[list(index[0]),list(index[1])] * random_albedo

            light_batch1.append(light1)
            lamb_batch1.append(lamb1)
            light_batch2.append(light2)
            lamb_batch2.append(lamb2)
            dem_batch.append(dem)
            albedo_batch1.append(albedo)
            albedo_batch2.append(albedo)


        lamb_batch1 = np.array(lamb_batch1, dtype=np.float32)
        lamb_batch1 = np.reshape(lamb_batch1, (batch_size, 224, 224, 1))

        light_batch1 = np.array(light_batch1, dtype=np.float32)
        light_batch1 = np.reshape(light_batch1, (batch_size, 512))

        lamb_batch2 = np.array(lamb_batch2, dtype=np.float32)
        lamb_batch2 = np.reshape(lamb_batch2, (batch_size, 224, 224, 1))

        light_batch2 = np.array(light_batch2, dtype=np.float32)
        light_batch2 = np.reshape(light_batch2, (batch_size, 512))

        dem_batch = np.array(dem_batch, dtype=np.float32)
        dem_batch = np.reshape(dem_batch, (batch_size, 224, 224, 1))

        albedo_batch1 = np.array(albedo_batch1, dtype=np.float32)
        albedo_batch1 = np.reshape(albedo_batch1, (batch_size, 224, 224, 1))

        albedo_batch2 = np.array(albedo_batch2, dtype=np.float32)
        albedo_batch2 = np.reshape(albedo_batch2, (batch_size, 224, 224, 1))

        yield [lamb_batch1, lamb_batch2, light_batch1, light_batch2, dem_batch, albedo_batch1, albedo_batch2], np.zeros(shape=(batch_size))
        #yield [dem_batch1, img_batch1, light_batch1], albedo_batch1


def validation_data_loader_pretrain(batch_size):
    data_list = ["E:/alos_dem/pretrain_dataset/30m",
                "E:/alos_dem/pretrain_dataset/40m",
                "E:/alos_dem/pretrain_dataset/50m",
                "E:/alos_dem/pretrain_dataset/60m",
                "E:/alos_dem/pretrain_dataset/70m",
                "E:/alos_dem/pretrain_dataset/80m",
                "E:/alos_dem/pretrain_dataset/90m",
                "E:/alos_dem/pretrain_dataset/100m",
                "E:/alos_dem/pretrain_dataset/110m",
                "E:/alos_dem/pretrain_dataset/120m",
                "E:/alos_dem/pretrain_dataset/130m",
                "E:/alos_dem/pretrain_dataset/140m",
                "E:/alos_dem/pretrain_dataset/150m",
                "E:/alos_dem/pretrain_dataset/160m",
                "E:/alos_dem/pretrain_dataset/170m",
                "E:/alos_dem/pretrain_dataset/180m",
                "E:/alos_dem/pretrain_dataset/190m",
                "E:/alos_dem/pretrain_dataset/200m",
                "E:/alos_dem/pretrain_dataset/210m",
                "E:/alos_dem/pretrain_dataset/220m",
                "E:/alos_dem/pretrain_dataset/230m",
                "E:/alos_dem/pretrain_dataset/240m",
                "E:/alos_dem/pretrain_dataset/250m",
                "E:/alos_dem/pretrain_dataset/260m",
                "E:/alos_dem/pretrain_dataset/270m",
                "E:/alos_dem/pretrain_dataset/280m",
                "E:/alos_dem/pretrain_dataset/290m",
                "E:/alos_dem/pretrain_dataset/300m",
                ]
    dem_list = []
    for z in range(len(data_list)):
        img_list = os.listdir(data_list[z])
        dem_list.append(img_list)
    prob = [0.40, 0.25, 0.10, 0.01, 0.01, 0.01,
             0.01, 0.01, 0.01,  0.01, 0.01, 0.01,
             0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
             0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
             0.01, 0.01, 0.01, 0.01]


    while True:

        lamb_batch1 = []
        light_batch1 = []

        lamb_batch2 = []
        light_batch2 = []
        albedo_batch1 = []
        albedo_batch2 = []
        dem_batch = []

        for i in range(batch_size):
            random_path = np.random.choice(data_list, 1, False, prob)[0]
            resolution = int(random_path.split("/")[-1][:-1])
            index = int((resolution - 30) / 10)
            shuff_number = int(np.random.randint(0, len(dem_list[index]), 1))
            dem = tff.imread(os.path.join(random_path, dem_list[index][shuff_number]))
            dem = np.expand_dims(dem, axis=2)

            fake_azimuth = np.random.randint(0, 360, 2)
            elevation_angle = np.random.randint(10, 80, 2)

            sun_azimuth1 = encoding(int(fake_azimuth[0]), 256)
            sun_elevation1 = encoding(int(elevation_angle[0]), 256)
            light1 = np.concatenate((sun_azimuth1, sun_elevation1), axis=0)
            lamb1 = lambertian_surface(dem, resolution, int(fake_azimuth[0]), int(elevation_angle[0]))

            sun_azimuth2 = encoding(fake_azimuth[1], 256)
            sun_elevation2 = encoding(elevation_angle[1], 256)
            light2 = np.concatenate((sun_azimuth2, sun_elevation2), axis=0)
            lamb2 = lambertian_surface(dem, resolution, int(fake_azimuth[1]), int(elevation_angle[1]))

            dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem) + np.random.uniform(0.2, 0.5))
            dem = np.squeeze(dem)
            # 计算要设置为0的像素数量
            percentage = np.random.uniform(0.95, 0.999)
            coor = generate_coordinates(percentage)
            # 将选定的像素值设置为0
            dem[coor[:, 0], coor[:, 1]] = 0.

            lamb_int = np.uint8(lamb1 * 255)
            slic = cv2.ximgproc.createSuperpixelSLIC(lamb_int, region_size=70, ruler=50)
            slic.iterate(10)
            mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
            label_slic = slic.getLabels()
            num_region = np.max(label_slic) - np.min(label_slic) + 1

            albedo = np.ones((224, 224))
            for k in range(num_region):
                random_albedo = np.random.uniform(0.3, 0.9, 1)
                index = list(np.where(label_slic == k))
                lamb1[list(index[0]), list(index[1])] = lamb1[list(index[0]), list(index[1])] * random_albedo
                lamb2[list(index[0]), list(index[1])] = lamb2[list(index[0]), list(index[1])] * random_albedo
                albedo[list(index[0]), list(index[1])] = albedo[list(index[0]), list(index[1])] * random_albedo

            light_batch1.append(light1)
            lamb_batch1.append(lamb1)
            light_batch2.append(light2)
            lamb_batch2.append(lamb2)
            dem_batch.append(dem)
            albedo_batch1.append(albedo)
            albedo_batch2.append(albedo)

        lamb_batch1 = np.array(lamb_batch1, dtype=np.float32)
        lamb_batch1 = np.reshape(lamb_batch1, (batch_size, 224, 224, 1))

        light_batch1 = np.array(light_batch1, dtype=np.float32)
        light_batch1 = np.reshape(light_batch1, (batch_size, 512))

        lamb_batch2 = np.array(lamb_batch2, dtype=np.float32)
        lamb_batch2 = np.reshape(lamb_batch2, (batch_size, 224, 224, 1))

        light_batch2 = np.array(light_batch2, dtype=np.float32)
        light_batch2 = np.reshape(light_batch2, (batch_size, 512))

        dem_batch = np.array(dem_batch, dtype=np.float32)
        dem_batch = np.reshape(dem_batch, (batch_size, 224, 224, 1))

        albedo_batch1 = np.array(albedo_batch1, dtype=np.float32)
        albedo_batch1 = np.reshape(albedo_batch1, (batch_size, 224, 224, 1))

        albedo_batch2 = np.array(albedo_batch2, dtype=np.float32)
        albedo_batch2 = np.reshape(albedo_batch2, (batch_size, 224, 224, 1))

        yield [lamb_batch1, lamb_batch2, light_batch1, light_batch2, dem_batch, albedo_batch1, albedo_batch2], np.zeros(
            shape=(batch_size))
        #yield [img_batch1, dem_batch1, light_batch1], albedo_batch1

def training_data_loader_real(batch_size, resolution):
    data_dict = np.load(NPY_FILE_PATH, allow_pickle=True).item()
    folder_list = list(data_dict.keys())
    while True:
        dem_batch = []
        image_batch1 = []
        light_batch1 = []
        image_batch2 = []
        light_batch2 = []
        albedo_batch1 = []
        albedo_batch2 = []

        for i in range(batch_size):
            folder_shuff = int(np.random.randint(0, len(folder_list) - 1, 1))
            data = data_dict[folder_list[folder_shuff]]
            data_shuff = list(np.random.choice(len(data), 2, replace=False))
            file_shuff = int(np.random.randint(0, int(data[0]["total_number"]), 1))

            img_path1 = data[data_shuff[0]]["image_save_path"]
            img_path2 = data[data_shuff[1]]["image_save_path"]
            dem_path = data[data_shuff[0]]["dem_save_path"]
            albedo_path1 = data[data_shuff[0]]["albedo_save_path"]
            albedo_path2 = data[data_shuff[1]]["albedo_save_path"]

            azimuth1 = float(data[data_shuff[0]]["azimuth_angle"]) + 90.
            elevation_angle1 = 90 - data[data_shuff[0]]["incidence_angle"]
            azimuth2 = float(data[data_shuff[1]]["azimuth_angle"]) + 90.
            elevation_angle2 = 90 - data[data_shuff[1]]["incidence_angle"]

            image1 = np.nan_to_num(tff.imread(os.path.join(img_path1, "{}.tif".format(file_shuff))))
            image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
            image2 = np.nan_to_num(tff.imread(os.path.join(img_path2, "{}.tif".format(file_shuff))))
            image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))
            albedo1 = np.nan_to_num(tff.imread(os.path.join(albedo_path1, "{}.tif".format(file_shuff))))
            albedo1 = (albedo1 - np.min(albedo1)) / (np.max(albedo1) - np.min(albedo1))
            albedo2 = np.nan_to_num(tff.imread(os.path.join(albedo_path2, "{}.tif".format(file_shuff))))
            albedo2 = (albedo2 - np.min(albedo2)) / (np.max(albedo2) - np.min(albedo2))
            dem = tff.imread(os.path.join(dem_path, "{}.tif".format(file_shuff)))
            dem_index = np.where(dem!=100000)
            if len(dem_index[0] != 0):
                dem_max = np.max(dem[dem_index])
                dem_min = np.min(dem[dem_index])
                dem[dem_index] = np.nan_to_num((dem[dem_index] - dem_min) / (dem_max - dem_min))
            dem = np.where(dem > 1., 1., dem)
            dem = np.where(dem < 0., 0., dem)
            dem = np.expand_dims(dem, axis=2)


            # 计算要设置为0的像素数量
            percentage = np.random.uniform(0.95, 0.999)
            coor = generate_coordinates(percentage)
            # 将选定的像素值设置为0
            dem[coor[:, 0], coor[:, 1]] = 0.

            sun_azimuth1 = encoding(azimuth1, 256)
            sun_elevation1 = encoding(elevation_angle1, 256)
            light1 = np.concatenate((sun_azimuth1, sun_elevation1), axis=0)
            sun_azimuth2 = encoding(azimuth2, 256)
            sun_elevation2 = encoding(elevation_angle2, 256)
            light2 = np.concatenate((sun_azimuth2, sun_elevation2), axis=0)

            # tff.imsave(r"D:\albedo_test\test/image1.tif", image1)
            # tff.imsave(r"D:\albedo_test\test/image2.tif", image2)
            # tff.imsave(r"D:\albedo_test\test/albedo1.tif", albedo1)
            # tff.imsave(r"D:\albedo_test\test/albedo2.tif", albedo2)
            # tff.imsave(r"D:\albedo_test\test/dem.tif", dem)

            light_batch1.append(np.nan_to_num(light1))
            image_batch1.append(np.nan_to_num(image1))
            light_batch2.append(np.nan_to_num(light2))
            image_batch2.append(np.nan_to_num(image2))
            dem_batch.append(np.nan_to_num(dem))
            albedo_batch1.append(np.nan_to_num(albedo1))
            albedo_batch2.append(np.nan_to_num(albedo2))

        dem_batch = np.array(dem_batch, dtype=np.float32)
        dem_batch = np.reshape(dem_batch, (batch_size, 224, 224, 1))

        image_batch1 = np.array(image_batch1, dtype=np.float32)
        image_batch1 = np.reshape(image_batch1, (batch_size, 224, 224, 1))
        image_batch2 = np.array(image_batch2, dtype=np.float32)
        image_batch2 = np.reshape(image_batch2, (batch_size, 224, 224, 1))

        light_batch1 = np.array(light_batch1, dtype=np.float32)
        light_batch1 = np.reshape(light_batch1, (batch_size, 512))
        light_batch2 = np.array(light_batch2, dtype=np.float32)
        light_batch2 = np.reshape(light_batch2, (batch_size, 512))

        albedo_batch1 = np.array(albedo_batch1, dtype=np.float32)
        albedo_batch1 = np.reshape(albedo_batch1, (batch_size, 224, 224, 1))
        albedo_batch2 = np.array(albedo_batch2, dtype=np.float32)
        albedo_batch2 = np.reshape(albedo_batch2, (batch_size, 224, 224, 1))

        yield [image_batch1, image_batch2, light_batch1, light_batch2, dem_batch, albedo_batch1, albedo_batch2], np.zeros(
            shape=(batch_size))
        #return [dem_batch, lamb_batch, light_batch], albedo_batch

def validation_data_loader_real(batch_size, resolution):
    data_dict = np.load(NPY_FILE_PATH, allow_pickle=True).item()
    folder_list = list(data_dict.keys())
    while True:
        dem_batch = []
        image_batch1 = []
        light_batch1 = []
        image_batch2 = []
        light_batch2 = []
        albedo_batch1 = []
        albedo_batch2 = []

        for i in range(batch_size):
            folder_shuff = int(np.random.randint(len(folder_list) - 1, len(folder_list), 1))
            data = data_dict[folder_list[folder_shuff]]
            data_shuff = list(np.random.choice(len(data), 2, replace=False))
            file_shuff = int(np.random.randint(0, int(data[0]["total_number"]), 1))

            img_path1 = data[data_shuff[0]]["image_save_path"]
            img_path2 = data[data_shuff[1]]["image_save_path"]
            dem_path = data[data_shuff[0]]["dem_save_path"]
            albedo_path1 = data[data_shuff[0]]["albedo_save_path"]
            albedo_path2 = data[data_shuff[1]]["albedo_save_path"]

            azimuth1 = float(data[data_shuff[0]]["azimuth_angle"]) + 90.
            elevation_angle1 = 90 - data[data_shuff[0]]["incidence_angle"]
            azimuth2 = float(data[data_shuff[1]]["azimuth_angle"]) + 90.
            elevation_angle2 = 90 - data[data_shuff[1]]["incidence_angle"]

            image1 = np.nan_to_num(tff.imread(os.path.join(img_path1, "{}.tif".format(file_shuff))))
            image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
            image2 = np.nan_to_num(tff.imread(os.path.join(img_path2, "{}.tif".format(file_shuff))))
            image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))
            albedo1 = np.nan_to_num(tff.imread(os.path.join(albedo_path1, "{}.tif".format(file_shuff))))
            albedo1 = (albedo1 - np.min(albedo1)) / (np.max(albedo1) - np.min(albedo1))
            albedo2 = np.nan_to_num(tff.imread(os.path.join(albedo_path2, "{}.tif".format(file_shuff))))
            albedo2 = (albedo2 - np.min(albedo2)) / (np.max(albedo2) - np.min(albedo2))
            dem = tff.imread(os.path.join(dem_path, "{}.tif".format(file_shuff)))
            dem_index = np.where(dem!=100000)
            if len(dem_index[0] != 0):
                dem_max = np.max(dem[dem_index])
                dem_min = np.min(dem[dem_index])
                dem[dem_index] = np.nan_to_num((dem[dem_index] - dem_min) / (dem_max - dem_min))
            dem = np.where(dem > 1., 1., dem)
            dem = np.where(dem < 0., 0., dem)
            dem = np.expand_dims(dem, axis=2)

            # 计算要设置为0的像素数量
            percentage = np.random.uniform(0.95, 0.999)
            coor = generate_coordinates(percentage)
            # 将选定的像素值设置为0
            dem[coor[:, 0], coor[:, 1]] = 0.

            sun_azimuth1 = encoding(azimuth1, 256)
            sun_elevation1 = encoding(elevation_angle1, 256)
            light1 = np.concatenate((sun_azimuth1, sun_elevation1), axis=0)
            sun_azimuth2 = encoding(azimuth2, 256)
            sun_elevation2 = encoding(elevation_angle2, 256)
            light2 = np.concatenate((sun_azimuth2, sun_elevation2), axis=0)

            # tff.imsave(r"D:\albedo_test\test/image1.tif", image1)
            # tff.imsave(r"D:\albedo_test\test/image2.tif", image2)
            # tff.imsave(r"D:\albedo_test\test/albedo1.tif", albedo1)
            # tff.imsave(r"D:\albedo_test\test/albedo2.tif", albedo2)
            # tff.imsave(r"D:\albedo_test\test/dem.tif", dem)

            light_batch1.append(np.nan_to_num(light1))
            image_batch1.append(np.nan_to_num(image1))
            light_batch2.append(np.nan_to_num(light2))
            image_batch2.append(np.nan_to_num(image2))
            dem_batch.append(np.nan_to_num(dem))
            albedo_batch1.append(np.nan_to_num(albedo1))
            albedo_batch2.append(np.nan_to_num(albedo2))

        dem_batch = np.array(dem_batch, dtype=np.float32)
        dem_batch = np.reshape(dem_batch, (batch_size, 224, 224, 1))

        image_batch1 = np.array(image_batch1, dtype=np.float32)
        image_batch1 = np.reshape(image_batch1, (batch_size, 224, 224, 1))
        image_batch2 = np.array(image_batch2, dtype=np.float32)
        image_batch2 = np.reshape(image_batch2, (batch_size, 224, 224, 1))

        light_batch1 = np.array(light_batch1, dtype=np.float32)
        light_batch1 = np.reshape(light_batch1, (batch_size, 512))
        light_batch2 = np.array(light_batch2, dtype=np.float32)
        light_batch2 = np.reshape(light_batch2, (batch_size, 512))

        albedo_batch1 = np.array(albedo_batch1, dtype=np.float32)
        albedo_batch1 = np.reshape(albedo_batch1, (batch_size, 224, 224, 1))
        albedo_batch2 = np.array(albedo_batch2, dtype=np.float32)
        albedo_batch2 = np.reshape(albedo_batch2, (batch_size, 224, 224, 1))

        yield [image_batch1, image_batch2, light_batch1, light_batch2, dem_batch, albedo_batch1, albedo_batch2], np.zeros(
            shape=(batch_size))
        #return [dem_batch, lamb_batch, light_batch], albedo_batch

# class PerformanceCallback(tf.keras.callbacks.Callback):
#     def __init__(self, albedo_model):
#         super().__init__()
#         self.albedo_model = albedo_model
#
#
#     def on_epoch_end(self, epoch, logs=None):
#         dem_list = os.listdir(r"G:\Albedo\Lunar_albedo\NAC_Images\NAC_albedo_dataset_img\dem_clip\regionA")
#
#         img_list = os.listdir(r"G:\Albedo\Lunar_albedo\NAC_Images\NAC_albedo_dataset_img\img_clip\regionA\M1142603254LE")
#
#         for i in range(len(dem_list)):
#             img = tff.imread(os.path.join(r"G:\Albedo\Lunar_albedo\NAC_Images\NAC_albedo_dataset_img\img_clip\regionA\M1142603254LE", img_list[i]))
#             dem = tff.imread(os.path.join(r"G:\Albedo\Lunar_albedo\NAC_Images\NAC_albedo_dataset_img\dem_clip\regionA", dem_list[i]))
#
#             index = np.where(dem == 100000)
#             dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
#
#             # 计算要设置为0的像素数量
#             percentage = np.random.uniform(0.95, 0.999)
#             coor = generate_coordinates(percentage)
#             # 将选定的像素值设置为0
#             dem[coor[:, 0], coor[:, 1]] = 0.
#
#             dem = np.expand_dims(dem, axis=2)
#             img = ((img - np.min(img)) / (np.max(img) - np.min(img)))
#             img = np.expand_dims(img, axis=2)
#             #img = np.concatenate((img, img, img), axis=2)
#
#             fake_azimuth = 222
#             elevation_angle = 16
#             sun_azimuth1 = encoding(fake_azimuth, 256)
#             sun_elevation1 = encoding(elevation_angle, 256)
#             light1 = np.concatenate((sun_azimuth1, sun_elevation1), axis=0)
#             light1 = np.reshape(light1, (1, 512))
#             dem = np.float32(np.reshape(dem, (1, 224, 224, 1)))
#             img = np.float32(np.reshape(img, (1, 224, 224, 1)))
#             predict = self.albedo_model.predict([dem, img, light1])
#             predict = np.reshape(predict, (224, 224))
#             #predict = np.uint8(255 * ((predict - np.min(predict)) / (np.max(predict) - np.min(predict))))
#             tff.imwrite(r"G:\Albedo\Lunar_albedo\NAC_Images\DL_results\results_ori_net\test/" + img_list[i], predict)
#         self.albedo_model.save(r"G:\Albedo\Lunar_albedo\NAC_Images\DL_results\results_ori_net\test/" + str(epoch + 1) + ".h5")



