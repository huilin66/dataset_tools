import numpy as np
from matplotlib import pyplot as plt


def apply_masks_to_image(image, masks, colors, alpha):
    """
    将带有颜色的mask叠加到图像上

    参数:
        image: 原始图像 (H, W, 3)
        masks: 多个mask (H, W, n)
        colors: 颜色数组 (n, 3)
        alpha: 透明度 (标量或n,)

    返回:
        叠加后的图像 (H, W, 3)
    """
    # 确保alpha是数组形式
    alpha = np.asarray(alpha)
    if alpha.ndim == 0:
        alpha = np.full(masks.shape[2], alpha)

    # 将masks从(H,W,n)转为(H,W,n,1)以便广播
    masks_exp = masks[..., np.newaxis]

    # 将colors从(n,3)转为(1,1,n,3)以便广播
    colors_exp = colors.reshape(1, 1, *colors.shape)

    # 调整alpha的形状为(1,1,n,1)以便广播
    alpha_exp = alpha.reshape(1, 1, -1, 1)

    # 计算每个mask的彩色贡献 (H,W,n,3)
    colored_masks = masks_exp * colors_exp

    # 加权合并所有mask (H,W,3)
    combined_masks = np.sum(colored_masks * alpha_exp, axis=2)

    # 计算所有mask的总权重 (H,W)
    total_weight = np.sum(masks * alpha, axis=2)

    # 归一化权重 (避免除以0)
    total_weight = np.clip(total_weight, 0, 1)
    total_weight_exp = np.expand_dims(total_weight, axis=-1)

    # 计算最终的mask效果 (H,W,3)
    final_mask_effect = np.where(total_weight_exp > 0,
                                 combined_masks / total_weight_exp,
                                 0)

    # 计算原始图像的保留权重 (H,W,1)
    original_weight = 1 - total_weight_exp

    # 叠加效果
    result = image * original_weight + final_mask_effect

    # 确保结果在合理范围内
    return np.clip(result, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    # 假设输入数据
    H, W = 256, 256
    n = 5
    image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)  # 随机图像
    masks = np.random.rand(H, W, n)  # 随机mask
    colors = np.random.randint(0, 256, (n, 3))  # 随机颜色
    alpha = 0.7  # 统一透明度

    print(image.shape, masks.shape, colors, alpha)
    # 应用mask
    result_image = apply_masks_to_image(image, masks, colors, alpha)
    plt.imshow(result_image)
    plt.show()