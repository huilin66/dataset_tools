# utils/geo_utils.py
import math

def calculate_floor(relative_altitude, floor_config):
    """
    根据相对高度和楼层配置计算楼层数。
    
    Args:
        relative_altitude (float): 无人机的相对高度（相对于起飞点）
        floor_config (list): [地面高度, 裙楼1高度, 裙楼2高度..., 标准层高, 顶层高度]
                             第1个和最后2个数字是固定的逻辑，中间是裙楼高度列表
    Returns:
        str: 楼层字符串 (e.g., "5F", "Podium 1", "Roof")
    """
    if relative_altitude is None:
        return "N/A"
    
    try:
        h = float(relative_altitude)
    except (ValueError, TypeError):
        return "N/A"

    # 解析配置
    base_alt = floor_config[0]
    normal_h = floor_config[-2]
    top_h = floor_config[-1]
    
    # 获取中间的裙楼高度列表
    podium_heights = floor_config[1:-2] 

    # 1. 减去基础地面高度
    current_h = h - base_alt
    
    if current_h < 0:
        return "G/F" # 或者 "Basement"

    # 2. 遍历裙楼
    floor_count = 0
    for i, p_h in enumerate(podium_heights):
        floor_count += 1
        current_h -= p_h
        if current_h < 0:
            return f"{floor_count}F (Podium)"

    # 3. 计算标准层
    # 此时 current_h 是扣除裙楼后的剩余高度
    # 计算剩余高度包含了多少个标准层
    normal_floors = int(current_h / normal_h)
    floor_count += normal_floors
    
    # 剩余高度减去这些标准层
    current_h -= (normal_floors * normal_h)
    
    # 逻辑修正：如果刚好除尽，通常算作下一层的地板，但这里我们假设无人机在楼层中间
    # floor_count 已经是当前层数（基于裙楼累加） + 标准层数量
    # 比如裙楼1层，剩余高度4米，层高3米。 4/3 = 1. floor_count = 1+1 = 2F.
    # 实际上无人机在 2F 的位置 (裙楼顶+1米)。
    
    # 4. 简单的楼层判定
    # +1 是因为 floor_count 初始为裙楼数，normal_floors 是增量，
    # 这里的 floor_count 代表的是“已经完整经过的楼层数”，当前所在的应该是 +1
    final_floor = floor_count + 1

    return f"{final_floor}F"

def calculate_orientation(gimbal_yaw):
    """
    根据云台 Yaw 角计算拍摄的立面朝向。
    逻辑：无人机朝向 North (0°)，拍摄的是建筑的 South Elevation。
    
    Args:
        gimbal_yaw (float): 云台偏航角 (-180 ~ 180)
    Returns:
        str: 朝向 (e.g., "North", "South-East")
    """
    if gimbal_yaw is None:
        return "N/A"
    
    try:
        yaw = float(gimbal_yaw)
    except (ValueError, TypeError):
        return "N/A"

    # 1. 将无人机朝向转换为 0-360 度 (0 is North, 90 is East)
    # DJI Yaw: 0(N), 90(E), 180(S), -90(W)
    if yaw < 0:
        drone_heading = 360 + yaw
    else:
        drone_heading = yaw
        
    # 2. 计算“视图”方向（Building Elevation）
    # 视图方向与无人机朝向相反 (+180度)
    view_heading = (drone_heading + 180) % 360
    
    # 3. 映射到 8 个方向
    # 每个方向占 45 度，中心点为 0, 45, 90...
    # North: 337.5 - 22.5
    # NE: 22.5 - 67.5
    directions = [
        "North", "North-East", "East", "South-East", 
        "South", "South-West", "West", "North-West", "North"
    ]
    
    # 加上 22.5 度偏移方便计算索引
    index = int((view_heading + 22.5) // 45)
    return directions[index]


def calculate_gsd(distance_mm, focal_length_mm, sensor_width_mm, image_width_pix):
    """
    计算地面采样距离 (GSD): 每个像素代表的物理世界毫米数
    Formula: GSD = (Distance * SensorWidth) / (FocalLength * ImageWidth)
    """
    if any(v is None or v == 0 for v in [distance_mm, focal_length_mm, sensor_width_mm, image_width_pix]):
        return None
        
    try:
        gsd = (float(distance_mm) * float(sensor_width_mm)) / (float(focal_length_mm) * float(image_width_pix))
        return gsd # unit: mm/pixel
    except ZeroDivisionError:
        return None

def pixel_to_physical(pix_value, gsd):
    """
    将像素值转换为厘米 (cm)
    """
    if pix_value is None or gsd is None:
        return None
    
    mm_value = pix_value * gsd
    cm_value = mm_value / 10.0
    return cm_value