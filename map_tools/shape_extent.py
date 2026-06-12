# # 1. 加载数据
# gdf_a = gpd.read_file(r"E:\data\202603_MTR_UAV\route_kml.shp")
# gdf_b = gpd.read_file(r"E:\data\202603_MTR_UAV\test.shp")
# # 2. 核心步骤：给 B 定义它原本的“家”（坐标系）
# # 如果是香港图纸，通常是 2326；如果是内地，可能是 4547 等。
# # 这里我们假设它是 EPSG:2326
# gdf_b.crs = "EPSG:2326"
# # 3. 让 Python 自动计算：把 B 从“米”转换成“经纬度”
# # 这一步会自动把那些巨大的数值（几万）换算成 114, 22 这种小数
# gdf_b_transformed = gdf_b.to_crs(gdf_a.crs)
# # 4. 检查转换后的范围
# print("转换后 B 的范围:", gdf_b_transformed.total_bounds)
# # 5. 保存
# gdf_b_transformed.to_file("route_dwg1.shp")
# # 1. 加载
# gdf_a = gpd.read_file(r"E:\data\202603_MTR_UAV\route_kml.shp")
# gdf_b = gpd.read_file(r"E:\data\202603_MTR_UAV\test.shp")
# # 2. 获取 A 的边界范围 [minx, miny, maxx, maxy]
# minxa, minya, maxxa, maxya = gdf_a.total_bounds
# # 获取 B 的边界范围
# minxb, minyb, maxxb, maxyb = gdf_b.total_bounds
# # 3. 计算缩放比例 (让 B 的大小变得和 A 一样)
# width_a = maxxa - minxa
# height_a = maxya - minya
# width_b = maxxb - minxb
# height_b = maxyb - minyb
# scale_x = width_a / width_b
# scale_y = height_a / height_b
# # 4. 执行缩放
# gdf_b.geometry = scale(
#     gdf_b.geometry, xfact=scale_x, yfact=scale_y, origin=(minxb, minyb)
# )
# # 5. 执行平移 (把缩放后的 B 搬到 A 的起点)
# dx = minxa - minxb
# dy = minya - minyb
# gdf_b.geometry = translate(gdf_b.geometry, xoff=dx, yoff=dy)
# # 6. 赋予坐标系并保存
# gdf_b.crs = gdf_a.crs
# gdf_b.to_file("route_dwg2.shp")
# import geopandas as gpd
# import pandas as pd
# from shapely.affinity import scale, translate


# def fix_cad_to_map(path_a, path_b, output_path):
#     # 1. 加载数据
#     gdf_a = gpd.read_file(path_a)  # 正确的 KML/SHP
#     gdf_b = gpd.read_file(path_b)  # CAD 转出的破碎 SHP

#     # --- 关键清洗步骤：剔除无效几何体 ---
#     # 移除空值、无效值以及空几何对象
#     gdf_b = gdf_b[gdf_b.geometry.notnull()]
#     gdf_b = gdf_b[~gdf_b.geometry.is_empty]
#     # 只保留有限坐标的数据
#     gdf_b = gdf_b[gdf_b.geometry.is_valid]

#     # 2. 获取范围
#     minxa, minya, maxxa, maxya = gdf_a.total_bounds
#     # 使用 total_bounds 避开单个 Series 判断
#     bounds_b = gdf_b.total_bounds
#     minxb, minyb, maxxb, maxyb = bounds_b

#     # 3. 计算缩放和位移
#     width_a = maxxa - minxa
#     height_a = maxya - minya
#     width_b = maxxb - minxb
#     height_b = maxyb - minyb

#     # 防止除以 0
#     if width_b == 0 or height_b == 0:
#         print("错误：源文件 B 的宽度或高度为 0，请检查数据。")
#         return

#     scale_x = width_a / width_b
#     scale_y = height_a / height_b

#     # 4. 执行缩放 (使用 lambda 遍历处理每一行，解决 ValueError)
#     # origin 使用 B 的最小点作为缩放基准
#     gdf_b.geometry = gdf_b.geometry.apply(
#         lambda g: scale(g, xfact=scale_x, yfact=scale_y, origin=(minxb, minyb))
#     )

#     # 5. 执行平移
#     dx = minxa - minxb
#     dy = minya - minyb
#     gdf_b.geometry = gdf_b.geometry.translate(xoff=dx, yoff=dy)

#     # 6. 设置坐标系并导出
#     gdf_b.crs = gdf_a.crs

#     # 导出时强制使用经纬度，并跳过无法写入的坏数据
#     try:
#         gdf_b.to_file(output_path)
#         print(f"成功！文件已保存至: {output_path}")
#         print(f"最终范围: {gdf_b.total_bounds}")
#     except Exception as e:
#         print(f"导出失败，尝试保存为 GeoJSON 格式以提高兼容性...")
#         gdf_b.to_file(output_path.replace(".shp", ".json"), driver="GeoJSON")


# # 运行
# fix_cad_to_map(
#     r"E:\data\202603_MTR_UAV\route_kml.shp",
#     r"E:\data\202603_MTR_UAV\test.shp",
#     "route_dwg3.shp",
# )


# import math

# import geopandas as gpd
# from shapely.affinity import scale, translate
# from shapely.geometry import Point


# def smart_align_cad_to_map(path_a, path_b, output_path):
#     print("开始处理...")
#     # 1. 读取数据
#     gdf_a = gpd.read_file(path_a)
#     gdf_b = gpd.read_file(path_b)

#     # 2. 基础清洗 (剔除空对象和损坏几何体)
#     gdf_b = gdf_b[
#         gdf_b.geometry.notnull() & ~gdf_b.geometry.is_empty & gdf_b.geometry.is_valid
#     ]

#     # ---------------------------------------------------------
#     # 3. 核心突破：利用空间统计自动剔除“飞点”(Outlier Removal)
#     # ---------------------------------------------------------
#     centroids = gdf_b.geometry.centroid

#     # 寻找几何数据的“中位数中心” (比平均值更抗干扰，飞点不会影响中位数)
#     median_x = centroids.x.median()
#     median_y = centroids.y.median()
#     median_center = Point(median_x, median_y)

#     # 计算每条线段离“中位数中心”的距离
#     distances = centroids.distance(median_center)

#     # 设定动态阈值：超过中位数距离 10倍 的线段，一律视为 CAD 飞点并删掉
#     # (如果是正常路线，距离通常很近；飞点的距离往往是几万倍)
#     dist_threshold = distances.median() * 10

#     # 兜底：给一个最小容差，防止路线本身太小被误删
#     dist_threshold = max(dist_threshold, 50)

#     # 提取干净的路线
#     gdf_b_clean = gdf_b[distances <= dist_threshold].copy()
#     print(
#         f"--> 原要素: {len(gdf_b)} 个 | 自动清理飞点后剩余有效路线: {len(gdf_b_clean)} 个"
#     )

#     if gdf_b_clean.empty:
#         print("错误：清理后没有数据了，请检查 B 文件本身是否只包含飞点。")
#         return

#     # ---------------------------------------------------------
#     # 4. 计算等比例缩放因子 (防止路线被强行拉伸变形)
#     # ---------------------------------------------------------
#     minxa, minya, maxxa, maxya = gdf_a.total_bounds
#     minxb, minyb, maxxb, maxyb = gdf_b_clean.total_bounds

#     # 使用对角线长度计算真实比例差，确保 X 和 Y 缩放倍数一致
#     diag_a = math.hypot(maxxa - minxa, maxya - minya)
#     diag_b = math.hypot(maxxb - minxb, maxyb - minyb)

#     if diag_b == 0:
#         print("错误：清洗后的 B 面积为0，无法缩放。")
#         return

#     scale_factor = diag_a / diag_b
#     print(f"--> 计算得到的等比例放大倍数: {scale_factor:.2f} 倍")

#     # ---------------------------------------------------------
#     # 5. 执行等比例缩放与平移对齐
#     # ---------------------------------------------------------
#     # 找准 B 的几何中心
#     center_b_x = (maxxb + minxb) / 2
#     center_b_y = (maxyb + minyb) / 2

#     # 以 B 的中心点为原点，将其等比例放大
#     gdf_b_clean.geometry = gdf_b_clean.geometry.apply(
#         lambda g: scale(
#             g, xfact=scale_factor, yfact=scale_factor, origin=(center_b_x, center_b_y)
#         )
#     )

#     # 找准 A 的几何中心
#     center_a_x = (maxxa + minxa) / 2
#     center_a_y = (maxya + minya) / 2

#     # 计算 B 放大后，搬运到 A 的中心需要走多远
#     dx = center_a_x - center_b_x
#     dy = center_a_y - center_b_y

#     # 执行平移搬运
#     gdf_b_clean.geometry = gdf_b_clean.geometry.translate(xoff=dx, yoff=dy)

#     # 6. 赋予正确的经纬度坐标系并保存
#     gdf_b_clean.crs = gdf_a.crs

#     try:
#         gdf_b_clean.to_file(output_path)
#         print(f"--> 成功！最终完美对齐的文件已保存至: {output_path}")
#     except Exception as e:
#         print(f"保存失败: {e}")


# # ==================== 运行区 ====================
# path_A = r"E:\data\202603_MTR_UAV\route_kml.shp"
# path_B = r"E:\data\202603_MTR_UAV\test.shp"
# path_Output = r"E:\data\202603_MTR_UAV\route_dwg4.shp"

# smart_align_cad_to_map(path_A, path_B, path_Output)


import geopandas as gpd
from shapely.geometry import Point


def extract_clean_layer(input_shp, output_shp):
    print("正在读取数据...")
    # 1. 加载原始 B 图层
    gdf = gpd.read_file(input_shp)
    initial_count = len(gdf)
    print(f"--> 原始数据总量: {initial_count} 个要素")

    # 2. 基础清理：剔除空几何和无效几何
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]

    # 3. 计算所有要素的中心点
    centroids = gdf.geometry.centroid

    # 4. 计算“空间中位数”作为真正的集群核心
    # 使用中位数而不是平均数，因为平均数会被极远的飞点严重拉偏
    median_x = centroids.x.median()
    median_y = centroids.y.median()
    median_center = Point(median_x, median_y)

    # 5. 计算每一个要素到集群核心的距离
    distances = centroids.distance(median_center)

    # 6. 使用 IQR (四分位距) 算法科学界定什么是“异常飞点”
    Q1 = distances.quantile(0.25)  # 前 25% 要素的距离
    Q3 = distances.quantile(0.75)  # 前 75% 要素的距离
    IQR = Q3 - Q1

    # 设定异常值阈值：
    # 对于 CAD 数据，飞点通常极其遥远。设定为 Q3 + 10 倍的 IQR 是一个极度安全的界限。
    # 它能保证 100% 留住您那团 1 万多个碎线的主干，同时毫不留情地杀掉几公里外的游离点。
    threshold = Q3 + 10 * IQR

    # 7. 执行过滤：只保留距离在阈值以内的要素
    clean_mask = distances <= threshold
    gdf_clean = gdf[clean_mask].copy()

    # 8. 统计战果
    removed_count = initial_count - len(gdf_clean)
    print(f"--> 成功侦测并剔除了 {removed_count} 个异常飞点！")
    print(f"--> 剩余抱团有效要素: {len(gdf_clean)} 个")

    # 9. 导出纯净版图层
    if not gdf_clean.empty:
        # 继承原有的坐标系元数据
        gdf_clean.crs = gdf.crs
        gdf_clean.to_file(output_shp)
        print(f"✅ 提取完成！干净的图层已保存为: {output_shp}")
    else:
        print("❌ 错误：所有数据都被判定为异常了，请检查源数据。")


# ==================== 运行区 ====================
# 输入您那个充满飞点和 1 万条碎线的原始 B 文件
input_file = r"E:\data\202603_MTR_UAV\route_dwg_clean.shp"
output_file = r"E:\data\202603_MTR_UAV\route_dwg_clean_only.shp"

extract_clean_layer(input_file, output_file)
