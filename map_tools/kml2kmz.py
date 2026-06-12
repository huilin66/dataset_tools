import os
import zipfile

import fiona
import geopandas as gpd

# 1. 开启 fiona 对 KML 的读写支持
fiona.drvsupport.supported_drivers["KML"] = "rw"

# 2. 读取你的空间数据 (比如 Shapefile 或 GeoJSON)
gdf = gpd.read_file("your_data.shp")

# 3. 核心步骤：强制转换为 WGS84 坐标系 (EPSG:4326)，这是 KML 的硬性要求
if gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# 4. 导出为 KML
kml_path = "output.kml"
gdf.to_file(kml_path, driver="KML")
print("KML 导出成功！")

# 5. 将 KML 打包为 KMZ
kmz_path = "output.kmz"
with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as kmz:
    # 在压缩包中，kml 文件通常命名为 doc.kml
    kmz.write(kml_path, arcname="doc.kml")

# 清理临时的 KML 文件
os.remove(kml_path)
print(f"已成功生成 KMZ 文件：{kmz_path}")
