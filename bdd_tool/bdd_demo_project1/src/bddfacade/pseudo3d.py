from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json
import math
import numpy as np

from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
from pygltflib import Asset, Material, PbrMetallicRoughness, Texture, Image, Sampler
from pygltflib.utils import ImageFormat  # type: ignore


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    # 简易 ENU（足够伪3D外壳使用），单位：米
    # 近似：1 deg lat ~ 111320m, 1 deg lon ~ 111320*cos(lat)
    k_lat = 111320.0
    k_lon = 111320.0 * math.cos(_deg2rad(lat0))
    east = (lon - lon0) * k_lon
    north = (lat - lat0) * k_lat
    up = (alt - alt0)
    return np.array([east, north, up], dtype=np.float64)


def _wrap_angle_deg(a: float) -> float:
    # wrap to [-180, 180)
    x = (a + 180.0) % 360.0 - 180.0
    return x


def cluster_by_yaw(poses: List[Dict[str, Any]], yaw_bin_deg: float = 20.0, min_count: int = 10):
    """
    按 gimbal_yaw 分桶聚类（MVP）。
    yaw_bin_deg: 20度一个簇
    """
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for p in poses:
        y = p.get("gimbal_yaw")
        if y is None:
            continue
        y = float(y)
        y = _wrap_angle_deg(y)
        b = int(math.floor((y + 180.0) / yaw_bin_deg))
        buckets.setdefault(b, []).append(p)

    # 过滤太小的簇
    clusters = [v for v in buckets.values() if len(v) >= min_count]
    # 按数量降序
    clusters.sort(key=len, reverse=True)
    return clusters


def pick_representative_rgb(cluster: List[Dict[str, Any]]) -> str:
    # 取中间高度的那张，通常比较“正”
    cs = sorted(cluster, key=lambda x: (x.get("rel_alt") is None, x.get("rel_alt", 0.0)))
    return cs[len(cs)//2]["rgb_path"]


def build_planes_from_clusters(clusters: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    输出每个 plane 的：center, normal, width, height, texture_path
    """
    # 参考原点（第一张有 RTK 的）
    good = [p for c in clusters for p in c if p.get("lat") is not None and p.get("lon") is not None and p.get("abs_alt") is not None]
    if not good:
        raise RuntimeError("No valid lat/lon/alt found in poses.")
    lat0, lon0, alt0 = float(good[0]["lat"]), float(good[0]["lon"]), float(good[0]["abs_alt"])

    planes: List[Dict[str, Any]] = []
    for i, c in enumerate(clusters):
        pts = []
        yaws = []
        pitches = []
        for p in c:
            if p.get("lat") is None or p.get("lon") is None or p.get("abs_alt") is None:
                continue
            pts.append(_latlon_to_enu(float(p["lat"]), float(p["lon"]), float(p["abs_alt"]), lat0, lon0, alt0))
            if p.get("gimbal_yaw") is not None:
                yaws.append(float(p["gimbal_yaw"]))
            if p.get("gimbal_pitch") is not None:
                pitches.append(float(p["gimbal_pitch"]))

        if len(pts) < 3:
            continue

        P = np.stack(pts, axis=0)
        center = P.mean(axis=0)

        # 用 yaw 估计法向（相机看向≈-法向，这里只要一致即可）
        yaw = float(np.median(np.array([_wrap_angle_deg(x) for x in yaws]))) if yaws else 0.0
        # 约定：yaw=0 指向北；转 ENU: x=east, y=north
        forward = np.array([math.sin(_deg2rad(yaw)), math.cos(_deg2rad(yaw)), 0.0], dtype=np.float64)
        normal = -forward  # plane normal
        normal = normal / (np.linalg.norm(normal) + 1e-9)

        # plane 的 width/height：用相机点云在平面切向方向投影范围估计（MVP）
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tangent_u = np.cross(up, normal)
        if np.linalg.norm(tangent_u) < 1e-6:
            tangent_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_u = tangent_u / (np.linalg.norm(tangent_u) + 1e-9)
        tangent_v = np.cross(normal, tangent_u)
        tangent_v = tangent_v / (np.linalg.norm(tangent_v) + 1e-9)

        rel = P - center[None, :]
        u = rel @ tangent_u
        v = rel @ tangent_v

        # 给一点余量（因为相机点云不等于墙面尺寸）
        width = float((u.max() - u.min()) * 2.5)
        height = float((v.max() - v.min()) * 2.5)

        width = max(width, 10.0)
        height = max(height, 10.0)

        tex = pick_representative_rgb(c)

        planes.append({
            "id": f"facade_{i}",
            "center": center.tolist(),
            "normal": normal.tolist(),
            "tangent_u": tangent_u.tolist(),
            "tangent_v": tangent_v.tolist(),
            "width": width,
            "height": height,
            "texture_rgb": tex,
        })

    return planes

def _pack_f32(a: np.ndarray) -> bytes:
    return a.astype(np.float32).tobytes(order="C")

def _pack_u16(a: np.ndarray) -> bytes:
    return a.astype(np.uint16).tobytes(order="C")

def write_glb_with_textures(planes: List[Dict[str, Any]], out_glb: Path, out_assets_dir: Path) -> None:
    """
    生成一个 glb + 旁路纹理文件（png/jpg 直接引用文件），便于 Web 加载。
    简化：glb 只存几何与材质引用，纹理放在 outputs/textures/ 下。
    """
    out_assets_dir.mkdir(parents=True, exist_ok=True)

    # 统一拷贝纹理（先用原始 RGB jpg；后续可换成拼接纹理）
    textures = []
    for p in planes:
        src = Path(p["texture_rgb"])
        dst = out_assets_dir / f"{p['id']}_rgb{src.suffix.lower()}"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        p["texture_rgb_rel"] = str(dst.relative_to(out_glb.parent).as_posix())
        textures.append(p["texture_rgb_rel"])

    # 构建 glTF（单 buffer）
    positions = []
    uvs = []
    indices = []
    nodes = []
    meshes = []
    materials = []
    images = []
    textures_gltf = []
    samplers = [Sampler()]  # default

    # 每个 plane 一个 mesh/material/texture/image
    vertex_offset = 0
    for i, p in enumerate(planes):
        w = p["width"]
        h = p["height"]
        c = np.array(p["center"], dtype=np.float64)
        tu = np.array(p["tangent_u"], dtype=np.float64)
        tv = np.array(p["tangent_v"], dtype=np.float64)

        # 4 corners
        v0 = c - tu*(w/2) - tv*(h/2)
        v1 = c + tu*(w/2) - tv*(h/2)
        v2 = c + tu*(w/2) + tv*(h/2)
        v3 = c - tu*(w/2) + tv*(h/2)
        positions.extend([v0, v1, v2, v3])

        # UVs
        uvs.extend([[0,0],[1,0],[1,1],[0,1]])

        # 2 triangles
        indices.extend([
            vertex_offset+0, vertex_offset+1, vertex_offset+2,
            vertex_offset+0, vertex_offset+2, vertex_offset+3
        ])
        vertex_offset += 4

        # image/texture/material
        images.append(Image(uri=p["texture_rgb_rel"]))
        textures_gltf.append(Texture(source=len(images)-1, sampler=0))
        materials.append(Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture={"index": len(textures_gltf)-1},
                metallicFactor=0.0,
                roughnessFactor=1.0
            ),
            doubleSided=True
        ))

    pos_arr = np.array(positions, dtype=np.float32)
    uv_arr = np.array(uvs, dtype=np.float32)
    idx_arr = np.array(indices, dtype=np.uint16)

    # buffer layout: positions | uvs | indices
    pos_bytes = _pack_f32(pos_arr)
    uv_bytes = _pack_f32(uv_arr)
    idx_bytes = _pack_u16(idx_arr)

    # 4-byte align
    def pad4(b: bytes) -> bytes:
        return b + b"\x00" * ((4 - (len(b) % 4)) % 4)

    pos_bytes = pad4(pos_bytes)
    uv_bytes = pad4(uv_bytes)
    idx_bytes = pad4(idx_bytes)

    blob = pos_bytes + uv_bytes + idx_bytes

    # BufferView offsets
    pos_off = 0
    uv_off = len(pos_bytes)
    idx_off = len(pos_bytes) + len(uv_bytes)

    bv_pos = BufferView(buffer=0, byteOffset=pos_off, byteLength=len(pos_bytes), target=34962)
    bv_uv  = BufferView(buffer=0, byteOffset=uv_off,  byteLength=len(uv_bytes),  target=34962)
    bv_idx = BufferView(buffer=0, byteOffset=idx_off, byteLength=len(idx_bytes), target=34963)

    # Accessors
    acc_pos = Accessor(
        bufferView=0, byteOffset=0, componentType=5126, count=len(pos_arr),
        type="VEC3",
        min=pos_arr.min(axis=0).tolist(),
        max=pos_arr.max(axis=0).tolist()
    )
    acc_uv = Accessor(
        bufferView=1, byteOffset=0, componentType=5126, count=len(uv_arr),
        type="VEC2"
    )
    acc_idx = Accessor(
        bufferView=2, byteOffset=0, componentType=5123, count=len(idx_arr),
        type="SCALAR"
    )

    # Mesh + Nodes
    primitives = []
    # 每个 plane 6 indices、4 vertices
    for i in range(len(planes)):
        prim = Primitive(
            attributes={"POSITION": 0, "TEXCOORD_0": 1},
            indices=2,
            material=i
        )
        meshes.append(Mesh(primitives=[prim]))
        nodes.append(Node(mesh=i, name=planes[i]["id"]))

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        scenes=[Scene(nodes=list(range(len(nodes))))],
        scene=0,
        nodes=nodes,
        meshes=meshes,
        materials=materials,
        textures=textures_gltf,
        images=images,
        samplers=samplers,
        buffers=[Buffer(byteLength=len(blob))],
        bufferViews=[bv_pos, bv_uv, bv_idx],
        accessors=[acc_pos, acc_uv, acc_idx],
    )

    # 写 GLB（buffer 内嵌）
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    gltf.set_binary_blob(blob)
    gltf.save_binary(str(out_glb))
