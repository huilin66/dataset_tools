from __future__ import annotations
import argparse
from pathlib import Path
import json

from .pseudo3d import cluster_by_yaw, build_planes_from_clusters, write_glb_with_textures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=str, default="outputs/poses_rgb.json")
    ap.add_argument("--out-glb", type=str, default="outputs/building.glb")
    ap.add_argument("--textures-dir", type=str, default="outputs/textures")
    ap.add_argument("--yaw-bin", type=float, default=20.0)
    ap.add_argument("--min-count", type=int, default=15)
    args = ap.parse_args()

    poses = json.loads(Path(args.poses).read_text(encoding="utf-8"))
    clusters = cluster_by_yaw(poses, yaw_bin_deg=args.yaw_bin, min_count=args.min_count)
    planes = build_planes_from_clusters(clusters)

    out_glb = Path(args.out_glb)
    tex_dir = Path(args.textures_dir)
    write_glb_with_textures(planes, out_glb, tex_dir)

    # 也把 plane 参数导出，便于调试
    Path("outputs/facades.json").write_text(json.dumps(planes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: {out_glb} (planes={len(planes)})")

if __name__ == "__main__":
    main()
