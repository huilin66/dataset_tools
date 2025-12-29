from __future__ import annotations
import argparse
from pathlib import Path

from .rgbt_indexer import build_index, save_index
from .pipeline import build_poses, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="你的数据根目录（包含多个航线文件夹）")
    ap.add_argument("--out-dir", type=str, default="outputs")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    items = build_index(data_root)
    save_index(items, out_dir / "index.json")

    poses = build_poses(items)
    save_json(poses, out_dir / "poses_rgb.json")

    # pairs：基于 index 直接输出
    pairs = [{"rgb": x.rgb_path, "t": x.t_path} for x in items if x.t_path]
    save_json(pairs, out_dir / "pairs.json")

    print(f"Done. index={out_dir/'index.json'} poses={out_dir/'poses_rgb.json'} pairs={out_dir/'pairs.json'}")

if __name__ == "__main__":
    main()
