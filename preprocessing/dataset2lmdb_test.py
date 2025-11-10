import os
import argparse
import lmdb
import yaml
from pathlib import Path
from tqdm import tqdm

# 可选：支持 numpy / cv2 解码验证
# import numpy as np
# import cv2

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
NPY_EXTS = {'.npy'}
VALID_EXTS = IMG_EXTS | NPY_EXTS


def file_to_binary(file_path: Path) -> bytes:
    # 对于 .npy，直接保存文件原始字节（包含 header，便于还原）
    with open(file_path, 'rb') as f:
        return f.read()


def iter_files(root: Path):
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def compute_map_size_bytes(root: Path, overhead_ratio: float = 1.5) -> int:
    total = 0
    for p in iter_files(root):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return int(total * overhead_ratio)


def create_lmdb_dataset(source_folder: Path, lmdb_path: Path, dataset_name: str,
                        map_size_bytes: int, commit_interval: int = 2000):
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size_bytes,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
    )

    txn = env.begin(write=True)
    count = 0

    files = list(iter_files(source_folder))
    for fp in tqdm(files, desc=f'Writing LMDB ({dataset_name})'):
        rel = fp.relative_to(source_folder).as_posix()  # 使用正斜杠
        key = f'{dataset_name}/{rel}'.encode('utf-8')
        value = file_to_binary(fp)
        txn.put(key, value)
        count += 1

        if count % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.sync()
    env.close()

    print(f'Done. {count} files written to {lmdb_path} (map_size={map_size_bytes/1024**3:.2f} GB)')


def main():
    parser = argparse.ArgumentParser(description='Convert RGB dataset to LMDB.')
    parser.add_argument('--source', type=str, help='path to dataset root (e.g., datasets/rgb/FaceForensics++)')
    parser.add_argument('--dataset_name', type=str, help='dataset name for keys')
    parser.add_argument('--output', type=str, help='output directory for lmdb files')
    parser.add_argument('--dataset_size', type=float, default=-1,
                        help='map_size in GB; if <=0, auto-compute')
    parser.add_argument('--config', type=str, default='./preprocessing/config.yaml',
                        help='fallback config file (section: to_lmdb)')
    args = parser.parse_args()

    # 允许从 config.yaml 回退读取
    if not (args.source and args.dataset_name and args.output):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)['to_lmdb']
        source = Path(args.source or (Path(cfg['dataset_root_path']['default']) / cfg['dataset_name']['default']))
        dataset_name = args.dataset_name or cfg['dataset_name']['default']
        output_dir = Path(args.output or cfg['output_lmdb_dir']['default'])
    else:
        source = Path(args.source)
        dataset_name = args.dataset_name
        output_dir = Path(args.output)

    assert source.exists(), f'source not found: {source}'

    lmdb_path = output_dir / f'{dataset_name}_lmdb'
    if args.dataset_size and args.dataset_size > 0:
        map_size_bytes = int(args.dataset_size * 1024 * 1024 * 1024)
    else:
        map_size_bytes = compute_map_size_bytes(source, overhead_ratio=1.5)

    create_lmdb_dataset(source, lmdb_path, dataset_name, map_size_bytes)


if __name__ == '__main__':
    main()