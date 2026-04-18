# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io as sio
from pathlib import Path

# ---------- 配置 ----------
CONFIG = {
    "input_npy": "/home/siton02/md0/crf/gjzw/ancient_images/test_features.npy",  # 输入.npy文件路径
    "output_mat": "/home/siton02/md0/crf/gjzw/ancient_images/test_features.mat",  # 输出.mat文件路径
}

def print_data_info(data, prefix=""):
    """递归打印数据结构信息"""
    if isinstance(data, dict):
        print(f"{prefix}字典类型，包含 {len(data)} 个键:")
        for key in data.keys():
            print(f"{prefix}  - {key}")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}数组类型: shape={data.shape}, dtype={data.dtype}")
    elif isinstance(data, list):
        print(f"{prefix}列表类型: 长度={len(data)}")
    else:
        print(f"{prefix}其他类型: {type(data)}")

def convert_npy_to_mat(input_path, output_path):
    """
    将.npy文件转换为.mat文件
    
    Args:
        input_path: 输入.npy文件路径
        output_path: 输出.mat文件路径
    """
    print("="*60)
    print("NPY to MAT 格式转换器")
    print("="*60)
    
    # 检查输入文件
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    input_size = Path(input_path).stat().st_size / 1024 / 1024
    print(f"\n输入文件: {input_path}")
    print(f"文件大小: {input_size:.2f} MB")
    
    # 读取.npy文件
    print("\n正在读取.npy文件...")
    data = np.load(input_path, allow_pickle=True)
    
    # 如果是0维数组（即保存的字典），提取其内容
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    
    print("\n数据结构:")
    print_data_info(data)
    
    # 如果是字典，显示详细信息
    if isinstance(data, dict):
        print("\n详细信息:")
        for key, value in data.items():
            print(f"\n  [{key}]:")
            if isinstance(value, np.ndarray):
                print(f"    类型: ndarray")
                print(f"    形状: {value.shape}")
                print(f"    数据类型: {value.dtype}")
                if value.ndim <= 2:
                    mem_size = value.nbytes / 1024 / 1024
                    print(f"    内存大小: {mem_size:.2f} MB")
            elif isinstance(value, list):
                print(f"    类型: list")
                print(f"    长度: {len(value)}")
                if len(value) > 0:
                    print(f"    示例: {value[:3]}")
            else:
                print(f"    类型: {type(value)}")
                print(f"    值: {value}")
    
    # 准备保存的数据
    print("\n正在准备MAT格式数据...")
    
    if isinstance(data, dict):
        # 处理字典中的数据
        save_dict = {}
        for key, value in data.items():
            # 跳过不支持的类型或配置信息
            if key == 'config' and isinstance(value, dict):
                print(f"  跳过 '{key}' (配置字典)")
                continue
            
            if isinstance(value, np.ndarray):
                save_dict[key] = value
                print(f"  ✓ 添加 '{key}': {value.shape}")
            elif isinstance(value, list):
                # 尝试转换列表为数组
                try:
                    # 如果是字符串列表，保持为cell array
                    if len(value) > 0 and isinstance(value[0], str):
                        save_dict[key] = np.array(value, dtype=object)
                        print(f"  ✓ 添加 '{key}': 字符串列表 (长度={len(value)})")
                    else:
                        save_dict[key] = np.array(value)
                        print(f"  ✓ 添加 '{key}': {np.array(value).shape}")
                except:
                    print(f"  ✗ 跳过 '{key}': 无法转换列表")
            elif isinstance(value, (int, float, str)):
                save_dict[key] = value
                print(f"  ✓ 添加 '{key}': {value}")
            else:
                print(f"  ✗ 跳过 '{key}': 不支持的类型 {type(value)}")
    else:
        # 如果直接是数组，使用默认键名
        save_dict = {'data': data}
        print(f"  ✓ 添加 'data': {data.shape}")
    
    # 保存为.mat文件
    print(f"\n正在保存MAT文件到: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用'-v7.3'格式支持大文件（>2GB）
    try:
        sio.savemat(output_path, save_dict, format='5', do_compression=True)
        print("  使用 MATLAB v5 格式 (带压缩)")
    except:
        print("  文件过大，使用 MATLAB v7.3 格式...")
        import h5py
        with h5py.File(output_path, 'w') as f:
            for key, value in save_dict.items():
                f.create_dataset(key, data=value, compression='gzip')
        print("  使用 MATLAB v7.3 格式 (HDF5, 带压缩)")
    
    output_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n✓ 转换完成!")
    print(f"  输出文件: {output_path}")
    print(f"  文件大小: {output_size:.2f} MB")
    print(f"  压缩率: {(1 - output_size/input_size)*100:.1f}%")
    
    print("\n" + "="*60)
    print("在MATLAB中加载:")
    print("="*60)
    if output_path.endswith('.mat'):
        print(f"data = load('{Path(output_path).name}');")
        if isinstance(data, dict):
            for key in save_dict.keys():
                print(f"{key} = data.{key};")
    
    print("\n" + "="*60)

def main():
    try:
        convert_npy_to_mat(
            CONFIG['input_npy'],
            CONFIG['output_mat']
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
使用说明:

1. 修改 CONFIG 字典中的路径:
   - input_npy: 输入的.npy文件路径
   - output_mat: 输出的.mat文件路径

2. 运行:
   python npy_to_mat.py

3. 在MATLAB中加载:
   data = load('look.mat');
   
   % 如果保存的是训练/验证集划分
   X_train = data.X_train;
   y_train = data.y_train;
   X_val = data.X_val;
   y_val = data.y_val;
   
   % 如果保存的是完整数据
   features = data.features;
   labels = data.labels;
   image_names = data.image_names;

4. 注意事项:
   - 文件过大时会自动使用MATLAB v7.3格式（HDF5）
   - 字符串列表会转换为cell array
   - 配置字典会被跳过（因为MATLAB不支持嵌套结构）

需要安装的依赖:
pip install numpy scipy h5py
"""