import h5py
import numpy as np
from PIL import Image
import os
import math
import pandas as pd

def analyze_labels(mat_file):
    """
    分析标签结构
    """
    with h5py.File(mat_file, 'r') as f:
        TrC = np.array(f['TrC'])
        TeC = np.array(f['TeC'])
        
        print("训练集标签 TrC:")
        print(f"  形状: {TrC.shape}")
        print(f"  前10行:\n{TrC[:10]}")
        print(f"  第一列范围: [{TrC[:, 0].min()}, {TrC[:, 0].max()}]")
        print(f"  第二列唯一值: {np.unique(TrC[:, 1])}")
        
        print("\n测试集标签 TeC:")
        print(f"  形状: {TeC.shape}")
        print(f"  前10行:\n{TeC[:10]}")
        print(f"  第一列范围: [{TeC[:, 0].min()}, {TeC[:, 0].max()}]")
        print(f"  第二列唯一值: {np.unique(TeC[:, 1])}")
        
        return TrC, TeC

def save_images_with_labels(mat_file, output_dir='output_images'):
    """
    根据标签保存图片，并创建映射文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"正在读取文件: {mat_file}\n")
    
    with h5py.File(mat_file, 'r') as f:
        # 读取数据
        img = np.array(f['img'])
        TrC = np.array(f['TrC'])
        TeC = np.array(f['TeC'])
        num_training = int(np.array(f['num_training'])[0, 0])
        num_test = int(np.array(f['num_test'])[0, 0])
        
        # 推测图片尺寸
        num_pixels = img.shape[0]
        possible_sizes = []
        for h in range(1, int(math.sqrt(num_pixels)) + 500):
            if num_pixels % h == 0:
                w = num_pixels // h
                possible_sizes.append((h, w))
        best_size = min(possible_sizes, key=lambda x: abs(x[0] - x[1]))
        height, width = best_size
        
        print(f"图片尺寸: {height} × {width}")
        print(f"训练集: {num_training} 张")
        print(f"测试集: {num_test} 张\n")
        
        # 分析标签列的含义
        print("标签分析:")
        print(f"TrC 第一列 (图片索引): [{TrC[:, 0].min():.0f}, {TrC[:, 0].max():.0f}]")
        print(f"TrC 第二列 (类别标签): {np.unique(TrC[:, 1])}")
        print(f"TeC 第一列 (图片索引): [{TeC[:, 0].min():.0f}, {TeC[:, 0].max():.0f}]")
        print(f"TeC 第二列 (类别标签): {np.unique(TeC[:, 1])}\n")
        
        # 创建映射列表
        train_mapping = []
        test_mapping = []
        
        # 保存训练集
        train_dir = os.path.join(output_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        
        print("保存训练集...")
        for i, (img_idx, label) in enumerate(TrC):
            img_idx = int(img_idx) - 1  # MATLAB索引从1开始，Python从0开始
            label = int(label)
            
            # 提取图片
            image = img[:, img_idx].reshape(height, width)
            image_uint8 = (image * 255).astype(np.uint8)
            
            # 文件名包含标签信息
            filename = f'train_{i+1:04d}_idx{img_idx+1}_label{label}.png'
            save_path = os.path.join(train_dir, filename)
            Image.fromarray(image_uint8).save(save_path)
            
            train_mapping.append({
                '序号': i+1,
                '原始索引': img_idx+1,
                '标签': label,
                '文件名': filename
            })
            
            if (i + 1) % 200 == 0:
                print(f"  进度: {i+1}/{num_training}")
        
        # 保存测试集
        test_dir = os.path.join(output_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        
        print("\n保存测试集...")
        for i, (img_idx, label) in enumerate(TeC):
            img_idx = int(img_idx) - 1
            label = int(label)
            
            # 提取图片
            image = img[:, img_idx].reshape(height, width)
            image_uint8 = (image * 255).astype(np.uint8)
            
            # 文件名包含标签信息
            filename = f'test_{i+1:04d}_idx{img_idx+1}_label{label}.png'
            save_path = os.path.join(test_dir, filename)
            Image.fromarray(image_uint8).save(save_path)
            
            test_mapping.append({
                '序号': i+1,
                '原始索引': img_idx+1,
                '标签': label,
                '文件名': filename
            })
            
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{num_test}")
        
        # 保存映射文件
        df_train = pd.DataFrame(train_mapping)
        df_test = pd.DataFrame(test_mapping)
        
        df_train.to_csv(os.path.join(output_dir, 'train_mapping.csv'), index=False, encoding='utf-8-sig')
        df_test.to_csv(os.path.join(output_dir, 'test_mapping.csv'), index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 保存完成!")
        print(f"  输出目录: {output_dir}")
        print(f"  - train/ ({len(train_mapping)} 张)")
        print(f"  - test/ ({len(test_mapping)} 张)")
        print(f"  - train_mapping.csv (训练集映射表)")
        print(f"  - test_mapping.csv (测试集映射表)")
        
        # 显示标签统计
        print(f"\n标签分布:")
        print(f"训练集: {df_train['标签'].value_counts().sort_index().to_dict()}")
        print(f"测试集: {df_test['标签'].value_counts().sort_index().to_dict()}")

# 使用
if __name__ == "__main__":
    mat_file = "/home/siton02/md0/crf/gjzw/ancient(DATA)AL6.mat"
    output_dir = "/home/siton02/md0/crf/gjzw/ancient_images"
    
    # 先分析标签
    print("="*60)
    print("第一步：分析标签结构")
    print("="*60)
    analyze_labels(mat_file)
    
    print("\n" + "="*60)
    print("第二步：保存图片")
    print("="*60)
    input("\n按回车键继续...")
    
    save_images_with_labels(mat_file, output_dir)