# -*- coding: utf-8 -*-
"""
快速可视化脚本 - 展示DINOv3模型的注意力热力图
适用于 pooling_method='none' 的特征文件
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 配置 ==========
NPY_FILE = "/home/siton02/md0/crf/gjzw/ancient_images/look.npy"
IMAGE_FOLDER = "/home/siton02/md0/crf/gjzw/ancient_images/test"
NUM_IMAGES = 12  # 一次展示多少张图片
SAVE_PATH = "/home/siton02/md0/crf/gjzw/ancient_images/attention_grid.png"
USE_TRAIN_SET = True  # 如果数据分了训练/验证集，选择使用训练集还是验证集
# ===========================

def load_data(npy_path):
    """加载特征数据"""
    data = np.load(npy_path, allow_pickle=True).item()
    
    # 兼容两种数据格式
    if 'X_train' in data:
        # 训练/验证集分开的格式
        if USE_TRAIN_SET:
            features = data['X_train']
            image_names = data['names_train']
            labels = data.get('y_train', None)
            shapes = data.get('shapes_train', None)
            print("使用训练集数据")
        else:
            features = data['X_val']
            image_names = data['names_val']
            labels = data.get('y_val', None)
            shapes = data.get('shapes_val', None)
            print("使用验证集数据")
    else:
        # 未分割的格式
        features = data['features']
        image_names = data['image_names']
        labels = data.get('labels', None)
        shapes = data.get('shapes', None)
        print("使用完整数据集")
    
    pooling = data.get('pooling_method', 'unknown')
    
    print(f"特征形状: {features.shape}")
    print(f"池化方法: {pooling}")
    
    if pooling != 'none':
        print(f"\n⚠️ 警告: 当前使用的是 '{pooling}' 池化")
        print("   如果要看注意力图，需要重新提取特征:")
        print("   CONFIG['pooling_method'] = 'none'")
        return None
    
    # 检查shapes
    if shapes is None:
        print(f"\n⚠️ 错误: 数据文件中没有 'shapes' 信息")
        print("   这可能是因为特征提取时没有保存shapes字段")
        print("   请重新运行特征提取脚本，确保保存shapes信息")
        print("   或者修改特征提取脚本，参考方案1")
        return None
    
    return features, image_names, labels, shapes

def create_attention_heatmap(patch_features, h_patches, w_patches):
    """创建注意力热力图（使用特征范数）"""
    # 计算每个patch的L2范数作为注意力分数
    attention = np.linalg.norm(patch_features, axis=1)
    attention_map = attention.reshape(h_patches, w_patches)
    
    # 归一化
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    return attention_map

def overlay_heatmap(img_bgr, attention_map, alpha=0.5):
    """将热力图叠加到原图上"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 上采样注意力图
    attention_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 生成彩色热力图
    heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 叠加
    overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)
    
    return img_rgb, overlay

def visualize_grid(features, image_names, labels, shapes, image_folder, num_images, save_path):
    """以网格形式展示多张图片的注意力"""
    
    # 随机选择图片
    num_images = min(num_images, len(features))
    indices = np.random.choice(len(features), num_images, replace=False)
    
    # 计算网格布局
    ncols = 4
    nrows = (num_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\n正在生成 {num_images} 张注意力可视化...")
    
    for i, idx in enumerate(indices):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        
        img_name = image_names[idx]
        img_path = Path(image_folder) / img_name
        
        if not img_path.exists():
            ax.axis('off')
            ax.text(0.5, 0.5, f'图片不存在\n{img_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # 读取图片
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            ax.axis('off')
            continue
        
        # 生成注意力图
        patch_features = features[idx]
        h_patches, w_patches, _ = shapes[idx]
        attention_map = create_attention_heatmap(patch_features, h_patches, w_patches)
        
        # 叠加
        _, overlay = overlay_heatmap(img_bgr, attention_map, alpha=0.5)
        
        # 显示
        ax.imshow(overlay)
        title = img_name
        if labels is not None:
            title += f'\n类别: {labels[idx]}'
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        print(f"  [{i+1}/{num_images}] {img_name}")
    
    # 隐藏多余的子图
    for i in range(num_images, nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 保存到: {save_path}")
    
    # 也显示出来（如果在notebook环境）
    try:
        plt.show()
    except:
        pass
    finally:
        plt.close()

def main():
    print("="*60)
    print("DINOv3 注意力热力图可视化")
    print("="*60)
    
    # 加载数据
    result = load_data(NPY_FILE)
    if result is None:
        return
    
    features, image_names, labels, shapes = result
    
    # 生成可视化
    visualize_grid(
        features, image_names, labels, shapes,
        IMAGE_FOLDER, NUM_IMAGES, SAVE_PATH
    )
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)

if __name__ == "__main__":
    main()