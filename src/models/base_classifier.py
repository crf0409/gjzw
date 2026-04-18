# -*- coding: utf-8 -*-
"""
抽象基类 - 提取所有分类器的公共逻辑 (PyTorch 版)

所有图像分类器都继承此基类，只需实现 build_model() 方法。
"""

from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ClassificationHead(nn.Module):
    """通用分类头: BN -> Dropout -> FC -> ReLU -> Dropout -> FC(logits)"""

    def __init__(self, in_features, num_classes, fc_units=256,
                 dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout1),
            nn.Linear(in_features, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class AncientCharDataset(Dataset):
    """古建筑文字图像数据集"""

    def __init__(self, image_paths, labels, transform, target_size,
                 target_is_landscape, to_rgb=True):
        """
        Args:
            image_paths: 图片路径列表
            labels: 标签列表
            transform: torchvision transforms
            target_size: (height, width) 目标尺寸
            target_is_landscape: 目标图像是否为横向
            to_rgb: 是否转换为 RGB（3通道模型需要）
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size  # (H, W)
        self.target_is_landscape = target_is_landscape
        self.to_rgb = to_rgb

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # PIL 加载灰度图
        img = Image.open(img_path).convert('L')

        # 朝向矫正
        w, h = img.size
        current_is_landscape = w > h
        if self.target_is_landscape is not None and current_is_landscape != self.target_is_landscape:
            img = img.rotate(90, expand=True)

        # resize
        img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)

        # 转 RGB（3通道模型）
        if self.to_rgb:
            img = img.convert('RGB')

        # 应用 transform（包含 ToTensor）
        img = self.transform(img)

        return img, label


class BaseClassifier(ABC):
    """图像分类器抽象基类"""

    def __init__(self, config):
        """
        初始化分类器

        Args:
            config: 配置对象 (DictConfig)
        """
        self.config = config
        self.data_dir = config.paths.data
        self.batch_size = config.training.batch_size
        self.img_height = config.data.get('img_height')
        self.img_width = config.data.get('img_width')

        self.model = None
        self.history = None
        self.num_classes = None
        self.class_weights = None
        self.target_is_landscape = None
        self._to_rgb = True  # 子类可覆写为 False（如 custom_mlp 用灰度输入）

        # 设备管理
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        seed = config.project.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_transforms(self, is_training=True):
        """构建 torchvision transforms 管道（子类可覆写）"""
        aug = self.config.data.augmentation
        if is_training:
            return transforms.Compose([
                transforms.RandomRotation(
                    degrees=aug.rotation * 360,
                    fill=0,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(aug.translation, aug.translation),
                    scale=(1.0 - aug.zoom, 1.0 + aug.zoom),
                    fill=0,
                ),
                transforms.ColorJitter(
                    brightness=aug.brightness,
                    contrast=(aug.contrast_lower, aug.contrast_upper),
                ),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
            ])

    def load_data(self):
        """加载训练数据"""
        train_dir = os.path.join(self.data_dir, 'train')
        train_csv = os.path.join(self.data_dir, self.config.data.train_mapping)

        df = pd.read_csv(train_csv)
        print(f"Loading {len(df)} training images.")

        image_paths = []
        labels = []

        for idx, row in df.iterrows():
            img_path = os.path.join(train_dir, row['文件名'])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(row['标签'] - 1)

        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # 自动检测图像尺寸
        if self.img_height is None or self.img_width is None:
            print("Detecting image dimensions...")
            temp_sizes = []
            for img_path in image_paths[:200]:
                try:
                    with Image.open(img_path) as img:
                        temp_sizes.append(img.size)
                except Exception as e:
                    print(f"Warning: Skipping corrupted image {img_path}: {e}")
                    continue

            size_counts = Counter(temp_sizes)
            if not size_counts:
                raise ValueError("No valid images found to determine size.")

            most_common_size = size_counts.most_common(1)[0][0]
            w, h = most_common_size
            self.img_width, self.img_height = w, h
            self.target_is_landscape = self.img_width > self.img_height

            print(f"Detected most common size (W x H): {self.img_width} x {self.img_height}")
            print(f"Target orientation: {'Landscape' if self.target_is_landscape else 'Portrait'}\n")

        self.num_classes = len(np.unique(labels))
        print(f"Number of classes: {self.num_classes}")
        print(f"Label distribution:\n{pd.Series(labels).value_counts().sort_index()}\n")

        # 计算类别权重
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        self.class_weights = dict(enumerate(class_weights_array))
        print(f"Class weights: {self.class_weights}\n")

        # 划分训练集和验证集
        test_split = self.config.data.test_split
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels,
            test_size=test_split,
            random_state=self.config.project.seed,
            stratify=labels
        )

        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Training label distribution: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
        print(f"Validation label distribution: {pd.Series(y_val).value_counts().sort_index().to_dict()}\n")

        return X_train, X_val, y_train, y_val

    def create_dataset(self, image_paths, labels, is_training=True):
        """创建 PyTorch DataLoader"""
        transform = self._build_transforms(is_training=is_training)
        dataset = AncientCharDataset(
            image_paths=image_paths,
            labels=labels,
            transform=transform,
            target_size=(self.img_height, self.img_width),
            target_is_landscape=self.target_is_landscape,
            to_rgb=self._to_rgb,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=is_training,
            num_workers=4,
            pin_memory=True,
        )
        return loader

    @abstractmethod
    def build_model(self):
        """
        构建模型架构 - 子类必须实现

        Returns:
            nn.Module: 构建好的模型
        """
        pass

    def _create_optimizer(self):
        """
        创建优化器

        Returns:
            torch.optim.Optimizer
        """
        opt_config = self.config.optimizer
        lr = opt_config.learning_rate
        l2_reg = self.config.model.get('l2_reg', 0.0)

        if opt_config.type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=l2_reg
            )
        elif opt_config.type == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(), lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=l2_reg,
            )
        else:
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=l2_reg
            )

    def _create_scheduler(self, optimizer, epochs):
        """
        创建学习率调度器

        Returns:
            主调度器, ReduceLROnPlateau 调度器
        """
        opt_config = self.config.optimizer
        cfg = self.config.training

        scheduler = None
        if opt_config.schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=0.0
            )
        elif opt_config.schedule == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.94
            )

        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
        )

        return scheduler, reduce_lr

    def train(self, X_train, X_val, y_train, y_val, epochs=None):
        """训练模型"""
        if epochs is None:
            epochs = self.config.training.epochs

        train_loader = self.create_dataset(X_train, y_train, is_training=True)
        val_loader = self.create_dataset(X_val, y_val, is_training=False)

        model_name = self.config.model.get('name', 'model')

        print("\n" + "=" * 60)
        print(f"Starting {model_name} Training")
        print("=" * 60)
        print(f"Training Parameters:")
        print(f"  - Device: {self.device}")
        print(f"  - Optimizer: {self.config.optimizer.type}")
        print(f"  - Learning Rate: {self.config.optimizer.learning_rate}")
        print(f"  - Schedule: {self.config.optimizer.schedule}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Epochs: {epochs}")
        print("=" * 60 + "\n")

        # 移动模型到设备
        self.model.to(self.device)

        # 创建损失函数（带类别权重）
        if self.class_weights is not None:
            weights_tensor = torch.tensor(
                [self.class_weights[i] for i in range(self.num_classes)],
                dtype=torch.float32
            ).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        # 创建优化器和调度器
        optimizer = self._create_optimizer()
        scheduler, reduce_lr = self._create_scheduler(optimizer, epochs)

        # 训练历史
        history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': [],
        }

        # Early stopping 和 checkpoint
        best_val_acc = 0.0
        patience_counter = 0
        patience = self.config.training.early_stopping_patience
        output_dir = self.config.paths.outputs
        best_ckpt_path = os.path.join(output_dir, 'models', f'best_{model_name}.pth')
        os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)

        for epoch in range(epochs):
            # ---- Train phase ----
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                        leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(loss=loss.item(), acc=correct/total)

            train_loss = running_loss / total
            train_acc = correct / total

            # ---- Validation phase ----
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # 记录历史
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
                  f"lr: {current_lr:.2e}")

            # 学习率调度
            if scheduler is not None:
                scheduler.step()
            reduce_lr.step(val_loss)

            # Checkpoint + Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_ckpt_path)
                print(f"  -> Saved best model (val_accuracy: {val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} "
                          f"(patience={patience})")
                    # 恢复最佳权重
                    self.model.load_state_dict(torch.load(best_ckpt_path,
                                                          weights_only=True))
                    break

        self.history = history
        return history

    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        if self.history is None:
            print("No training history to plot")
            return

        model_name = self.config.model.get('name', 'model')

        if save_path is None:
            save_path = os.path.join(
                self.config.paths.outputs, 'figures',
                f'training_history_{model_name}.png'
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 准确率
        axes[0].plot(self.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title(f'Model Accuracy ({model_name})', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # 损失
        axes[1].plot(self.history['loss'], label='Train Loss')
        axes[1].plot(self.history['val_loss'], label='Val Loss')
        axes[1].set_title(f'Model Loss ({model_name})', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nTraining history plot saved: {save_path}")

    def evaluate_test_set(self, save_dir=None):
        """评估测试集"""
        test_dir = os.path.join(self.data_dir, 'test')
        test_csv = os.path.join(self.data_dir, self.config.data.test_mapping)

        if not os.path.exists(test_csv):
            print("Test set mapping file not found.")
            return None, None

        model_name = self.config.model.get('name', 'model')

        if save_dir is None:
            save_dir = os.path.join(self.config.paths.outputs, 'figures')
        os.makedirs(save_dir, exist_ok=True)

        df = pd.read_csv(test_csv)
        print(f"\nEvaluating on test set: {len(df)} images")

        image_paths = []
        labels = []
        for _, row in df.iterrows():
            img_path = os.path.join(test_dir, row['文件名'])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(row['标签'] - 1)

        test_loader = self.create_dataset(
            np.array(image_paths), np.array(labels), is_training=False
        )

        self.model.to(self.device)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []

        with torch.no_grad():
            for images, batch_labels in test_loader:
                images = images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, batch_labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                test_total += batch_labels.size(0)
                test_correct += predicted.eq(batch_labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())

        test_loss = test_loss / test_total
        test_accuracy = test_correct / test_total

        print(f"\nTest Set Results:")
        print(f"  - Loss: {test_loss:.4f}")
        print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        predicted_classes = np.array(all_predictions)
        true_classes = np.array(labels)

        # 分类报告
        print("\nClassification Report:")
        print(classification_report(
            true_classes, predicted_classes,
            target_names=[f'Class {i}' for i in range(self.num_classes)]
        ))

        # 混淆矩阵
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(self.num_classes),
                    yticklabels=range(self.num_classes))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        cm_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved: {cm_path}")

        return test_loss, test_accuracy

    def predict_sample_images(self, num_samples=9, save_path=None):
        """预测并可视化样本图片"""
        test_dir = os.path.join(self.data_dir, 'test')
        test_csv = os.path.join(self.data_dir, self.config.data.test_mapping)

        if not os.path.exists(test_csv):
            print("Test set mapping file not found.")
            return

        model_name = self.config.model.get('name', 'model')

        if save_path is None:
            save_path = os.path.join(
                self.config.paths.outputs, 'figures',
                f'predictions_{model_name}.png'
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df = pd.read_csv(test_csv)
        samples = df.sample(n=min(num_samples, len(df)), random_state=42)

        self.model.to(self.device)
        self.model.eval()

        val_transform = self._build_transforms(is_training=False)

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()

        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx >= 9:
                break

            img_path = os.path.join(test_dir, row['文件名'])
            img_display = Image.open(img_path).convert('L')

            # 预处理单张图片
            img = Image.open(img_path).convert('L')
            w, h = img.size
            current_is_landscape = w > h
            if self.target_is_landscape is not None and current_is_landscape != self.target_is_landscape:
                img = img.rotate(90, expand=True)
            img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
            if self._to_rgb:
                img = img.convert('RGB')
            img_tensor = val_transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                probs = torch.softmax(output, dim=1)
                predicted_class = probs.argmax(dim=1).item() + 1
                confidence = probs.max().item() * 100

            true_class = row['标签']

            axes[idx].imshow(img_display, cmap='gray')
            color = 'green' if predicted_class == true_class else 'red'
            axes[idx].set_title(
                f'True: {true_class} | Pred: {predicted_class}\nConf: {confidence:.1f}%',
                color=color, fontsize=10
            )
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nPrediction samples saved: {save_path}")

    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            model_name = self.config.model.get('name', 'model')
            path = os.path.join(
                self.config.paths.outputs, 'models',
                f'final_{model_name}.pth'
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"\nModel saved: {path}")
