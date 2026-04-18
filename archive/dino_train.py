import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

class DINOv3FeatureClassifier:
    """基于DINOv3预提取特征的分类器"""
    
    def __init__(self, feature_path, data_dir=None, batch_size=32, train_ratio=0.1):
        """
        初始化分类器
        Args:
            feature_path: DINOv3特征的.npy文件路径
            data_dir: 原始图像目录（仅用于可视化）
            batch_size: 批次大小
            train_ratio: 训练集比例（0.1表示1:9的训练/验证划分）
        """
        self.feature_path = feature_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.model = None
        self.history = None
        self.num_classes = None
        self.feature_dim = None
        self.class_weights = None
        
        # 数据容器
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.names_train = None
        self.names_val = None
        
    def load_features(self):
        """加载预提取的DINOv3特征并重新划分数据集"""
        print(f"\n加载特征文件: {self.feature_path}")
        
        data = np.load(self.feature_path, allow_pickle=True).item()
        
        # 收集所有数据
        if 'X_train' in data:
            # 已经划分好的数据，先合并
            print("检测到已划分的训练/验证集，将重新划分")
            X_all = np.concatenate([data['X_train'], data['X_val']], axis=0)
            y_all = np.concatenate([data['y_train'], data['y_val']], axis=0)
            
            names_train = data.get('names_train', None)
            names_val = data.get('names_val', None)
            if names_train is not None and names_val is not None:
                names_all = np.concatenate([names_train, names_val], axis=0)
            else:
                names_all = None
            
            self.num_classes = data['num_classes']
            self.feature_dim = data['feature_dim']
            
        elif 'features' in data:
            # 未划分的数据
            print("检测到未划分的数据")
            X_all = data['features']
            y_all = data['labels']
            names_all = data.get('image_names', None)
            self.num_classes = len(np.unique(y_all))
            self.feature_dim = X_all.shape[1] if len(X_all.shape) > 1 else 1
        else:
            raise ValueError("无法识别的数据格式")
        
        print(f"  总样本数: {len(X_all)}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  类别数: {self.num_classes}")
        
        # 按照指定比例重新划分数据集（分层抽样）
        print(f"\n重新划分数据集 (训练集比例: {self.train_ratio:.1%}, 验证集比例: {1-self.train_ratio:.1%})")
        
        if names_all is not None:
            self.X_train, self.X_val, self.y_train, self.y_val, self.names_train, self.names_val = \
                train_test_split(
                    X_all, y_all, names_all,
                    train_size=self.train_ratio,
                    stratify=y_all,  # 分层抽样，保持类别比例
                    random_state=42
                )
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = \
                train_test_split(
                    X_all, y_all,
                    train_size=self.train_ratio,
                    stratify=y_all,
                    random_state=42
                )
            self.names_train = None
            self.names_val = None
        
        print(f"  训练集: {len(self.X_train)} 样本 ({len(self.X_train)/len(X_all)*100:.1f}%)")
        print(f"  验证集: {len(self.X_val)} 样本 ({len(self.X_val)/len(X_all)*100:.1f}%)")
        
        # 确保特征是2D的
        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train.reshape(-1, 1)
            self.X_val = self.X_val.reshape(-1, 1)
            self.feature_dim = 1
        
        print(f"  训练集形状: {self.X_train.shape}")
        print(f"  验证集形状: {self.X_val.shape}")
        
        # 计算类别权重
        unique_classes = np.unique(self.y_train)
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=self.y_train
        )
        self.class_weights = dict(enumerate(class_weights_array))
        
        print(f"\n标签统计:")
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        val_dist = pd.Series(self.y_val).value_counts().sort_index()
        print(f"  训练集分布: {train_dist.to_dict()}")
        print(f"  验证集分布: {val_dist.to_dict()}")
        
        # 显示每个类别的样本数比较
        print(f"\n每个类别的划分详情:")
        for cls in range(self.num_classes):
            train_count = train_dist.get(cls, 0)
            val_count = val_dist.get(cls, 0)
            total = train_count + val_count
            print(f"  类别 {cls}: 训练={train_count:3d}, 验证={val_count:3d}, 总计={total:3d}")
        
        print(f"\n类别权重: {self.class_weights}\n")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def create_dataset(self, X, y, is_training=True):
        """创建TensorFlow数据集"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def build_model(self, architecture='mlp'):
        """
        构建分类模型
        Args:
            architecture: 模型架构
                - 'mlp': 多层感知机（推荐）
                - 'deep_mlp': 更深的MLP
                - 'residual': 带残差连接的网络
        """
        inputs = keras.Input(shape=(self.feature_dim,), name='feature_input')
        
        if architecture == 'mlp':
            # 标准MLP
            x = layers.Dense(512, activation='relu', name='fc1')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(256, activation='relu', name='fc2')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(128, activation='relu', name='fc3')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
        elif architecture == 'deep_mlp':
            # 更深的MLP
            x = layers.Dense(1024, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
        elif architecture == 'residual':
            # 带残差连接的网络
            # 第一层投影
            x = layers.Dense(512, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # 残差块1
            residual = x
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.2)(x)
            
            # 降维
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # 输出层
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print(f"\n模型结构 (架构: {architecture}):")
        model.summary()
        
        return model
    
    def train(self, epochs=100, architecture='mlp'):
        """训练模型"""
        if self.X_train is None:
            raise ValueError("请先调用 load_features() 加载数据")
        
        # 构建模型
        if self.model is None:
            self.build_model(architecture=architecture)
        
        # 创建数据集
        train_ds = self.create_dataset(self.X_train, self.y_train, is_training=True)
        val_ds = self.create_dataset(self.X_val, self.y_val, is_training=False)
        
        # 回调函数 - 对于小训练集，可能需要调整patience
        patience_epochs = max(15, int(epochs * 0.15))  # 动态调整patience
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience_epochs,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(8, int(patience_epochs * 0.4)),
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_dinov3_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\n开始训练...")
        print(f"训练配置: Epochs={epochs}, Batch_size={self.batch_size}, Patience={patience_epochs}")
        print(f"训练集大小: {len(self.X_train)} 样本")
        print(f"验证集大小: {len(self.X_val)} 样本")
        
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self, save_path='dinov3_training_history.png'):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可绘制")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 准确率
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 损失
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n训练历史图已保存: {save_path}")
        
        # 打印最佳结果
        best_epoch = np.argmax(self.history.history['val_accuracy'])
        best_val_acc = self.history.history['val_accuracy'][best_epoch]
        best_train_acc = self.history.history['accuracy'][best_epoch]
        
        print(f"\n最佳结果 (Epoch {best_epoch + 1}):")
        print(f"  训练准确率: {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
        print(f"  验证准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    def evaluate_on_validation(self):
        """在验证集上评估"""
        val_ds = self.create_dataset(self.X_val, self.y_val, is_training=False)
        
        val_loss, val_accuracy = self.model.evaluate(val_ds, verbose=0)
        print(f"\n验证集评估:")
        print(f"  损失: {val_loss:.4f}")
        print(f"  准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # 预测
        predictions = self.model.predict(val_ds, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 分类报告
        print("\n分类报告:")
        report = classification_report(
            self.y_val, 
            predicted_classes, 
            target_names=[f'Class {i}' for i in range(self.num_classes)],
            digits=4
        )
        print(report)
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_val, predicted_classes)
        print("\n混淆矩阵:")
        print(cm)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(self.num_classes), 
                    yticklabels=range(self.num_classes),
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('dinov3_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("混淆矩阵图已保存: dinov3_confusion_matrix.png")
        
        return val_loss, val_accuracy, predictions
    
    def evaluate_test_set(self, test_feature_path):
        """
        评估测试集
        Args:
            test_feature_path: 测试集特征.npy文件路径
        """
        print(f"\n加载测试集特征: {test_feature_path}")
        
        test_data = np.load(test_feature_path, allow_pickle=True).item()
        
        if 'X_test' in test_data:
            X_test = test_data['X_test']
            y_test = test_data['y_test']
            names_test = test_data.get('names_test', None)
        elif 'features' in test_data:
            X_test = test_data['features']
            y_test = test_data['labels']
            names_test = test_data.get('image_names', None)
        else:
            raise ValueError("无法识别的测试集数据格式")
        
        print(f"  测试集: {len(X_test)} 样本")
        print(f"  特征形状: {X_test.shape}")
        
        # 创建数据集
        test_ds = self.create_dataset(X_test, y_test, is_training=False)
        
        # 评估
        test_loss, test_accuracy = self.model.evaluate(test_ds)
        print(f"\n测试集结果:")
        print(f"  损失: {test_loss:.4f}")
        print(f"  准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # 预测
        predictions = self.model.predict(test_ds, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 分类报告
        print("\n分类报告:")
        report = classification_report(
            y_test, 
            predicted_classes, 
            target_names=[f'Class {i}' for i in range(self.num_classes)],
            digits=4
        )
        print(report)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, predicted_classes)
        print("\n混淆矩阵:")
        print(cm)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=range(self.num_classes), 
                    yticklabels=range(self.num_classes),
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('dinov3_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("测试集混淆矩阵图已保存: dinov3_test_confusion_matrix.png")
        
        return test_loss, test_accuracy
    
    def predict_and_visualize_samples(self, num_samples=9):
        """预测并可视化验证集样本（如果有原始图像）"""
        if self.data_dir is None or self.names_val is None:
            print("无法可视化：缺少原始图像目录或文件名信息")
            return
        
        # 随机选择样本
        indices = np.random.choice(len(self.X_val), size=min(num_samples, len(self.X_val)), replace=False)
        
        # 预测
        val_ds = self.create_dataset(self.X_val[indices], self.y_val[indices], is_training=False)
        predictions = self.model.predict(val_ds, verbose=0)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for idx, sample_idx in enumerate(indices[:9]):
            img_name = self.names_val[sample_idx]
            true_label = self.y_val[sample_idx]
            pred_label = np.argmax(predictions[idx])
            confidence = np.max(predictions[idx]) * 100
            
            # 尝试加载图像
            img_path = os.path.join(self.data_dir, 'train', img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_dir, 'test', img_name)
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('L')
                axes[idx].imshow(img, cmap='gray')
            else:
                # 如果找不到图像，显示特征的统计信息
                axes[idx].text(0.5, 0.5, f'Feature\nMean: {self.X_val[sample_idx].mean():.3f}\nStd: {self.X_val[sample_idx].std():.3f}',
                              ha='center', va='center', fontsize=10)
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)
            
            color = 'green' if pred_label == true_label else 'red'
            axes[idx].set_title(
                f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%\n{img_name}',
                color=color, fontsize=9
            )
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('dinov3_predictions.png', dpi=300, bbox_inches='tight')
        print("\n预测可视化已保存: dinov3_predictions.png")


# 主程序
if __name__ == "__main__":
    # ==================== 配置参数 ====================
    FEATURE_PATH = "/home/siton02/md0/crf/gjzw/ancient_images/train_features.npy"  # DINOv3特征文件
    TEST_FEATURE_PATH = "/home/siton02/md0/crf/gjzw/ancient_images/test_features.npy"  # 测试集特征（可选）
    DATA_DIR = "/home/siton02/md0/crf/gjzw/ancient_images"  # 原始图像目录（仅用于可视化）
    
    BATCH_SIZE = 64  # 特征训练可以用更大的batch size
    EPOCHS = 150
    ARCHITECTURE = 'mlp'  # 'mlp', 'deep_mlp', 'residual'
    TRAIN_RATIO = 0.7  # 训练集比例：0.1 = 10%训练 + 90%验证（1:9比例）
    
    print("="*60)
    print("基于DINOv3特征的古代文字图像分类")
    print(f"极端数据划分: {TRAIN_RATIO:.0%} 训练集 + {1-TRAIN_RATIO:.0%} 验证集")
    print("="*60)
    
    # 初始化分类器
    classifier = DINOv3FeatureClassifier(
        feature_path=FEATURE_PATH,
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO  # 设置极端的1:9划分
    )
    
    # 加载特征并重新划分
    X_train, X_val, y_train, y_val = classifier.load_features()
    
    # 训练模型
    classifier.train(epochs=EPOCHS, architecture=ARCHITECTURE)
    
    # 绘制训练历史
    classifier.plot_training_history()
    
    # 评估验证集
    classifier.evaluate_on_validation()
    
    # 如果有测试集特征，评估测试集
    if os.path.exists(TEST_FEATURE_PATH):
        classifier.evaluate_test_set(TEST_FEATURE_PATH)
    else:
        print(f"\n测试集特征文件不存在: {TEST_FEATURE_PATH}")
        print("如需评估测试集，请先提取测试集特征")
    
    # 可视化预测结果
    classifier.predict_and_visualize_samples()
    
    # 保存最终模型
    classifier.model.save('final_dinov3_classifier.keras')
    print("\n最终模型已保存: final_dinov3_classifier.keras")
    print("="*60)