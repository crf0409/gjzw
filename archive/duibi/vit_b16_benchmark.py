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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

class AncientImageClassifier:
    def __init__(self, data_dir, img_height=None, img_width=None, batch_size=32):
        """
        初始化图像分类器（ViT-Base/16架构）
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        self.num_classes = None
        self.class_weights = None
        self.target_is_landscape = None
        
        # 数据增强管道
        self.data_augmentation = keras.Sequential([
            layers.RandomRotation(factor=0.15, fill_mode='constant', fill_value=0.0),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=0.0),
            layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode='constant', fill_value=0.0),
        ], name='data_augmentation')
        
    def load_data(self):
        """加载训练数据"""
        train_dir = os.path.join(self.data_dir, 'train')
        train_csv = os.path.join(self.data_dir, 'train_mapping.csv')
        
        df = pd.read_csv(train_csv)
        print(f"加载 {len(df)} 张训练图片")
        
        image_paths = []
        labels = []
        
        for idx, row in df.iterrows():
            img_path = os.path.join(train_dir, row['文件名'])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(row['标签'] - 1)
        
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        if self.img_height is None or self.img_width is None:
            print("检测图像尺寸...")
            from collections import Counter
            
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

            print(f"检测到最常见的尺寸 (宽×高): {self.img_width} × {self.img_height}")
            print(f"将以此尺寸作为目标格式。目标格式: {'横屏' if self.target_is_landscape else '竖屏'}")
            print(f"注意: ViT-Base/16推荐输入尺寸为224×224或384×384，当前使用 {self.img_width}×{self.img_height}\n")
        
        self.num_classes = len(np.unique(labels))
        print(f"类别数: {self.num_classes}")
        print(f"标签分布:\n{pd.Series(labels).value_counts().sort_index()}\n")
        
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        self.class_weights = dict(enumerate(class_weights_array))
        print(f"类别权重: {self.class_weights}\n")
        
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, 
            test_size=0.3, 
            random_state=42,
            stratify=labels
        )
        
        print(f"训练集: {len(X_train)} 张")
        print(f"验证集: {len(X_val)} 张")
        print(f"训练集标签分布: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
        print(f"验证集标签分布: {pd.Series(y_val).value_counts().sort_index().to_dict()}\n")
        
        return X_train, X_val, y_train, y_val
    
    def create_dataset(self, image_paths, labels, is_training=True):
        """创建TensorFlow数据集"""
        def load_and_preprocess_image(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=1)
            
            shape = tf.shape(img)
            height, width = shape[0], shape[1]
            
            current_is_landscape = width > height
            need_rotate = tf.not_equal(current_is_landscape, self.target_is_landscape)
            
            img = tf.cond(
                need_rotate,
                lambda: tf.image.rot90(img, k=1),
                lambda: img
            )
            
            img = tf.image.resize(img, [self.img_height, self.img_width])
            img = tf.cast(img, tf.float32) / 255.0
            
            return img, label
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.shuffle(1000)
            dataset = dataset.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def augment(self, image, label):
        """数据增强"""
        image = self.data_augmentation(image, training=True)
        
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def build_vit_model(self, patch_size=16, projection_dim=768, num_heads=12, 
                        transformer_layers=12, mlp_head_units=[2048]):
        """
        构建Vision Transformer (ViT) Base/16模型
        
        参数:
            patch_size: 图像patch的大小 (16×16)
            projection_dim: Transformer的隐藏维度 (768)
            num_heads: 多头注意力的头数 (12)
            transformer_layers: Transformer编码器层数 (12)
            mlp_head_units: 分类头的隐藏单元
        """
        inputs = keras.Input(shape=(self.img_height, self.img_width, 1))
        # 转换为RGB
        x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(inputs)
        
        # === Patch Embedding ===
        # 将图像分割成patches并线性投影
        num_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        
        # 使用卷积层提取patches
        patches = layers.Conv2D(
            filters=projection_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='patch_embedding'
        )(x)
        
        # Reshape: (batch, h/p, w/p, dim) -> (batch, num_patches, dim)
        patches = layers.Reshape(
            (num_patches, projection_dim),
            name='patches_reshape'
        )(patches)
        
        # === Position Embedding ===
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim,
            name='position_embedding'
        )(positions)
        
        # 添加位置编码
        encoded_patches = patches + position_embedding
        
        # === Transformer Encoder ===
        for i in range(transformer_layers):
            # Layer Normalization 1
            x1 = layers.LayerNormalization(epsilon=1e-6, name=f'ln1_{i}')(encoded_patches)
            
            # Multi-Head Self-Attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=projection_dim // num_heads,
                dropout=0.1,
                name=f'mha_{i}'
            )(x1, x1)
            
            # Skip connection 1
            x2 = layers.Add(name=f'add1_{i}')([attention_output, encoded_patches])
            
            # Layer Normalization 2
            x3 = layers.LayerNormalization(epsilon=1e-6, name=f'ln2_{i}')(x2)
            
            # MLP (Feed-Forward Network)
            mlp_dim = projection_dim * 4  # 通常是4倍
            x3 = layers.Dense(mlp_dim, activation='gelu', name=f'mlp_dense1_{i}')(x3)
            x3 = layers.Dropout(0.1, name=f'mlp_dropout1_{i}')(x3)
            x3 = layers.Dense(projection_dim, name=f'mlp_dense2_{i}')(x3)
            x3 = layers.Dropout(0.1, name=f'mlp_dropout2_{i}')(x3)
            
            # Skip connection 2
            encoded_patches = layers.Add(name=f'add2_{i}')([x3, x2])
        
        # === MLP Head (分类器) ===
        representation = layers.LayerNormalization(epsilon=1e-6, name='final_ln')(encoded_patches)
        
        # 使用[CLS] token或全局平均池化
        # 这里使用全局平均池化
        representation = layers.GlobalAveragePooling1D(name='global_avg_pool')(representation)
        
        # 分类头
        features = representation
        for units in mlp_head_units:
            features = layers.Dense(units, activation='gelu', name=f'mlp_head_{units}')(features)
            features = layers.Dropout(0.3, name=f'mlp_head_dropout_{units}')(features)
        
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(features)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='ViT_Base_16_Classifier')
        
        return model, num_patches
    
    def build_model(self):
        """构建ViT-Base/16模型 (仅定义架构)"""
        print("="*60)
        print("构建Vision Transformer (ViT) Base/16模型")
        print("="*60)
        
        model, num_patches = self.build_vit_model(
            patch_size=16,
            projection_dim=768,
            num_heads=12,
            transformer_layers=12,
            mlp_head_units=[2048, 1024]
        )
        
        # Model is only built here. It will be compiled in the `train` method
        # right before fitting, once the learning rate schedule is created.
        self.model = model

        print("\n=== Vision Transformer (ViT) Base/16 模型 ===")
        print(f"架构类型: Transformer (非CNN!)")
        print(f"核心创新: Self-Attention机制处理图像patches")
        print(f"Patch大小: 16×16")
        print(f"Patches数量: {num_patches}")
        print(f"隐藏维度: 768")
        print(f"Transformer层数: 12")
        print(f"注意力头数: 12")
        print(f"总参数: {model.count_params():,}")
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        print(f"可训练参数: {trainable_params:,}")
        print("\nViT特点:")
        print("  ✓ 完全基于Transformer架构")
        print("  ✓ 无卷积层！")
        print("  ✓ 全局自注意力机制")
        print("  ✓ 可扩展性强")
        print("  ✓ 适合大规模数据集")
        print("\n⚠️ 注意:")
        print("  - ViT需要大量数据才能发挥最佳性能")
        print("  - 从头训练需要大规模预训练")
        print("  - 本实现为从零训练（无ImageNet预训练）")
        print("  - 建议数据增强要充分")
        model.summary()
        
        return model

    def train(self, X_train, X_val, y_train, y_val, epochs=150):
        """训练模型"""
        train_ds = self.create_dataset(X_train, y_train, is_training=True)
        val_ds = self.create_dataset(X_val, y_val, is_training=False)
        
        print("\n" + "="*60)
        print("Vision Transformer (ViT) Base/16 训练")
        print("="*60)
        print("训练参数：")
        print("  - Optimizer: AdamW (Adam with Weight Decay)")
        print("  - Initial Learning Rate: 0.0001")
        print("  - Weight Decay: 0.0001")
        print("  - Learning Rate Warmup + Cosine Decay")
        print(f"  - Batch Size: {self.batch_size}")
        print("  - Dropout: 0.1 (Transformer) + 0.3 (分类头)")
        print(f"  - Epochs: {epochs} (ViT需要更长训练)")
        print("="*60)
        print("⚠️ ViT训练建议:")
        print("  - 需要更多epochs (150+)")
        print("  - 需要学习率warmup")
        print("  - 数据增强很重要")
        print("  - 小数据集上可能不如CNN")
        print("="*60 + "\n")
        
        # Warmup + Cosine Decay学习率调度
        steps_per_epoch = len(X_train) // self.batch_size
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * 10  # 前10个epoch做warmup
        
        # 创建warmup + cosine decay调度器
        class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, learning_rate_base, total_steps, warmup_steps, name=None):
                super().__init__()
                self.learning_rate_base = learning_rate_base
                self.total_steps = total_steps
                self.warmup_steps = warmup_steps
                self.name = name

            def __call__(self, step):
                step_float = tf.cast(step, dtype=tf.float32)
                warmup_steps_float = tf.cast(self.warmup_steps, dtype=tf.float32)
                total_steps_float = tf.cast(self.total_steps, dtype=tf.float32)

                learning_rate = tf.cond(
                    step < self.warmup_steps,
                    lambda: self.learning_rate_base * (step_float / warmup_steps_float),
                    lambda: self.learning_rate_base * 0.5 * (
                        1.0 + tf.cos(
                            np.pi * (step_float - warmup_steps_float) /
                            (total_steps_float - warmup_steps_float)
                        )
                    )
                )
                return learning_rate

            def get_config(self):
                """Returns the configuration of the schedule."""
                return {
                    "learning_rate_base": self.learning_rate_base,
                    "total_steps": self.total_steps,
                    "warmup_steps": self.warmup_steps,
                    "name": self.name
                }
        
        lr_schedule = WarmUpCosineDecay(
            learning_rate_base=0.0001,
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )
        
        # Create the optimizer with the learning rate schedule
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # FIX: Removed ReduceLROnPlateau callback as it's incompatible with LearningRateSchedule
        # The learning rate is already controlled by the WarmUpCosineDecay schedule
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,  # ViT需要更多耐心
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model_vit_base16.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            # ReduceLROnPlateau removed - incompatible with LearningRateSchedule
            # Learning rate is already controlled by WarmUpCosineDecay schedule
        ]
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        self.history = history
        return history
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy (ViT-Base/16)', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss (ViT-Base/16)', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_vit_base16.png', dpi=300, bbox_inches='tight')
        print("\n训练历史图已保存: training_history_vit_base16.png")
    
    def evaluate_test_set(self):
        """评估测试集"""
        test_dir = os.path.join(self.data_dir, 'test')
        test_csv = os.path.join(self.data_dir, 'test_mapping.csv')
        
        if not os.path.exists(test_csv):
            print("测试集映射文件不存在")
            return
        
        df = pd.read_csv(test_csv)
        print(f"\n评估测试集: {len(df)} 张图片")
        
        image_paths = [os.path.join(test_dir, row['文件名']) for _, row in df.iterrows() if os.path.exists(os.path.join(test_dir, row['文件名']))]
        labels = [row['标签'] - 1 for _, row in df.iterrows() if os.path.exists(os.path.join(test_dir, row['文件名']))]

        test_ds = self.create_dataset(np.array(image_paths), 
                                      np.array(labels), 
                                      is_training=False)
        
        test_loss, test_accuracy = self.model.evaluate(test_ds)
        print(f"\n测试集结果:")
        print(f"  损失: {test_loss:.4f}")
        print(f"  准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  ViT-Base/16 ImageNet-21k预训练 + ImageNet微调: ~77.9%")
        print(f"  注意: 本模型从头训练，未使用预训练权重")
        
        predictions = self.model.predict(test_ds)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.array(labels)
        
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns

        print("\n分类报告:")
        print(classification_report(true_classes, predicted_classes, 
                                      target_names=[f'Class {i}' for i in range(self.num_classes)]))
        
        print("\n混淆矩阵:")
        cm = confusion_matrix(true_classes, predicted_classes)
        print(cm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(self.num_classes), 
                    yticklabels=range(self.num_classes))
        plt.title('Confusion Matrix - ViT-Base/16')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_vit_base16.png', dpi=300, bbox_inches='tight')
        print("混淆矩阵图已保存: confusion_matrix_vit_base16.png")

        return test_loss, test_accuracy
    
    def predict_sample_images(self, num_samples=9):
        """预测并可视化样本图片"""
        test_dir = os.path.join(self.data_dir, 'test')
        test_csv = os.path.join(self.data_dir, 'test_mapping.csv')
        
        if not os.path.exists(test_csv):
            print("测试集映射文件不存在")
            return
        
        df = pd.read_csv(test_csv)
        samples = df.sample(n=min(num_samples, len(df)), random_state=42)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx >= 9: break
            
            img_path = os.path.join(test_dir, row['文件名'])
            img_display = Image.open(img_path).convert('L')
            
            img_tensor, _ = self.create_dataset(
                np.array([img_path]), 
                np.array([0]), 
                is_training=False
            ).as_numpy_iterator().next()

            prediction = self.model.predict(img_tensor, verbose=0)
            predicted_class = np.argmax(prediction) + 1
            true_class = row['标签']
            confidence = np.max(prediction) * 100
            
            axes[idx].imshow(img_display, cmap='gray')
            color = 'green' if predicted_class == true_class else 'red'
            axes[idx].set_title(f'True: {true_class} | Pred: {predicted_class}\nConf: {confidence:.1f}%',
                                  color=color, fontsize=10)
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions_vit_base16.png', dpi=300, bbox_inches='tight')
        print("\n预测结果已保存: predictions_vit_base16.png")

# 主程序
if __name__ == "__main__":
    DATA_DIR = "/home/siton02/md0/crf/gjzw/ancient_images" 
    BATCH_SIZE = 16
    EPOCHS = 150  # ViT需要更多epochs
    
    print("="*60)
    print("古代文字图像分类训练")
    print("架构: Vision Transformer (ViT) Base/16")
    print("参数量: ~86 Million")
    print("推荐输入尺寸: 224×224 或 384×384")
    print("="*60)
    print("ViT革命性特点:")
    print("  - 完全抛弃卷积层！")
    print("  - 将图像看作序列的patches")
    print("  - 使用纯Transformer架构")
    print("  - 全局自注意力机制")
    print("="*60)
    print("⚠️ 重要提示:")
    print("  - 本实现为从头训练（无预训练权重）")
    print("  - ViT在小数据集上可能不如CNN")
    print("  - 建议数据集 >10,000 张图片")
    print("  - 需要更长的训练时间")
    print("="*60)
    
    classifier = AncientImageClassifier(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )
    
    X_train, X_val, y_train, y_val = classifier.load_data()
    
    classifier.build_model()
    
    classifier.train(X_train, X_val, y_train, y_val, epochs=EPOCHS)
    
    classifier.plot_training_history()
    
    classifier.evaluate_test_set()
    
    classifier.predict_sample_images()
    
    classifier.model.save('final_model_vit_base16.keras')
    print("\n最终模型已保存: final_model_vit_base16.keras")
    print("="*60)