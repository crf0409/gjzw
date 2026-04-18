import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import InceptionResNetV2  # 改为Inception-ResNet-v2
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
        初始化图像分类器（Inception-ResNet-v2架构）
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
        
        # 数据增强管道 - 对齐论文设置
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
            print(f"将以此尺寸作为目标格式。目标格式: {'横屏' if self.target_is_landscape else '竖屏'}\n")
        
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
    
    def build_model(self):
        # 注意: 修复了此方法内的缩进
        inputs = keras.Input(shape=(self.img_height, self.img_width, 1))
        # 转换为RGB（Inception-ResNet-v2需要3通道输入）
        x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(inputs)

        # 本地预训练权重路径
        local_weights_path = '/home/siton02/md0/crf/gjzw/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
        
        # 使用Inception-ResNet-v2作为基础模型
        base_model = InceptionResNetV2(
            include_top=False,
            weights=None,  # 先不加载权重
            input_tensor=x,
            input_shape=(self.img_height, self.img_width, 3),
            pooling='avg'  # 使用全局平均池化
        )
        
        # 加载本地预训练权重
        print(f"加载本地预训练权重: {local_weights_path}")
        try:
            base_model.load_weights(local_weights_path, by_name=True, skip_mismatch=True)
            print("✓ 预训练权重加载成功")
        except Exception as e:
            print(f"警告: 无法加载预训练权重 - {e}")
            print("将从随机初始化开始训练")
        
        # 微调策略：解冻部分顶层进行训练
        # 论文中的微调方法通常解冻顶部层
        base_model.trainable = True
        
        # 冻结前面的层，只微调后面的层（前80%冻结）
        fine_tune_at = int(len(base_model.layers) * 0.8)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        print(f"Inception-ResNet-v2: 总层数 {len(base_model.layers)}, 冻结前 {fine_tune_at} 层")

        # 添加分类头（相对简单，避免过拟合）
        x = base_model.output
        x = layers.BatchNormalization(name='bn_fc')(x)
        x = layers.Dropout(0.4, name='dropout1')(x)
        x = layers.Dense(256, activation='relu', 
                         kernel_regularizer=keras.regularizers.l2(0.00004),
                         name='fc1')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='InceptionResNetV2_Classifier')
        
        # 论文参数：
        # - Momentum: 0.9
        # - Initial Learning Rate: 0.01
        # - Weight Decay: 0.00004
        optimizer = keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9,
            nesterov=False
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("\n=== Inception-ResNet-v2 微调模型 ===")
        print(f"基础模型: Inception-ResNet-v2 (ImageNet预训练)")
        print(f"权重来源: 本地文件")
        print(f"模型大小: ~215MB")
        print(f"总参数: {model.count_params():,}")
        print(f"可训练参数: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        print(f"不可训练参数: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
        model.summary()
        
        return model

    def train(self, X_train, X_val, y_train, y_val, epochs=100):
        """
        训练模型 - 单阶段微调
        对齐论文Table 5参数设置
        """
        train_ds = self.create_dataset(X_train, y_train, is_training=True)
        val_ds = self.create_dataset(X_val, y_val, is_training=False)
        
        print("\n" + "="*60)
        print("Inception-ResNet-v2 微调训练")
        print("="*60)
        print("训练参数（对齐论文）：")
        print("  - Optimizer: SGD with Momentum=0.9")
        print("  - Initial Learning Rate: 0.01")
        print("  - Learning Rate Decay: 0.94 per 2 epochs")
        print("  - Weight Decay: 0.00004 (L2)")
        print(f"  - Batch Size: {self.batch_size}")
        print("="*60 + "\n")
        
        # 计算每2个epoch衰减一次
        steps_per_epoch = len(X_train) // self.batch_size
        decay_steps = steps_per_epoch * 2  # 每2个epoch
        
        # 学习率调度器 - 指数衰减（对齐论文decay=0.94）
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=decay_steps,
            decay_rate=0.94,
            staircase=True
        )
        
        # 更新优化器的学习率
        self.model.optimizer.learning_rate = lr_schedule
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),

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
        axes[0].set_title('Model Accuracy (Inception-ResNet-v2)', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss (Inception-ResNet-v2)', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("\n训练历史图已保存: training_history.png")
    
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
        print(f"  损失: {test_loss:.4f}")
        print(f"  准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  论文Inception-ResNet-v2 (128×128): 93.19%")
        print(f"  论文Inception-ResNet-v2 (64×64): 91.03%")
        
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
        plt.title('Confusion Matrix - Inception-ResNet-v2')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("混淆矩阵图已保存: confusion_matrix.png")

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
        plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
        print("\n预测结果已保存: predictions.png")

# 主程序
if __name__ == "__main__":
    DATA_DIR = "/home/siton02/md0/crf/gjzw/ancient_images" 
    BATCH_SIZE = 32      # 论文中batch size = 32
    EPOCHS = 100         # 论文中约77-82 epochs，这里设置100作为上限
    
    print("="*60)
    print("古代文字图像分类训练")
    print("架构: Inception-ResNet-v2 (Fine-tuning)")
    print("对齐论文: Classification of Architectural Heritage Images")
    print("        Using Deep Learning Techniques (MDPI 2017)")
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
    
    classifier.model.save('final_model_inception_resnet_v2.keras')
    print("\n最终模型已保存: final_model_inception_resnet_v2.keras")
    print("="*60)