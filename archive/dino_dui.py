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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


class ContrastiveLearningEncoder(keras.Model):
    """对比学习编码器 - 改进版"""
    
    def __init__(self, input_dim, hidden_dim=1024, projection_dim=256, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        # Encoder (特征微调 - 保持维度不降低)
        self.encoder = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', name='encoder_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),  # 降低dropout
            layers.Dense(hidden_dim, activation='relu', name='encoder_2'),  # 保持维度
            layers.BatchNormalization(),
        ], name='encoder')
        
        # 残差连接投影（如果输入输出维度不同）
        if input_dim != hidden_dim and use_residual:
            self.residual_proj = layers.Dense(hidden_dim, use_bias=False)
        else:
            self.residual_proj = None
        
        # Projection head (用于对比学习，不用于下游任务)
        self.projection_head = keras.Sequential([
            layers.Dense(projection_dim, activation='relu', name='proj_1'),
            layers.BatchNormalization(),
            layers.Dense(projection_dim, name='proj_2'),
        ], name='projection_head')
    
    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        
        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                x_proj = self.residual_proj(x)
            else:
                x_proj = x
            encoded = encoded + x_proj  # 残差连接
        
        projected = self.projection_head(encoded, training=training)
        return encoded, projected


class ContrastiveLearner:
    """无标签对比学习训练器"""
    
    def __init__(self, feature_path, batch_size=256, temperature=0.1):
        """
        Args:
            feature_path: DINOv3特征文件路径
            batch_size: 批次大小（对比学习通常需要较大batch）
            temperature: 温度参数（控制对比损失的平滑度）
        """
        self.feature_path = feature_path
        self.batch_size = batch_size
        self.temperature = temperature
        
        self.encoder_model = None
        self.X_all = None
        self.y_all = None  # 标签仅用于评估
        self.feature_dim = None
        self.num_classes = None
        self.history = {'loss': [], 'val_loss': []}
        
    def load_features(self, ignore_labels=True):
        """加载特征（对比学习阶段不使用标签）"""
        print(f"\n{'='*60}")
        print("第一阶段：无标签对比学习")
        print(f"{'='*60}")
        print(f"\n加载特征文件: {self.feature_path}")
        
        data = np.load(self.feature_path, allow_pickle=True).item()
        
        # 合并所有数据
        if 'X_train' in data:
            self.X_all = np.concatenate([data['X_train'], data['X_val']], axis=0)
            self.y_all = np.concatenate([data['y_train'], data['y_val']], axis=0)
            self.num_classes = data['num_classes']
            self.feature_dim = data['feature_dim']
        elif 'features' in data:
            self.X_all = data['features']
            self.y_all = data['labels']
            self.num_classes = len(np.unique(self.y_all))
            self.feature_dim = self.X_all.shape[1]
        else:
            raise ValueError("无法识别的数据格式")
        
        if len(self.X_all.shape) == 1:
            self.X_all = self.X_all.reshape(-1, 1)
            self.feature_dim = 1
        
        print(f"  总样本数: {len(self.X_all)}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  类别数: {self.num_classes}")
        
        if ignore_labels:
            print(f"\n⚠️  对比学习阶段：标签被忽略，仅用于后续评估")
        
        return self.X_all
    
    def create_augmented_pairs(self, features):
        """
        创建增强样本对 - 改进版
        对于已经很强的DINOv3特征，使用更轻微的增强
        """
        batch_size = tf.shape(features)[0]
        
        # 策略1：轻微噪声（降低强度）
        noise_factor = 0.02  # 从0.1降低到0.02
        aug1 = features + tf.random.normal(tf.shape(features), mean=0, stddev=noise_factor)
        
        # 策略2：更轻微的dropout + 噪声
        dropout_rate = 0.05  # 从0.2降低到0.05
        mask = tf.cast(tf.random.uniform(tf.shape(features)) > dropout_rate, tf.float32)
        aug2 = features * mask + tf.random.normal(tf.shape(features), mean=0, stddev=noise_factor * 0.5)
        
        # 保持特征的原始范围（不做L2归一化，避免改变DINOv3特征的分布）
        return aug1, aug2
    
    def contrastive_loss(self, projections1, projections2):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
        SimCLR使用的对比损失
        """
        # 归一化投影
        projections1 = tf.nn.l2_normalize(projections1, axis=1)
        projections2 = tf.nn.l2_normalize(projections2, axis=1)
        
        # 计算相似度矩阵
        batch_size = tf.shape(projections1)[0]
        
        # 拼接正负样本
        projections = tf.concat([projections1, projections2], axis=0)
        
        # 计算所有样本对的相似度
        similarity_matrix = tf.matmul(projections, projections, transpose_b=True)
        similarity_matrix = similarity_matrix / self.temperature
        
        # 创建标签：对角线位置为正样本对
        labels = tf.range(batch_size)
        labels = tf.concat([labels + batch_size, labels], axis=0)
        
        # 移除自身相似度
        mask = tf.eye(2 * batch_size, dtype=tf.bool)
        similarity_matrix = tf.where(mask, -1e9, similarity_matrix)
        
        # 计算交叉熵损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=similarity_matrix
        )
        
        return tf.reduce_mean(loss)
    
    @tf.function
    def train_step(self, features):
        """单步训练"""
        with tf.GradientTape() as tape:
            # 创建增强对
            aug1, aug2 = self.create_augmented_pairs(features)
            
            # 前向传播
            _, proj1 = self.encoder_model(aug1, training=True)
            _, proj2 = self.encoder_model(aug2, training=True)
            
            # 计算对比损失
            loss = self.contrastive_loss(proj1, proj2)
        
        # 反向传播
        gradients = tape.gradient(loss, self.encoder_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder_model.trainable_variables))
        
        return loss
    
    def train_contrastive(self, epochs=100, hidden_dim=1024, projection_dim=256, 
                         learning_rate=0.0001, use_residual=True):
        """训练对比学习模型 - 改进版"""
        print(f"\n构建对比学习模型...")
        print(f"  输入维度: {self.feature_dim}")
        print(f"  隐藏维度: {hidden_dim} (保持不降维)")
        print(f"  投影维度: {projection_dim}")
        print(f"  温度参数: {self.temperature}")
        print(f"  残差连接: {use_residual}")
        print(f"  学习率: {learning_rate} (降低以避免破坏DINOv3特征)")
        
        # 构建模型
        self.encoder_model = ContrastiveLearningEncoder(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            use_residual=use_residual
        )
        
        # 优化器 - 使用更小的学习率
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices(self.X_all)
        dataset = dataset.shuffle(buffer_size=len(self.X_all))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # 验证集（用于监控，不用于训练）
        val_size = int(0.1 * len(self.X_all))
        X_train_unsup = self.X_all[:-val_size]
        X_val_unsup = self.X_all[-val_size:]
        
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train_unsup)
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train_unsup))
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(X_val_unsup)
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"\n开始对比学习训练...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  训练样本: {len(X_train_unsup)}")
        print(f"  验证样本: {len(X_val_unsup)}")
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_losses = []
            for batch in train_dataset:
                loss = self.train_step(batch)
                train_losses.append(loss.numpy())
            
            train_loss = np.mean(train_losses)
            
            # 验证
            val_losses = []
            for batch in val_dataset:
                aug1, aug2 = self.create_augmented_pairs(batch)
                _, proj1 = self.encoder_model(aug1, training=False)
                _, proj2 = self.encoder_model(aug2, training=False)
                loss = self.contrastive_loss(proj1, proj2)
                val_losses.append(loss.numpy())
            
            val_loss = np.mean(val_losses)
            
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.encoder_model.save_weights('best_contrastive_encoder.weights.h5')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        self.encoder_model.load_weights('best_contrastive_encoder.weights.h5')
        print(f"\n对比学习训练完成！最佳验证损失: {best_val_loss:.4f}")
        
        return self.encoder_model
    
    def plot_contrastive_history(self, save_path='contrastive_training_history.png'):
        """绘制对比学习训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Contrastive Learning Training History', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Contrastive Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n对比学习训练历史已保存: {save_path}")
    
    def extract_learned_features(self):
        """提取对比学习后的特征表示"""
        print(f"\n提取学习到的特征表示...")
        
        # 使用encoder（不是projection head）提取特征
        dataset = tf.data.Dataset.from_tensor_slices(self.X_all)
        dataset = dataset.batch(self.batch_size)
        
        learned_features = []
        for batch in dataset:
            encoded, _ = self.encoder_model(batch, training=False)
            learned_features.append(encoded.numpy())
        
        learned_features = np.concatenate(learned_features, axis=0)
        
        print(f"  原始特征形状: {self.X_all.shape}")
        print(f"  学习后特征形状: {learned_features.shape}")
        
        return learned_features


class ContrastiveEvaluator:
    """对比学习效果评估器（使用标签）"""
    
    def __init__(self, features_original, features_learned, labels, num_classes):
        self.features_original = features_original
        self.features_learned = features_learned
        self.labels = labels
        self.num_classes = num_classes
        
    def linear_probe_evaluation(self, train_ratio=0.1):
        """
        线性探测评估：冻结特征，只训练线性分类器
        这是评估表示质量的标准方法
        """
        print(f"\n{'='*60}")
        print("第二阶段：监督评估（线性探测）")
        print(f"{'='*60}")
        print(f"\n使用 {train_ratio:.0%} 数据训练线性分类器评估表示质量")
        
        results = {}
        
        for feature_type, features in [
            ('原始DINOv3特征', self.features_original),
            ('对比学习特征', self.features_learned)
        ]:
            print(f"\n{'='*50}")
            print(f"评估: {feature_type}")
            print(f"{'='*50}")
            
            # 划分数据
            X_train, X_val, y_train, y_val = train_test_split(
                features, self.labels,
                train_size=train_ratio,
                stratify=self.labels,
                random_state=42
            )
            
            print(f"  训练集: {len(X_train)} 样本")
            print(f"  验证集: {len(X_val)} 样本")
            
            # 训练线性分类器
            model = keras.Sequential([
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 计算类别权重
            class_weights_array = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights = dict(enumerate(class_weights_array))
            
            # 训练
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=64,
                class_weight=class_weights,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=15,
                        restore_best_weights=True,
                        verbose=0
                    )
                ],
                verbose=0
            )
            
            # 评估
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            
            # 预测
            predictions = model.predict(X_val, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # 分类报告
            report = classification_report(
                y_val, predicted_classes,
                target_names=[f'Class {i}' for i in range(self.num_classes)],
                output_dict=True,
                zero_division=0
            )
            
            # 混淆矩阵
            cm = confusion_matrix(y_val, predicted_classes)
            
            results[feature_type] = {
                'accuracy': val_accuracy,
                'loss': val_loss,
                'report': report,
                'confusion_matrix': cm,
                'history': history,
                'model': model
            }
            
            print(f"\n  验证准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"  验证损失: {val_loss:.4f}")
            
            # 打印每个类别的F1分数
            print(f"\n  各类别F1分数:")
            for i in range(self.num_classes):
                f1 = report[f'Class {i}']['f1-score']
                print(f"    Class {i}: {f1:.4f}")
        
        return results
    
    def knn_evaluation(self, k=5):
        """KNN评估：用最近邻分类评估特征质量"""
        print(f"\n{'='*60}")
        print(f"KNN评估 (k={k})")
        print(f"{'='*60}")
        
        results = {}
        
        for feature_type, features in [
            ('原始DINOv3特征', self.features_original),
            ('对比学习特征', self.features_learned)
        ]:
            print(f"\n评估: {feature_type}")
            
            # 使用90%数据训练KNN，10%数据测试
            X_train, X_test, y_train, y_test = train_test_split(
                features, self.labels,
                test_size=0.9,
                stratify=self.labels,
                random_state=42
            )
            
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            knn.fit(X_train, y_train)
            
            accuracy = knn.score(X_test, y_test)
            
            results[feature_type] = accuracy
            print(f"  KNN准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return results
    
    def clustering_quality_evaluation(self):
        """聚类质量评估：评估特征的可分离性"""
        print(f"\n{'='*60}")
        print("聚类质量评估")
        print(f"{'='*60}")
        
        from sklearn.cluster import KMeans
        
        results = {}
        
        for feature_type, features in [
            ('原始DINOv3特征', self.features_original),
            ('对比学习特征', self.features_learned)
        ]:
            print(f"\n评估: {feature_type}")
            
            # K-means聚类
            kmeans = KMeans(n_clusters=self.num_classes, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Silhouette分数（轮廓系数）
            silhouette = silhouette_score(features, cluster_labels)
            
            # Adjusted Rand Index（与真实标签的相似度）
            ari = adjusted_rand_score(self.labels, cluster_labels)
            
            results[feature_type] = {
                'silhouette_score': silhouette,
                'adjusted_rand_index': ari
            }
            
            print(f"  Silhouette分数: {silhouette:.4f} (越高越好，范围-1到1)")
            print(f"  Adjusted Rand Index: {ari:.4f} (越高越好，范围0到1)")
        
        return results
    
    def visualize_comparison(self, linear_probe_results, save_path='evaluation_comparison.png'):
        """可视化对比结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 准确率对比
        feature_types = list(linear_probe_results.keys())
        accuracies = [linear_probe_results[ft]['accuracy'] for ft in feature_types]
        
        axes[0, 0].bar(range(len(feature_types)), accuracies, color=['#3498db', '#e74c3c'])
        axes[0, 0].set_xticks(range(len(feature_types)))
        axes[0, 0].set_xticklabels(feature_types, rotation=15, ha='right')
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Linear Probe Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=11)
        
        # 2. 训练曲线对比
        for ft in feature_types:
            history = linear_probe_results[ft]['history']
            axes[0, 1].plot(history.history['val_accuracy'], label=ft, linewidth=2)
        
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12)
        axes[0, 1].set_title('Training Curves Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 混淆矩阵 - 原始特征
        cm_original = linear_probe_results[feature_types[0]]['confusion_matrix']
        sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                    xticklabels=range(self.num_classes),
                    yticklabels=range(self.num_classes))
        axes[1, 0].set_title(f'{feature_types[0]}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('True Label', fontsize=11)
        axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
        
        # 4. 混淆矩阵 - 对比学习特征
        cm_learned = linear_probe_results[feature_types[1]]['confusion_matrix']
        sns.heatmap(cm_learned, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
                    xticklabels=range(self.num_classes),
                    yticklabels=range(self.num_classes))
        axes[1, 1].set_title(f'{feature_types[1]}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('True Label', fontsize=11)
        axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n评估对比图已保存: {save_path}")
        
    def print_detailed_comparison(self, linear_probe_results):
        """打印详细对比报告"""
        print(f"\n{'='*60}")
        print("详细对比报告")
        print(f"{'='*60}")
        
        feature_types = list(linear_probe_results.keys())
        
        # 准确率对比
        print(f"\n1. 整体准确率对比:")
        for ft in feature_types:
            acc = linear_probe_results[ft]['accuracy']
            print(f"   {ft}: {acc:.4f} ({acc*100:.2f}%)")
        
        acc_diff = linear_probe_results[feature_types[1]]['accuracy'] - \
                   linear_probe_results[feature_types[0]]['accuracy']
        print(f"\n   提升: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
        
        if acc_diff < 0:
            print(f"\n   ⚠️  对比学习性能下降的可能原因：")
            print(f"   1. DINOv3特征已经很强，难以进一步改进")
            print(f"   2. 降维损失了信息")
            print(f"   3. 样本数量不足（仅{len(self.labels)}个样本）")
            print(f"\n   💡 建议：")
            print(f"   • 直接使用原始DINOv3特征（{linear_probe_results[feature_types[0]]['accuracy']*100:.2f}%已经很好）")
            print(f"   • 或在原始图像上做对比学习，而不是在DINOv3特征上")
            print(f"   • 或收集更多数据")
        else:
            print(f"\n   ✅ 对比学习成功提升了特征质量！")
        
        # 各类别对比
        print(f"\n2. 各类别F1分数对比:")
        print(f"   {'类别':<10} {'原始特征':<12} {'对比学习':<12} {'提升':<12}")
        print(f"   {'-'*50}")
        
        for i in range(self.num_classes):
            f1_original = linear_probe_results[feature_types[0]]['report'][f'Class {i}']['f1-score']
            f1_learned = linear_probe_results[feature_types[1]]['report'][f'Class {i}']['f1-score']
            improvement = f1_learned - f1_original
            
            print(f"   Class {i:<5} {f1_original:<12.4f} {f1_learned:<12.4f} {improvement:+.4f}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("="*80)
    print("基于DINOv3特征的无标签对比学习与监督评估")
    print("="*80)
    
    # ==================== 配置参数 ====================
    FEATURE_PATH = "/home/siton02/md0/crf/gjzw/ancient_images/train_features.npy"
    
    # 对比学习参数 - 优化版
    CONTRASTIVE_EPOCHS = 100
    CONTRASTIVE_BATCH_SIZE = 256
    TEMPERATURE = 0.8
    HIDDEN_DIM = 1024  # 保持与DINOv3相同的维度
    PROJECTION_DIM = 256  # 投影头维度
    LEARNING_RATE = 0.0001  # 降低学习率
    USE_RESIDUAL = True  # 使用残差连接
    
    # 评估参数
    TRAIN_RATIO = 0.8
    
    print("\n⚠️  注意：DINOv3特征已经非常强大（通常98%+准确率）")
    print("   如果对比学习没有提升，说明原始特征已经足够好")
    print("   建议：1) 直接使用原始特征  2) 在原始图像上做对比学习")
    print("   本代码已优化：保持维度、残差连接、轻微增强、低学习率")
    
    # ==================== 阶段1：无标签对比学习 ====================
    learner = ContrastiveLearner(
        feature_path=FEATURE_PATH,
        batch_size=CONTRASTIVE_BATCH_SIZE,
        temperature=TEMPERATURE
    )
    
    # 加载特征
    features_original = learner.load_features()
    
    # 训练对比学习模型
    encoder = learner.train_contrastive(
        epochs=CONTRASTIVE_EPOCHS,
        hidden_dim=HIDDEN_DIM,
        projection_dim=PROJECTION_DIM,
        learning_rate=LEARNING_RATE,
        use_residual=USE_RESIDUAL
    )
    
    # 绘制训练历史
    learner.plot_contrastive_history()
    
    # 提取学习到的特征
    features_learned = learner.extract_learned_features()
    
    # ==================== 阶段2：监督评估 ====================
    evaluator = ContrastiveEvaluator(
        features_original=features_original,
        features_learned=features_learned,
        labels=learner.y_all,
        num_classes=learner.num_classes
    )
    
    # 1. 线性探测评估（主要评估方法）
    linear_probe_results = evaluator.linear_probe_evaluation(train_ratio=TRAIN_RATIO)
    
    # 2. KNN评估
    knn_results = evaluator.knn_evaluation(k=5)
    
    # 3. 聚类质量评估
    clustering_results = evaluator.clustering_quality_evaluation()
    
    # 可视化对比
    evaluator.visualize_comparison(linear_probe_results)
    
    # 打印详细报告
    evaluator.print_detailed_comparison(linear_probe_results)
    
    # 保存学习到的特征
    np.save('contrastive_learned_features.npy', {
        'features_original': features_original,
        'features_learned': features_learned,
        'labels': learner.y_all,
        'num_classes': learner.num_classes
    })
    print(f"\n学习到的特征已保存: contrastive_learned_features.npy")
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)