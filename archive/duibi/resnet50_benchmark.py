import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import ResNet50
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import warnings

# Set up Matplotlib for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AncientImageClassifier:
    """
    A class to classify ancient images using a fine-tuned ResNet50 model.
    """
    def __init__(self, data_dir, img_height=None, img_width=None, batch_size=32):
        """
        Initializes the image classifier with ResNet50 architecture.
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

        # Data augmentation pipeline
        self.data_augmentation = keras.Sequential([
            layers.RandomRotation(factor=0.15, fill_mode='constant', fill_value=0.0),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=0.0),
            layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode='constant', fill_value=0.0),
        ], name='data_augmentation')

    def load_data(self):
        """Loads and preprocesses the training data."""
        train_dir = os.path.join(self.data_dir, 'train')
        train_csv = os.path.join(self.data_dir, 'train_mapping.csv')

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

        if self.img_height is None or self.img_width is None:
            print("Detecting image dimensions...")
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

            print(f"Detected most common size (W×H): {self.img_width} × {self.img_height}")
            print(f"This will be the target format. Target orientation: {'Landscape' if self.target_is_landscape else 'Portrait'}\n")

        self.num_classes = len(np.unique(labels))
        print(f"Number of classes: {self.num_classes}")
        print(f"Label distribution:\n{pd.Series(labels).value_counts().sort_index()}\n")

        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        self.class_weights = dict(enumerate(class_weights_array))
        print(f"Class weights: {self.class_weights}\n")

        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels,
            test_size=0.3,
            random_state=42,
            stratify=labels
        )

        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Training set label distribution: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
        print(f"Validation set label distribution: {pd.Series(y_val).value_counts().sort_index().to_dict()}\n")

        return X_train, X_val, y_train, y_val

    def create_dataset(self, image_paths, labels, is_training=True):
        """Creates a TensorFlow dataset from image paths and labels."""
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
        """Applies data augmentation to an image."""
        image = self.data_augmentation(image, training=True)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def build_model(self):
        """Builds the ResNet50 model architecture without compiling it."""
        inputs = keras.Input(shape=(self.img_height, self.img_width, 1))
        # Convert grayscale to RGB as ResNet50 requires 3 channels
        x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(inputs)

        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=x,
            input_shape=(self.img_height, self.img_width, 3),
            pooling='avg'
        )
        print("✓ ResNet50 pre-trained weights loaded successfully (ImageNet)")

        base_model.trainable = True
        
        # Fine-tuning strategy: freeze early stages, train later stages
        for layer in base_model.layers:
            if 'conv5_block' in layer.name or 'conv4_block' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"ResNet50: Total layers {len(base_model.layers)}, Trainable layers {trainable_count}")

        # Add a classification head
        x = base_model.output
        x = layers.BatchNormalization(name='bn_fc')(x)
        x = layers.Dropout(0.3, name='dropout1')(x)
        x = layers.Dense(256, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.0001),
                         name='fc1')(x)
        x = layers.Dropout(0.2, name='dropout2')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='ResNet50_Classifier')
        self.model = model
        
        print("\n=== ResNet50 Fine-tuning Model Architecture ===")
        model.summary()
        
        return model

    def train(self, X_train, X_val, y_train, y_val, epochs=100):
        """Compiles and trains the model."""
        train_ds = self.create_dataset(X_train, y_train, is_training=True)
        val_ds = self.create_dataset(X_val, y_val, is_training=False)

        # Calculate total steps for Cosine Annealing
        steps_per_epoch = len(X_train) // self.batch_size
        total_steps = steps_per_epoch * epochs

        # Cosine Annealing learning rate schedule
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0001,
            decay_steps=total_steps,
            alpha=0.0  # Minimum learning rate
        )

        # Create the optimizer with the learning rate schedule
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999
        )
        
        # Compile the model with the optimizer
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\n" + "="*60)
        print("Starting ResNet50 Fine-tuning")
        print("="*60)
        print("Training Parameters:")
        print(f"  - Optimizer: Adam")
        print(f"  - Initial Learning Rate: 0.0001")
        print(f"  - Learning Rate Decay: Cosine Annealing")
        print(f"  - Batch Size: {self.batch_size}")
        print("="*60 + "\n")

        # FIX: Removed ReduceLROnPlateau callback as it's incompatible with CosineDecay
        # The learning rate is already controlled by the CosineDecay schedule
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model_resnet50.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            # ReduceLROnPlateau removed - incompatible with CosineDecay schedule
            # Learning rate is already controlled by CosineDecay
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
        """Plots and saves the training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy (ResNet50)', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss (ResNet50)', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('training_history_resnet50.png', dpi=300, bbox_inches='tight')
        print("\nTraining history plot saved: training_history_resnet50.png")

    def evaluate_test_set(self):
        """Evaluates the model on the test set."""
        test_dir = os.path.join(self.data_dir, 'test')
        test_csv = os.path.join(self.data_dir, 'test_mapping.csv')

        if not os.path.exists(test_csv):
            print("Test set mapping file not found.")
            return

        df = pd.read_csv(test_csv)
        print(f"\nEvaluating on test set: {len(df)} images")

        image_paths = [os.path.join(test_dir, row['文件名']) for _, row in df.iterrows() if os.path.exists(os.path.join(test_dir, row['文件名']))]
        labels = [row['标签'] - 1 for _, row in df.iterrows() if os.path.exists(os.path.join(test_dir, row['文件名']))]

        test_ds = self.create_dataset(np.array(image_paths), np.array(labels), is_training=False)

        test_loss, test_accuracy = self.model.evaluate(test_ds)
        print(f"\nTest Set Results:")
        print(f"  - Loss: {test_loss:.4f}")
        print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        predictions = self.model.predict(test_ds)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.array(labels)

        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns

        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=[f'Class {i}' for i in range(self.num_classes)]))

        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=range(self.num_classes),
                      yticklabels=range(self.num_classes))
        plt.title('Confusion Matrix - ResNet50')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_resnet50.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix plot saved: confusion_matrix_resnet50.png")

        return test_loss, test_accuracy

    def predict_sample_images(self, num_samples=9):
        """Predicts and visualizes a few sample images from the test set."""
        test_dir = os.path.join(self.data_dir, 'test')
        test_csv = os.path.join(self.data_dir, 'test_mapping.csv')

        if not os.path.exists(test_csv):
            print("Test set mapping file not found.")
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
        plt.savefig('predictions_resnet50.png', dpi=300, bbox_inches='tight')
        print("\nPrediction samples plot saved: predictions_resnet50.png")


if __name__ == "__main__":
    DATA_DIR = "/home/siton02/md0/crf/gjzw/ancient_images"
    BATCH_SIZE = 32
    EPOCHS = 100

    print("="*60)
    print("Ancient Character Image Classification")
    print("Architecture: ResNet50 (Fine-tuning)")
    print("="*60)

    classifier = AncientImageClassifier(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )

    X_train, X_val, y_train, y_val = classifier.load_data()

    # Build the model architecture
    classifier.build_model()

    # Compile the model with the dynamic learning rate and start training
    classifier.train(X_train, X_val, y_train, y_val, epochs=EPOCHS)

    classifier.plot_training_history()
    classifier.evaluate_test_set()
    classifier.predict_sample_images()

    classifier.model.save('final_model_resnet50.keras')
    print("\nFinal model saved: final_model_resnet50.keras")
    print("="*60)