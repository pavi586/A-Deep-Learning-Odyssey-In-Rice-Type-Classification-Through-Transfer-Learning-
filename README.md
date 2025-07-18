# A-Deep-Learning-Odyssey-In-Rice-Type-Classification-Through-Transfer-Learning-
dataset/
├── train/
│   ├── basmati/
│   ├── arborio/
│   ├── jasmine/
│   └── ...
├── val/
│   ├── basmati/
│   ├── arborio/
│   ├── jasmine/
│   └── ...
Sure! Here's a complete code example for a project titled "A Deep Learning Odyssey in Rice Type Classification Through Transfer Learning" using TensorFlow/Keras. It utilizes a pre-trained CNN model (like MobileNetV2) via transfer learning to classify different types of rice grains (e.g., Basmati, Arborio, Jasmine, etc.).


---

📁 Assumptions:

Dataset is organized as:


dataset/
├── train/
│   ├── basmati/
│   ├── arborio/
│   ├── jasmine/
│   └── ...
├── val/
│   ├── basmati/
│   ├── arborio/
│   ├── jasmine/
│   └── ...


---

✅ Full Code

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Set parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "dataset"  # Change this to your dataset path

# Data Augmentation and Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Load Pretrained Model (Transfer Learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Plot accuracy & loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.show()


