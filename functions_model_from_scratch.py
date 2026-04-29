from torchvision import transforms
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from PIL import Image
from typing import Tuple


def build_model_from_scratch(*, n_classes: int,
                             target_img_size: Tuple[int],
                             data_augm: keras.Sequential = None
                             ):
    inputs = keras.Input(shape=(*target_img_size, 3))

    x = inputs
    if data_augm:
        x = data_augm(inputs)

    x = layers.Rescaling(1./255)(x)

    x = layers.Conv2D(32, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)   # ← décommenter pour régulariser

    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def plot_accuracy_and_loss_values(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()


class MyKerasSequence(keras.utils.Sequence):
    """Classe personnalisée, compatible avec Keras, pour charger les images et les labels"""
    def __init__(self, paths, labels, batch_size, transform=None, target_size=None):
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(target_size),  # Ensure consistent image dimensions
            transforms.ToTensor(),
        ])
        # Fix Error: "Invalid dtype: str704": Convert string labels to numerical values if needed
        if isinstance(self.labels[0], str):
            unique_labels = np.unique(self.labels)
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.labels = np.array([self.label_to_idx[label] for label in self.labels])
        else:
            self.labels = np.array(self.labels)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.paths[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_images = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            # Convert from PyTorch NCHW to Keras NHWC format
            img_np = img.numpy()
            # PyTorch ToTensor() returns (C, H, W), Keras expects (H, W, C)
            if img_np.ndim == 3 and img_np.shape[0] == 3:  # Check if it's CHW format
                img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
            elif img_np.ndim == 3 and img_np.shape[2] == 3:  # Already in HWC format
                pass  # No transformation needed
            else:
                raise ValueError(f"Unexpected image shape: {img_np.shape}")
            batch_images.append(img_np.astype(np.float32))

        return np.array(batch_images), np.array(batch_labels)
