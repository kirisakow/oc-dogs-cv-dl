from keras import layers
from keras.src.callbacks.history import History
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from typing import List, Tuple, Union
import keras.models
import keras.utils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
import seaborn as sns

BREED_ID_SPLITTER_REGEX_PTRN = re.compile(r'^n\d+-')


def build_model_from_scratch(*,
                             n_classes: int,
                             target_img_size: Tuple[int],
                             data_augm: keras.models.Sequential = None,
                             dropout_rate: float = None,
                             filters: List[int] = [32, 64],
                             kernel_size: int = 3,
                             experiment_name: str = 'CNN_model',
                             ) -> keras.models.Model:
    inputs = keras.Input(shape=(*target_img_size, 3))

    x = inputs
    if data_augm:
        x = data_augm(inputs)

    x = layers.Rescaling(1./255)(x)

    for n_filters in filters:
        x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name=experiment_name)


def plot_accuracy_and_loss_values(history: History,
                                  *,
                                  suptitle: str,
                                  legend_location: dict,
                                  ) -> None:
    plt.figure(figsize=(12, 4))
    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    for i, metric_name in enumerate(['accuracy', 'loss'], start=1):
        plt.subplot(1, 2, i)
        plt.plot(history.history[f'{metric_name}'])
        plt.plot(history.history[f'val_{metric_name}'])
        plt.title(f'Model {metric_name}')
        plt.ylabel(f'{metric_name.capitalize()}')
        plt.xlabel(None)
        plt.xlim(1, len(history.history[f'{metric_name}']))
        plt.legend(['Train', 'Validation'], loc=legend_location[metric_name])
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model: keras.models.Model,
                          X_test, y_test, class_labels,
                          title: str = None,
                          ) -> None:
    y_pred = model.predict(X_test)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    cm = confusion_matrix(y_true, y_pred_cls)
    plt.figure(figsize=(6, 4))
    class_labels = tuple(BREED_ID_SPLITTER_REGEX_PTRN.split(label_with_id)[1] for label_with_id in class_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='top')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if title:
        plt.title(title)
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
