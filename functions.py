from lxml import etree
from pathlib import Path
from PIL import Image
from typing import Callable, Tuple, Union
import cv2
import numpy as np


def get_boundingbox(path: Union[str, Path]
                    ) -> str:
    annotation = etree.parse(path)
    return tuple(int(annotation.xpath(f"/annotation/object/bndbox/{coord}").pop().text) for coord in ['xmin', 'ymin', 'xmax', 'ymax'])


def get_breed(path: Union[str, Path]
              ) -> str:
    annotation = etree.parse(path)
    return annotation.xpath("/annotation/object/name").pop().text


def _convert_pil_to_cv2(image: Union[Image.Image, np.ndarray]
                        ) -> np.ndarray:
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_np
    return image


def _convert_cv2_to_pil(image: Union[Image.Image, np.ndarray]
                        ) -> Image.Image:
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return Image.fromarray(image)
    return image


def _process_and_return_same_format(image: Union[Image.Image, np.ndarray],
                                    process_func: Callable,
                                    ) -> Union[Image.Image, np.ndarray]:
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        image = _convert_pil_to_cv2(image)

    result = process_func(image)

    if is_pil:
        return _convert_cv2_to_pil(result)
    return result


def whiten_image(image: Union[Image.Image, np.ndarray]
                 ) -> np.ndarray:
    def _whiten_cv2(img):
        img_float = img.astype(np.float32) / 255.0
        h, w, c = img_float.shape
        img_reshaped = img_float.reshape(h * w, c)
        img_centered = img_reshaped - np.mean(img_reshaped, axis=0)
        covariance = np.cov(img_centered, rowvar=False)
        epsilon = 1e-5
        U, S, V = np.linalg.svd(covariance)
        whitening_matrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T
        whitened = img_centered @ whitening_matrix.T
        whitened_img = whitened.reshape(h, w, c)
        whitened_img = (whitened_img - np.min(whitened_img)) / (np.max(whitened_img) - np.min(whitened_img)) * 255.0
        return whitened_img.astype(np.uint8)

    return _process_and_return_same_format(image, _whiten_cv2)


def equalize_histogram(image: Union[Image.Image, np.ndarray],
                       *,
                       clip_limit: float = 2.0,
                       grid_size: Tuple[int, int] = (8, 8),
                       ) -> Union[Image.Image, np.ndarray]:
    def _equalize_cv2(img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])
            return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(img)

    return _process_and_return_same_format(image, _equalize_cv2)


def resize_image(image: Union[Image.Image, np.ndarray],
                 *,
                 scaling_factor: Tuple[Union[float, int], Union[float, int]] = None,
                 target_size: Tuple[Union[float, int], Union[float, int]] = None,
                 interpolation: int = cv2.INTER_LINEAR,
                 ) -> Union[Image.Image, np.ndarray]:
    def _resize_cv2(img, scaling_factor=None, ts=target_size):
        if isinstance(scaling_factor, tuple):
            wscafac, hscafac = scaling_factor
            h, w = img.shape[:2]
            ts = (int(w * wscafac), int(h * hscafac))
        return cv2.resize(img, ts, interpolation=interpolation)

    return _process_and_return_same_format(image, lambda img: _resize_cv2(img, scaling_factor, target_size))


def normalize_image(image: Union[Image.Image, np.ndarray],
                    mean: Union[float, Tuple[float, float, float]] = 0.0,
                    std: Union[float, Tuple[float, float, float]] = 1.0,
                    ) -> Union[Image.Image, np.ndarray]:
    def _normalize_cv2(img, mean=mean, std=std):
        img_normalized = img.astype(np.float32) / 255.0
        if isinstance(mean, (tuple, list)):
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            for i in range(img.shape[2]):
                img_normalized[:, :, i] = (img_normalized[:, :, i] - mean[i]) / std[i]
        else:
            img_normalized = (img_normalized - mean) / std
        return img_normalized

    return _process_and_return_same_format(image, lambda img: _normalize_cv2(img, mean, std))


def convert_to_grayscale(image: Union[Image.Image, np.ndarray]
                         ) -> Union[Image.Image, np.ndarray]:
    def _grayscale_cv2(img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    return _process_and_return_same_format(image, _grayscale_cv2)


def apply_gaussian_blur(image: Union[Image.Image, np.ndarray],
                        kernel_size: Tuple[int, int] = (5, 5),
                        sigma_x: float = 0.0,
                        ) -> Union[Image.Image, np.ndarray]:
    def _blur_cv2(img):
        return cv2.GaussianBlur(img, kernel_size, sigmaX=sigma_x)

    return _process_and_return_same_format(image, _blur_cv2)


def mirror_image(image: Union[Image.Image, np.ndarray],
                 horizontal: bool = False,
                 vertical: bool = False,
                 ) -> Union[Image.Image, np.ndarray]:
    def _mirror_cv2(img):
        if horizontal and vertical:
            return cv2.flip(img, -1)
        elif horizontal:
            return cv2.flip(img, 1)
        elif vertical:
            return cv2.flip(img, 0)
        return img.copy()

    return _process_and_return_same_format(image, _mirror_cv2)


def crop_image(image: Union[Image.Image, np.ndarray],
               x: int, y: int,
               width: int, height: int,
               ) -> Union[Image.Image, np.ndarray]:
    def _crop_cv2(img):
        return img[y:y + height, x:x + width]

    return _process_and_return_same_format(image, _crop_cv2)
