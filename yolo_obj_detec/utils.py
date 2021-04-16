from typing import List

import cv2 as cv
import os


def load_images_from_folder(folder_path: str) -> List:
    """Carrega as imagens de uma pasta.

    Args:
        folder_path: Path da pasta.

    Returns: Lista de imagens
    """
    images = []
    for filename in os.listdir(folder_path):
        img = cv.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
    return images
