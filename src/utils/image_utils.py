import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gradio as gr

from PIL import Image
from loguru import logger
from typing import Optional, List


class ImageUtils:
    """
    ImageUtils 类，用于处理图像相关的操作
    """
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Load image from path, return uint8 RGB array of shape (H, W, 3)
        
        Args:
            path: image path
        Returns:
            image: image numpy array
        """
        image = Image.open(path)
        image = np.array(image)
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def load_mask(path: str) -> np.ndarray:
        """
        Load mask from path, return binary mask of shape (H, W)
        
        Args:
            path: mask path
        Returns:
            mask: mask numpy array
        """
        mask = ImageUtils.load_image(path)
        mask = mask > 0
        if mask.ndim == 3: # 如果 mask 是三维的，则取最后一个通道
            mask = mask[..., -1]
        return mask

    @staticmethod
    def load_single_mask(folder_path, index=0, extension=".png"):
        masks = ImageUtils.load_masks(folder_path, [index], extension)
        return masks[0]

    @staticmethod
    def load_masks(folder_path, indices_list=None, extension=".png"):
        masks = []
        indices_list = [] if indices_list is None else list(indices_list)
        if not len(indices_list) > 0:  # get all all masks if not provided
            idx = 0
            while os.path.exists(os.path.join(folder_path, f"{idx}{extension}")):
                indices_list.append(idx)
                idx += 1

        for idx in indices_list:
            mask_path = os.path.join(folder_path, f"{idx}{extension}")
            assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
            mask = ImageUtils.load_mask(mask_path)
            masks.append(mask)
        return masks

    @staticmethod
    def display_image(image: np.ndarray, masks: Optional[List[np.ndarray]] = None):
        def imshow(image, ax):
            ax.axis("off")
            ax.imshow(image)

        grid = (1, 1) if masks is None else (2, 2)
        fig, axes = plt.subplots(*grid)
        if masks is not None:
            mask_colors = sns.color_palette("husl", len(masks))
            black_image = np.zeros_like(image[..., :3], dtype=float)  # background
            mask_display = np.copy(black_image)
            mask_union = np.zeros_like(image[..., :3])
            for i, mask in enumerate(masks):
                mask_display[mask] = mask_colors[i]
                mask_union |= mask[..., None] if mask.ndim == 2 else mask
            imshow(black_image, axes[0, 1])
            imshow(mask_display, axes[1, 0])
            imshow(image * mask_union, axes[1, 1])

        image_axe = axes if masks is None else axes[0, 0]
        imshow(image, image_axe)

        fig.tight_layout(pad=0)
        fig.show()

    @staticmethod
    def interactive_visualizer(ply_path: str):
        """
        Interactive visualizer for 3D Gaussian Splatting (ply file)

        Args:
            ply_path: 3D Gaussian Splatting ply file path
        """
        with gr.Blocks() as demo:
            gr.Markdown("# 3D Gaussian Splatting (black-screen loading might take a while)")
            gr.Model3D(
                value=ply_path,  # splat file
                label="3D Scene",
            )
        demo.launch(share=True)

