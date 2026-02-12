import matplotlib
import numpy as np
import requests
import torch

from loguru import logger
from PIL import Image
from transformers import Sam3Processor, Sam3Model, Sam3Config
from transformers import Sam3TrackerProcessor, Sam3TrackerModel, Sam3TrackerConfig
from typing import Literal, List, Union, Tuple


class Sam3Utils:
    def __init__(self, sam3_model_path: str, device: Literal["cuda", "cpu"] = "cuda"):
        """
        初始化 Sam3Utils 工具类: 用于处理 SAM3 模型的相关操作

        Args:
            sam3_model_path: SAM3 模型路径
            device: 设备类型, 默认使用 CUDA
        """
        # sam3_config = Sam3Config.from_pretrained(sam3_model_path, local_files_only=True)
        # self.sam3_model = Sam3Model.from_pretrained(sam3_model_path, config=sam3_config, local_files_only=True).to(device)
        # self.sam3_processor = Sam3Processor.from_pretrained(sam3_model_path, local_files_only=True)
        
        sam3_tracker_config = Sam3TrackerConfig.from_pretrained(sam3_model_path, local_files_only=True)
        self.sam3_tracker_model = Sam3TrackerModel.from_pretrained(sam3_model_path, config=sam3_tracker_config, local_files_only=True).to(device)
        self.sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(sam3_model_path, local_files_only=True)
        
        self.device = device


    def overlay_masks(self, image: Image.Image, masks: torch.Tensor) -> Image.Image:
        """
        在原始的 image 图片上面，根据 masks 绘制出不同的颜色，并返回新的 image 图片
        """
        image = image.convert("RGBA")
        masks = 255 * masks.cpu().numpy().astype(np.uint8)
        
        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

        for mask, color in zip(masks, colors):
            mask = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
        
        return image

    def overlay_masks_new(self, image: Image.Image, masks: torch.Tensor) -> Image.Image:
        """
        在原始的 image 图片上面，绘制所有的 masks，并返回新的 image 图片 (masks 绘制成粉色)
        """
        # masks 绘制成粉色
        image = image.convert("RGBA")
        masks = 255 * masks.cpu().numpy().astype(np.uint8)
        for mask in masks:
            mask = Image.fromarray(mask)
            # alpha_value = int(255 * 0.5)  # 127
            # overlay = Image.new("RGBA", image.size, (255, 0, 255, alpha_value))
            overlay = Image.new("RGBA", image.size, (255, 0, 255, 0))
            alpha = mask.point(lambda v: int(v * 0.5)) # 设置 alpha 通道为 0.5
            overlay.putalpha(alpha) # 设置 alpha 通道
            image = Image.alpha_composite(image, overlay)
        return image

    def get_masked_image_only(self, image: Image.Image, masks: torch.Tensor) -> Image.Image:
        """
        在原始的 image 图片上面，只保留 masks 的区域，其他区域设置为透明，并返回新的 image 图片
        """
        # 将图像转换为 RGBA 格式以支持透明通道
        image = image.convert("RGBA")
        
        # 将 masks 从 torch.Tensor 转换为 numpy 数组
        # masks 的形状应该是 [n_masks, H, W]，值为 0-1 之间的浮点数
        masks_np = masks.cpu().numpy()
        
        # 合并所有 masks：使用逻辑或操作，只要有一个 mask 在该位置为 True，就保留该位置
        # 先对每个 mask 进行二值化（阈值 0.5），然后合并
        combined_mask = np.zeros(masks_np.shape[1:], dtype=np.float32)  # [H, W]
        for mask in masks_np:
            # 将 mask 二值化（阈值 0.5）
            binary_mask = (mask > 0.5).astype(np.float32)
            # 使用逻辑或合并
            combined_mask = np.maximum(combined_mask, binary_mask)
        
        # 将合并后的 mask 转换为 0-255 的 uint8 格式，用作 alpha 通道
        alpha_channel = (combined_mask * 255).astype(np.uint8)
        
        # 将原始图像转换为 numpy 数组
        image_array = np.array(image)
        
        # 创建新的 RGBA 图像数组
        # RGB 通道保留原始图像的值，Alpha 通道使用合并后的 mask
        masked_image_array = image_array.copy()
        masked_image_array[:, :, 3] = alpha_channel  # 设置 alpha 通道
        
        # 将 numpy 数组转换回 PIL Image
        masked_image = Image.fromarray(masked_image_array, mode="RGBA")
        
        return masked_image


    def sam3_with_text_only_prompts(self):
        """
        Text-Only Prompts
        """
        # Load image
        image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        # Segment using text prompt
        inputs = self.sam3_processor(images=image, text="ear", return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs)

        # Post-process results
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        logger.debug(f"keys of results: {results.keys()}")
        # Results contain:
        # - masks: Binary masks resized to original image size
        # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
        # - scores: Confidence scores

        logger.debug(f"Found {len(results['masks'])} objects")

        masked_images = self.overlay_masks(image=image, masks=results['masks'])
        masked_images.save("masked_image.png")

        masked_image_only = self.get_masked_image_only(image=image, masks=results['masks'])
        masked_image_only.save("masked_image_only.png")


    def sam3_batch_with_text_only_prompts(self):
        """
        Batch Inference: Text-Only Prompts
        """
        cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
        kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
        images = [
            Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
            Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")
        ]
        # Different text prompt for each image
        text_prompts = ["ear", "dial"]

        inputs = self.sam3_processor(images=images, text=text_prompts, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs)

        # Post-process results for both images
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )

        print(f"Image 1: {len(results[0]['masks'])} objects found")
        print(f"Image 2: {len(results[1]['masks'])} objects found")


    def sam3_with_semantic_segmentation_output(self):
        """
        semantic segmentation output and instance masks
        
        与实例掩码一起的语义分割
        """
        # Load image
        image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        # Segment using text prompt: "ear"
        inputs = self.sam3_processor(images=image, text="ear", return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs)
        print(f"type of outputs: {type(outputs)}")

        # Instance segmentation masks
        instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

        # Semantic segmentation (single channel)
        semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]

        print(f"Instance masks: {instance_masks.shape}")
        print(f"Semantic segmentation: {semantic_seg.shape}")


    def sam3_with_single_bounding_box(
        self, 
        image_path: str, 
        box_xyxy: List[int], 
        input_boxes_label: List[int]
    ) -> Image.Image:
        """
        Single Bounding Box

        Args:
            image_path: 图片路径
            box_xyxy: 边界框坐标，格式为 [x1, y1, x2, y2]
            input_boxes_labels: 边界框标签，格式为 [1] = 正样本, [0] = 负样本
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
        input_boxes_labels = [input_boxes_label]  # 1 = positive box

        inputs = self.sam3_processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs)

        # Post-process results
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.3, # 默认值为 0.3
            mask_threshold=0.5, # 默认值为 0.5
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        logger.debug(f"keys of results: {results.keys()}")
        # Results contain:
        # - masks: Binary masks resized to original image size
        # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
        # - scores: Confidence scores

        logger.debug(f"Found {len(results['masks'])} objects")
        logger.debug(f"masks[0]: {results['masks'][0]}")
        logger.debug(f"boxes[0]: {results['boxes'][0]}")
        logger.debug(f"scores[0]: {results['scores'][0]}")

        masked_image_only = self.get_masked_image_only(image=image, masks=results['masks'])
        return masked_image_only

    def test_sam3_with_multiple_bounding_boxes(
        self, 
        image_path: str, 
        input_boxes: List[List[int]], 
        input_boxes_labels: List[List[int]]
    ) -> Image.Image:
        """
        Multiple Bounding Boxes: With positive and negative bounding boxes

        Args:
            image_path: 图片路径
            box_xyxy: 边界框坐标，格式为 [[x1, y1, x2, y2], [x1, y1, x2, y2]]
            input_boxes_labels: 边界框标签，格式为 [[1, 1], [1, 1]]
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        inputs = self.sam3_processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs)

        # Post-process results
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        logger.debug(f"keys of results: {results.keys()}")
        # Results contain:
        # - masks: Binary masks resized to original image size
        # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
        # - scores: Confidence scores

        logger.debug(f"Found {len(results['masks'])} objects")
        logger.debug(f"masks[0]: {results['masks'][0]}")
        logger.debug(f"boxes[0]: {results['boxes'][0]}")
        logger.debug(f"scores[0]: {results['scores'][0]}")

        masked_image_only = self.get_masked_image_only(image=image, masks=results['masks'])
        return masked_image_only

    def sam3_with_points(
        self,
        image: Union[Image.Image, str, np.ndarray],
        input_points: List[List[int]],
        input_labels: List[int],
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        使用点提示（point prompt）进行分割：
        - 点击坐标作为 positive(1) 或 negative(0) 点
        
        Args:
            image: PIL Image、路径或 RGB numpy 数组
            input_points: 点坐标列表 [[x1,y1], [x2,y2], ...]，像素坐标
            input_labels: 每个点的标签，1=正向点，0=负向点

        Returns:
            overlay_image: 在原图上叠加 mask 的 PIL 图（用于展示）
            mask_np: 二值 mask，形状 (H,W)，uint8 0/255，用于 3D 推理
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            image = image.convert("RGB") if image.mode != "RGB" else image
        
        w, h = image.size

        final_input_points = [[[[x, y] for x, y in input_points]]]
        logger.debug(f"shape of final_input_points: {len(final_input_points)}")
        logger.debug(f"final_input_points: {final_input_points}")
        final_input_labels = [[input_labels]]
        logger.debug(f"shape of final_input_labels: {len(final_input_labels)}")
        logger.debug(f"final_input_labels: {final_input_labels}")

        # 构建输入
        inputs = self.sam3_tracker_processor(
            images=image,
            input_points=final_input_points,
            input_labels=final_input_labels,
            return_tensors="pt"
        ).to(self.device)

        # SAM3 模型推理
        with torch.no_grad():
            outputs = self.sam3_tracker_model(**inputs)
        
        # 后处理结果
        results = self.sam3_tracker_processor.post_process_masks(
            masks=outputs.pred_masks.cpu(),
            original_sizes=inputs["original_sizes"]
        )[0]
        logger.debug(f"Generated {results.shape[1]} masks with shape {results.shape}")

        if results.shape[1] == 0: # 如果没有任何实例掩码，则返回全透明掩码
            logger.warning("没有找到任何实例掩码")
            mask_image_np = np.zeros((h, w), dtype=np.uint8)
            overlay = image.convert("RGBA")
            return overlay, mask_image_np
        
        # 叠加所有 mask 到原始图片中，用于展示 (masks 绘制成粉色)
        overlay = self.overlay_masks_new(image=image, masks=results[0])
        
        # 使用 get_masked_image_only 只保留 mask 的区域，其他区域设置为透明
        mask_image = self.get_masked_image_only(image=image, masks=results[0])
        # 将 mask_image 转换为 numpy 数组
        mask_image_np = np.array(mask_image)
        mask_image_np = mask_image_np.astype(np.uint8)
        mask_image_np = mask_image_np > 0
        if mask_image_np.ndim == 3:
            mask_image_np = mask_image_np[..., -1]

        return overlay, mask_image_np


