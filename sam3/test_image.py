import matplotlib
import numpy as np
import requests
import torch

from loguru import logger
from PIL import Image
from transformers import Sam3Processor, Sam3Model, Sam3Config
from transformers import Sam3TrackerProcessor, Sam3TrackerModel, Sam3TrackerConfig
from transformers import Sam2Processor, Sam2Model, Sam2Config


device = "cuda" if torch.cuda.is_available() else "cpu"
sam3_model_path = "/data/models/LLM-models-file/sam3"

config = Sam3Config.from_pretrained(
    sam3_model_path,
    local_files_only=True
)
# config.attn_implementation = "flash_attention_2" # 显式指定使用 flash attention 2

model = Sam3Model.from_pretrained(
    sam3_model_path,
    config=config,
    local_files_only=True
).to(device)

processor = Sam3Processor.from_pretrained(
    sam3_model_path,
    local_files_only=True
)

sam3_tracker_config = Sam3TrackerConfig.from_pretrained(
    sam3_model_path,
    local_files_only=True
)
sam3_tracker_model = Sam3TrackerModel.from_pretrained(
    sam3_model_path,
    config=sam3_tracker_config,
    local_files_only=True
).to(device)
sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(
    sam3_model_path,
    local_files_only=True
)




def overlay_masks(image: Image.Image, masks: torch.Tensor) -> Image.Image:
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


def get_masked_image_only(image: Image, masks: torch.Tensor) -> Image:
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


def test_sam3_with_text_only_prompts():
    """
    Text-Only Prompts
    """
    # Load image
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # Segment using text prompt
    inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    print(f"keys of results: {results.keys()}")
    # Results contain:
    # - masks: Binary masks resized to original image size
    # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
    # - scores: Confidence scores

    print(f"Found {len(results['masks'])} objects")
    
    print(f"results: {results}")

    masked_images = overlay_masks(image=image, masks=results['masks'])
    masked_images.save("masked_image.png")

    masked_image_only = get_masked_image_only(image=image, masks=results['masks'])
    masked_image_only.save("masked_image_only.png")


def test_sam3_batch_with_text_only_prompts():
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

    inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results for both images
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )

    print(f"Image 1: {len(results[0]['masks'])} objects found")
    print(f"Image 2: {len(results[1]['masks'])} objects found")


def test_sam3_with_semantic_segmentation_output():
    """
    semantic segmentation output and instance masks
    
    与实例掩码一起的语义分割
    """
    # Load image
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # Segment using text prompt: "ear"
    inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    print(f"type of outputs: {type(outputs)}")

    # Instance segmentation masks
    instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

    # Semantic segmentation (single channel)
    semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]

    print(f"Instance masks: {instance_masks.shape}")
    print(f"Semantic segmentation: {semantic_seg.shape}")


def test_sam3_with_single_bounding_box():
    """
    Single Bounding Box
    """
    # Load image
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # Box in xyxy format: [x1, y1, x2, y2] in pixel coordinates
    # Example: laptop region
    box_xyxy = [100, 150, 500, 450]
    input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
    input_boxes_labels = [[1]]  # 1 = positive box

    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
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


def test_sam3_with_multiple_bounding_boxes():
    """
    Multiple Bounding Boxes: With positive and negative bounding boxes
    """
    # Load kitchen image
    kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
    kitchen_image = Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")

    # Define two positive boxes (e.g., dial and button on oven)
    # Boxes are in xyxy format [x1, y1, x2, y2] in pixel coordinates
    box1_xyxy = [59, 144, 76, 163]  # Dial box
    box2_xyxy = [87, 148, 104, 159]  # Button box
    input_boxes = [[box1_xyxy, box2_xyxy]]
    input_boxes_labels = [[1, 1]]  # Both positive

    inputs = processor(
        images=kitchen_image,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
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
    logger.debug(f"Found {len(results['boxes'])} boxes")
    logger.debug(f"boxes[0]: {results['boxes'][0]}")
    logger.debug(f"length of scores: {len(results['scores'])}")
    logger.debug(f"scores[0]: {results['scores'][0]}")

    masked_images = overlay_masks(image=kitchen_image, masks=results['masks'])
    masked_images.save("masked_image_multiple_bounding_boxes.png")

    masked_image_only = get_masked_image_only(image=kitchen_image, masks=results['masks'])
    masked_image_only.save("masked_image_only_multiple_bounding_boxes.png")


def test_sam3_with_single_point():
    """
    Single Point
    """
    # Load image
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # Points in xy format: [x, y] in pixel coordinates
    # points = [[100, 100], [200, 200], [300, 300]]
    # input_points = [[x, y] for x, y in points]
    # input_points_labels = [1, 1, 1]

    input_points = [[[[500, 375]]]] # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
    input_labels = [[[1]]] # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

    inputs = sam3_tracker_processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)
    logger.debug(f"keys of inputs: {inputs.keys()}")
    # Inputs contain:
    # - pixel_values: (batch, num_masks, height, width)
    # - original_sizes: (batch, 2)
    # - input_points
    # - input_labels
    
    with torch.no_grad():
        outputs = sam3_tracker_model(**inputs)
        logger.debug(f"keys of outputs: {outputs.keys()}")
        # Outputs contain:
        # - iou_scores: (batch, num_masks)
        # - pred_masks: (batch, num_masks, height, width)
        # - object_score_logits: (batch, num_masks)
        # - image_embeddings: (batch, 1, height, width)
    
    # (batch, num_masks, height, width)
    results = sam3_tracker_processor.post_process_masks(
        masks=outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"]
    )[0]
    
    # The model outputs multiple mask predictions ranked by quality score
    logger.debug(f"Generated {results.shape[1]} masks with shape {results.shape}")
    logger.debug(f"Found {results.shape[1]} objects")

    masked_images = overlay_masks(image=image, masks=results[0])
    masked_images.save("masked_image_with_points.png")

    masked_image_only = get_masked_image_only(image=image, masks=results[0])
    masked_image_only.save("masked_image_only_with_points.png")


def test_sam3_with_multiple_points():
    """
    Multiple Points
    """
    # Load image
    # image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    # image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    image = Image.open("/data/machine_learning/cpx/sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png").convert("RGB")
    
    # Points in xy format: [x, y] in pixel coordinates
    # input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
    # input_labels = [[[1, 1]]]  # Both positive clicks

    input_points = [[[[3500, 1534], [3814, 1803]]]]  # Multiple points for refinement
    input_labels = [[[1, 1]]]  # Both positive clicks

    inputs = sam3_tracker_processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)
    logger.debug(f"keys of inputs: {inputs.keys()}")

    with torch.no_grad():
        outputs = sam3_tracker_model(**inputs)
        logger.debug(f"keys of outputs: {outputs.keys()}")
        # Outputs contain:
        # - iou_scores: (batch, num_masks)
        # - pred_masks: (batch, num_masks, height, width)
        # - object_score_logits: (batch, num_masks)
        # - image_embeddings: (batch, 1, height, width)
    
    results = sam3_tracker_processor.post_process_masks(
        masks=outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"]
    )[0]
    logger.debug(f"Generated {results.shape[1]} masks with shape {results.shape}")
    logger.debug(f"Found {results.shape[1]} objects")
    
    masked_images = overlay_masks(image=image, masks=results[0])
    masked_images.save("masked_image_with_multiple_points.png")

    masked_image_only = get_masked_image_only(image=image, masks=results[0])
    masked_image_only.save("masked_image_only_with_multiple_points.png")


def test_sam2_with_multiple_points():
    """
    Multiple Points
    """
    sam2_model_path = "/data/models/LLM-models-file/sam2.1-hiera-large"
    sam2_config = Sam2Config.from_pretrained(
        sam2_model_path,
        local_files_only=True
    )
    sam2_model = Sam2Model.from_pretrained(
        sam2_model_path,
        config=sam2_config,
        local_files_only=True
    ).to(device)
    sam2_processor = Sam2Processor.from_pretrained(
        sam2_model_path,
        local_files_only=True
    )
    
    # Load image
    # image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    # image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    image = Image.open("/data/machine_learning/cpx/sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png").convert("RGB")
    
    # Points in xy format: [x, y] in pixel coordinates
    # input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
    # input_labels = [[[1, 1]]]  # Both positive clicks

    input_points = [[[[3500, 1534], [3814, 1803]]]]  # Multiple points for refinement
    input_labels = [[[1, 1]]]  # Both positive clicks

    inputs = sam2_processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)
    logger.debug(f"keys of inputs: {inputs.keys()}")

    with torch.no_grad():
        outputs = sam2_model(**inputs)
        logger.debug(f"keys of outputs: {outputs.keys()}")
        # Outputs contain:
        # - iou_scores: (batch, num_masks)
        # - pred_masks: (batch, num_masks, height, width)
        # - object_score_logits: (batch, num_masks)
        # - image_embeddings: (batch, 1, height, width)
    
    results = sam2_processor.post_process_masks(
        masks=outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"]
    )[0]
    logger.debug(f"Generated {results.shape[1]} masks with shape {results.shape}")
    logger.debug(f"Found {results.shape[1]} objects")
    
    masked_images = overlay_masks(image=image, masks=results[0])
    masked_images.save("masked_image_with_multiple_points.png")

    masked_image_only = get_masked_image_only(image=image, masks=results[0])
    masked_image_only.save("masked_image_only_with_multiple_points.png")




if __name__ == "__main__":
    # test_sam3_with_text_only_prompts()

    # test_sam3_batch_with_text_only_prompts()

    # test_sam3_with_semantic_segmentation_output()

    # test_sam3_with_single_bounding_box()

    # test_sam3_with_multiple_bounding_boxes()
    
    # test_sam3_with_single_point()

    # test_sam3_with_multiple_points()

    test_sam2_with_multiple_points()

