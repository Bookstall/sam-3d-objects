import os
import numpy as np

from loguru import logger

from sam3d_inference import Sam3DInference
from utils.image_utils import ImageUtils
from utils.file_utils import FileUtils


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


config_file = "/data/models/LLM-models-file/sam-3d-objects/checkpoints/pipeline.yaml"
sam3_model_path = "/data/models/LLM-models-file/sam3"
compile = False
sam3d_inference = Sam3DInference(config_file, sam3_model_path, compile)

# image_path = "./images/000000136466.jpg"
image_path = "/data/machine_learning/cpx/sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
image = ImageUtils.load_image(image_path)

# mask_path = "./images/masked_image_only_multiple_bounding_boxes.png"
mask_path = "/data/machine_learning/cpx/sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567/0.png"
mask = ImageUtils.load_mask(mask_path)

ImageUtils.display_image(image, [mask])

output = sam3d_inference(image, mask, seed=42)
logger.debug(f"keys of output: {output.keys()}")

# export gaussian splat (as point cloud)
# 导出高斯溅射（作为点云）
ply_path = "./images/gs.ply"    
FileUtils.save_ply(ply_path, output)
logger.debug(f"Exported PLY to: {ply_path}")

# export glb
glb_path = "./images/gs.glb"
FileUtils.save_glb(glb_path, output)
logger.debug(f"Exported GLB to: {glb_path}")

