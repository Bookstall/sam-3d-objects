import imageio
import uuid
import os
import shutil
import sys

# import sam-3d-objects code
sys.path.append("/data/machine_learning/cpx/sam-3d-objects")


# 正确设置环境变量 PATH（需要先获取当前 PATH，然后拼接）
current_path = os.environ.get("PATH", "")
cuda_bin = "/data/CUDA/cuda-12.4/bin"
venv_bin = "/data/machine_learning/cpx/sam-3d-objects/.venv/bin"

# 将 CUDA bin 和 venv bin 添加到 PATH（确保 ninja 可执行文件能被找到）
# 注意：将 venv_bin 放在最前面，确保优先使用虚拟环境中的工具
new_path = f"{venv_bin}:{cuda_bin}:{current_path}"
os.environ["PATH"] = new_path

# 设置 CUDA 相关环境变量
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["CUDA_HOME"] = "/data/CUDA/cuda-12.4"
os.environ["MAX_JOBS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from inference import (
    Inference, 
    ready_gaussian_for_video_rendering, 
    render_video, 
    load_image, 
    load_masks, 
    display_image, 
    make_scene, 
    interactive_visualizer
)


# Load Pipeline Model
config_path = "/data/models/LLM-models-file/sam-3d-objects/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)

# Load Original Image and Mask Image
IMAGE_PATH = "./images/shutterstock_stylish_kidsroom_1640806567/image.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))
print(f"IMAGE_NAME: {IMAGE_NAME}")

# image ndarray (4480, 6720, 3) [[[234 214 187], ...]]
image = load_image(IMAGE_PATH)
# masks: list of ndarray, each is (4480, 6720) [[False False], [False, False]]
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")
# masks = None

# run model
outputs = [inference(image, mask, seed=42) for mask in masks]
print(f"length of outputs: {len(outputs)}")
print(f"keys of output: {outputs[0].keys()}")

# 保存为 GIF 动画
# render gaussian splat
# 渲染高斯溅射
scene_gs = make_scene(*outputs)
# export posed gaussian splatting (as point cloud)
scene_gs.save_ply(f"./gaussians/{IMAGE_NAME}_posed.ply")
scene_gs = ready_gaussian_for_video_rendering(scene_gs)
# export gaussian splatting (as point cloud)
scene_gs.save_ply(f"./gaussians/multi/{IMAGE_NAME}.ply")

# video: list
video = render_video(
    scene_gs,
    r=1,
    fov=60,
    pitch_deg=15,
    yaw_start_deg=-45,
    resolution=512,
)["color"]

# save video as gif
imageio.mimsave(
    os.path.join(f"./gaussians/multi/{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)



