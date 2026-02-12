import os
import sys
import shutil
import subprocess
import numpy as np
import builtins

# not ideal to put that here
# os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["CUDA_HOME"] = os.environ.get(
    "CONDA_PREFIX",
    os.environ.get("CUDA_HOME", sys.prefix),
)
os.environ["LIDRA_SKIP_INIT"] = "true"

# 导入 sam-3d-objects 代码
# 使用绝对路径，避免出现 ImportError: No module named 'sam3d_objects'
sys.path.append("/data/machine_learning/cpx/sam-3d-objects")

from typing import Union, Optional, List, Callable
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
from copy import deepcopy
from pytorch3d.transforms import quaternion_multiply, quaternion_invert

# import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils
from sam3d_objects.utils.visualization import SceneVisualizer

from utils.sam3_utils import Sam3Utils


class Sam3DInference:
    WHITELIST_FILTERS = [
        lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
    ]

    BLACKLIST_FILTERS = [
        lambda target: get_method(target)
        in {
            builtins.exec,
            builtins.eval,
            builtins.__import__,
            os.kill,
            os.system,
            os.putenv,
            os.remove,
            os.removedirs,
            os.rmdir,
            os.fchdir,
            os.setuid,
            os.fork,
            os.forkpty,
            os.killpg,
            os.rename,
            os.renames,
            os.truncate,
            os.replace,
            os.unlink,
            os.fchmod,
            os.fchown,
            os.chmod,
            os.chown,
            os.chroot,
            os.fchdir,
            os.lchown,
            os.getcwd,
            os.chdir,
            shutil.rmtree,
            shutil.move,
            shutil.chown,
            subprocess.Popen,
            builtins.help,
        },
    ]

    """
    Sam3DInference 类
    """
    def __init__(self, config_file: str, sam3_model_path: str, compile: bool = False):
        self.config_file = config_file
        self.compile = compile
        # self.inference = Inference(config_file, compile)
        
        # load inference pipeline
        config = OmegaConf.load(config_file) # 加载配置文件
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile # 编译模型
        config.workspace_dir = os.path.dirname(config_file) # 设置工作目录
        self._check_hydra_safety(config, self.WHITELIST_FILTERS, self.BLACKLIST_FILTERS) # 检查配置文件是否安全
        self._pipeline: InferencePipelinePointMap = instantiate(config) # 实例化推理管道

        # load sam3 model
        self.sam3_utils = Sam3Utils(sam3_model_path, device="cuda")

    def _check_target(
        self,
        target: str,
        whitelist_filters: List[Callable],
        blacklist_filters: List[Callable],
    ):
        if any(filt(target) for filt in whitelist_filters):
            if not any(filt(target) for filt in blacklist_filters):
                return
        raise RuntimeError(
            f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
        )

    def _check_hydra_safety(
        self,
        config: DictConfig,
        whitelist_filters: List[Callable],
        blacklist_filters: List[Callable],
    ):
        """
        检查配置文件是否安全，防止注入恶意代码

        Args:
            config: 配置文件
            whitelist_filters: 白名单过滤器
            blacklist_filters: 黑名单过滤器
        """
        to_check = [config]
        while len(to_check) > 0:
            node = to_check.pop()
            if isinstance(node, DictConfig):
                to_check.extend(list(node.values()))
                if "_target_" in node:
                    self._check_target(node["_target_"], whitelist_filters, blacklist_filters)
            elif isinstance(node, ListConfig):
                to_check.extend(list(node))

    def _merge_mask_to_rgba(
        self, 
        image: Union[Image.Image, np.ndarray], 
        mask: Optional[Union[None, Image.Image, np.ndarray]]
    ) -> np.ndarray:
        """
        将输入的图像 image 和 mask 合并为 RGBA 格式

        Args:
            image: 输入的图像
            mask: 输入的 mask
        Returns:
            rgba_image: 合并后的 RGBA 图像
        """
        if image is not None and isinstance(image, Image.Image): # 将输入的图像转换为 numpy 数组
            image = np.asarray(image)
        
        if mask is not None and isinstance(mask, Image.Image): # 将输入的 mask 转换为 numpy 数组
            mask = np.asarray(mask)

        if mask is not None:
            # 如果提供了 mask，将其转为 uint8 并放大到 255（便于作为 alpha 通道）
            mask = mask.astype(np.uint8) * 255
            if mask.ndim == 2:
                # 如果 mask 是二维的，增加一个通道维度以匹配图像 shape
                mask = mask[..., None]
        else:
            # 如果没有 mask，将 alpha 通道设置为 255（全不透明）
            h, w = image.shape[:2]
            mask = np.full((h, w, 1), 255, dtype=np.uint8)
        # 将 mask 嵌入为 alpha 通道，拼接为 RGBA 格式
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmap=None,
    ) -> dict:
        """
        Sam3DInference 推理入口
        
        Args:
            image: 输入的图像
            mask: 输入的 mask
            seed: 随机种子
            pointmap: 点图
        Returns:
            dict: 推理结果
        """
        image = self._merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            image=image,
            mask=None,
            seed=seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )

