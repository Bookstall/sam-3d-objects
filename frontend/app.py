"""
基于 Gradio 的 SAM 3 图像分割 + SAM-3D Objects 3D 生成前端

支持：点击添加正/负点生成 Mask、生成 3D、导出 PLY/GLB
"""
import os
import sys
import tempfile
import numpy as np
import gradio as gr

from typing import Optional
from PIL import Image, ImageDraw
from loguru import logger

# 确保可导入 src 与项目根
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from src.utils.file_utils import FileUtils
from src.utils.sam3_utils import Sam3Utils
from src.sam3d_inference import Sam3DInference


# 与 sam3d_inference 一致的环境设置
os.environ["CUDA_HOME"] = os.environ.get(
    "CONDA_PREFIX",
    os.environ.get("CUDA_HOME", sys.prefix),
)
os.environ["LIDRA_SKIP_INIT"] = "true"

# 设置临时文件目录
TMPDIR = "/data/machine_learning/cpx/sam-3d-objects/tmp"

# 默认配置（可通过环境变量覆盖）
SAM3_MODEL_PATH = os.environ.get(
    "SAM3_MODEL_PATH",
    "/data/models/LLM-models-file/sam3",
)
SAM3D_CONFIG_FILE = os.environ.get(
    "SAM3D_CONFIG",
    "/data/models/LLM-models-file/sam-3d-objects/checkpoints/pipeline.yaml",
)
SAM3D_COMPILE = False

# 点击与已有点的判定半径（像素）：在此范围内视为“再次点击该点”并移除
POINT_CLICK_RADIUS = 20
# DRAW_POINT_RADIUS = 8
DRAW_POINT_RADIUS = 20

# 懒加载模型
_sam3_utils = None
_sam3d_inference = None

def get_sam3():
    """获取 SAM3 模型"""
    global _sam3_utils
    if _sam3_utils is None:
        _sam3_utils = Sam3Utils(SAM3_MODEL_PATH, device="cuda")
    return _sam3_utils

def get_sam3d():
    """获取 SAM-3D 模型"""
    global _sam3d_inference
    if _sam3d_inference is None:
        _sam3d_inference = Sam3DInference(SAM3D_CONFIG_FILE, SAM3_MODEL_PATH, compile=SAM3D_COMPILE)
    return _sam3d_inference

def preload_models() -> str:
    """
    在页面加载时预加载 SAM3 和 SAM-3D 模型，后续操作无需再加载

    由 Gradio demo.load 在启动/首次打开页面时调用
    """
    try:
        # 先返回消息，让页面可以立即渲染
        # 模型加载在后台进行
        logger.info("开始预加载模型...")
        get_sam3()
        logger.info("SAM3 模型预加载完成")
        get_sam3d()
        logger.info("SAM-3D 模型预加载完成")
        return "SAM3 与 SAM-3D 模型已就绪，可上传图片开始使用"
    except Exception as e:
        logger.error(f"模型预加载失败: {e}")
        return f"模型预加载失败: {e}，首次使用「Edit mask」或「Generate 3D」时将尝试加载"

def draw_points_on_image(
    pil_image: Image.Image,
    points: list[tuple[int, int, int]],
    radius: int = DRAW_POINT_RADIUS,
) -> Image.Image:
    """
    在原始图像上绘制正/负点：圆形标记 + 内部 "+"（Add）或 "-"（Remove）符号

    Add 为绿色系，Remove 为红色系，样式与界面按钮一致

    Args:
        pil_image: 原始图像
        points: 点列表 [(x,y,label), ...]
        radius: 点半径（像素）

    Returns:
        绘制点后的图像
    """
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    # 符号线半长（像素），略小于半径
    symbol_half = max(4, int(radius * 0.45))
    line_w = max(2, radius // 8)
    for x, y, label in points:
        is_add = (label == 1) # True 表示 Add，False 表示 Remove
        outline_color = (0, 200, 0) if is_add else (220, 0, 0)
        fill_color = (230, 255, 230) if is_add else (255, 230, 230)
        # 外圈
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=outline_color,
            fill=fill_color,
            width=2,
        )
        # 符号颜色与边框一致
        symbol_color = outline_color
        if is_add:
            # "+"：竖线 + 横线
            draw.line(
                [(x, y - symbol_half), (x, y + symbol_half)],
                fill=symbol_color,
                width=line_w,
            )
            draw.line(
                [(x - symbol_half, y), (x + symbol_half, y)],
                fill=symbol_color,
                width=line_w,
            )
        else:
            # "-"：横线
            draw.line(
                [(x - symbol_half, y), (x + symbol_half, y)],
                fill=symbol_color,
                width=line_w,
            )
    return img


def toggle_point(
    points: list[tuple[int, int, int]], 
    click_x: int, 
    click_y: int, 
    point_mode: int, 
    radius: int = POINT_CLICK_RADIUS
) -> list[tuple[int, int, int]]:
    """
    点击时添加或移除点：
    - 若点击位置在已有点的 radius 内则移除该点
    - 否则添加新点 (click_x, click_y, point_mode)

    Args:
        points: 当前点列表 `[(x,y,label), ...]`
        click_x: 点击 x 坐标
        click_y: 点击 y 坐标
        point_mode: 点模式 1=Add, 0=Remove
        radius: 点击判定半径（像素）
    """
    points = list(points)
    best_i = -1 # 最佳匹配点索引
    best_d2 = radius * radius + 1 # 最佳匹配距离平方
    # 遍历所有点，找到距离点击位置最近的点
    for i, (px, py, _) in enumerate(points):
        d2 = (click_x - px) ** 2 + (click_y - py) ** 2 # 计算距离平方
        if d2 < best_d2:
            best_d2 = d2 # 更新最佳匹配距离平方
            best_i = i # 更新最佳匹配点索引
    if best_i >= 0:
        points.pop(best_i) # 移除最佳匹配点
        return points # 返回移除最佳匹配点后的点列表
    # 若无最佳匹配点，则添加新点
    points.append((click_x, click_y, point_mode)) # 添加新点
    return points # 返回添加新点后的点列表


def image_from_upload(value) -> tuple[Image.Image | None, list, Image.Image | None]:
    """
    用户上传新图时：清空点和 mask，返回原图用于显示
    
    Args:
        value: 上传的图像

    Returns:
        原图、点列表、显示图像
    """
    if value is None:
        logger.warning("上传的图像为空")
        return None, [], None
    if isinstance(value, dict) and "image" in value:
        img = value["image"]
    else:
        img = value
    if img is None:
        logger.warning("上传的图像为空")
        return None, [], None
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    return img, [], img.convert("RGB")


def _base_pil(
    state_image: Optional[Image.Image | np.ndarray], 
    state_display: Optional[Image.Image | np.ndarray]
) -> Optional[Image.Image]:
    """
    获取用于绘制的底图 PIL（优先 overlay，否则原图）

    Args:
        state_image: 原始图像
        state_display: 当前显示的图像

    Returns:
        底图 PIL
    """
    base = state_display if state_display is not None else state_image
    if base is None:
        return None
    if isinstance(base, np.ndarray):
        return Image.fromarray(base.astype(np.uint8))
    return base.convert("RGB") if base.mode != "RGB" else base


def run_generate_mask(
    state_image: Optional[Image.Image | np.ndarray],
    state_points: list[tuple[int, int, int]],
    state_display: Optional[Image.Image | np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    根据当前点运行 SAM3 模型生成 mask，将分割结果直接叠加到原图

    数据流说明（供 SAM-3D 使用）：
    - 传入 SAM3 的仅为 (input_points, input_labels)，即 Add/Remove 点的坐标与标签。
    - SAM3 返回的 mask_np 是二值 mask (H,W)，不含任何点绘制信息。
    - 展示用图在 overlay 上再绘制点（draw_points_on_image），仅用于界面显示。
    - 存入 state_mask_np 并传给 SAM-3D 的始终是「纯 mask」，不包含点的叠加。

    Args:
        state_image: 原始图像
        state_points: 当前点 [(x,y,label), ...]，label 1=Add/0=Remove
        state_display: 当前显示的图像

    返回：显示图（带点）、显示图、纯 mask 数组（用于 SAM-3D 模型）、消息
    """
    if state_image is None or not state_points or len(state_points) == 0: # 无点时，显示为原始图像
        base = _base_pil(state_image, state_display)
        if base is None: # 如果原始图像为空，则返回空图像
            logger.warning("原始图像为空，无法绘制点")
            return None, None, None, "原始图像为空"
        disp_np = np.array(draw_points_on_image(base, state_points)) # 绘制点后的图像
        return disp_np, disp_np, None, "请添加至少一个点后点击图像进行分割" # 返回绘制点后的图像、绘制点后的图像、空 mask、消息

    if isinstance(state_image, np.ndarray):
        state_image = Image.fromarray(state_image.astype(np.uint8)) # 将 numpy 数组转换为 PIL Image

    points = state_points
    input_points = [[x, y] for x, y, _ in points]
    input_labels = [label for _, _, label in points]
    logger.debug(f"input_points: {input_points}, input_labels: {input_labels}")
    try:
        sam3 = get_sam3() # Sam3Utils 工具类
        # 使用 SAM3 模型的点提示进行分割
        overlay_pil, mask_image_np = sam3.sam3_with_points(
            state_image,
            input_points,
            input_labels
        )
        overlay_with_points = draw_points_on_image(overlay_pil, points)  # 在图像上绘制正/负点，仅用于界面显示，不写入 state_mask_np
        overlay_np = np.array(overlay_with_points) # 将 PIL Image 转换为 numpy 数组
        logger.debug(f"overlay_np: {overlay_np.shape}, mask_np: {mask_image_np.shape}")
        return overlay_np, overlay_np, mask_image_np, "Mask 已生成，可点击「生成 3D」"
    except Exception as e:
        logger.error(f"生成 Mask 失败: {e}")
        base = _base_pil(state_image, state_display)
        if base is None: # 如果原始图像为空，则返回空图像
            logger.error(f"原始图像为空")
            return None, None, None, "原始图像为空"
        fallback = np.array(draw_points_on_image(base, points)) if base else state_display # 绘制点后的图像
        return fallback, fallback, None, f"生成 Mask 失败: {e}"


def run_generate_3d(
    state_image: Optional[np.ndarray | Image.Image], 
    state_mask_np: Optional[np.ndarray], 
    seed_num: Optional[int] = 42
) -> tuple[Optional[str], Optional[str], str]:
    """
    根据原始图像和「纯 mask」运行 SAM-3D 模型，返回 3D 模型文件路径及消息。

    约定：state_mask_np 来自 run_generate_mask 的 mask_np，为 SAM3 输出的二值 mask (H,W)，
    不包含 Add/Remove 点的绘制；仅原图 + 该 mask 传入 SAM-3D。
    """
    if state_image is None:
        return None, None, "请先上传图片"
    
    if state_mask_np is None:
        return None, None, "请先在图上添加至少一个点以生成 Mask，再点击「Generate 3D」"
    
    # 将图像转换为 numpy 数组并转换为 uint8
    if isinstance(state_image, np.ndarray):
        image = state_image
    else:
        image = np.array(state_image)
    image = image.astype(np.uint8)
    logger.debug(f"image: {image.shape}, dtype: {image.dtype}, numel: {image.size}")
    # logger.debug(f"image: {image}")
    # 保存图像
    image_path = "./image.png"
    Image.fromarray(image).save(image_path)
    logger.debug(f"image saved to {image_path}")
    
    # 将 mask 转换为 uint8
    mask = state_mask_np  # (H, W) uint8 0/255
    mask = mask.astype(np.uint8)
    if mask.ndim == 3: # 如果 mask 是三维的，则取最后一个通道
        mask = mask[..., -1]
    logger.debug(f"mask: {mask.shape}, dtype: {mask.dtype}, numel: {mask.size}")
    # logger.debug(f"mask: {mask}")
    # 保存 mask
    mask_path = "./mask.png"
    Image.fromarray(mask).save(mask_path)
    logger.debug(f"mask saved to {mask_path}")

    # 空 mask 会导致下游 3D 管道在求 max 时报错，必须先拦截
    if not np.any(mask > 0):
        logger.warning("Mask 全为 0，无法生成 3D")
        return None, None, "Mask 为空（无前景像素），请重新在图上添加点或调整点的位置后再点「Edit mask」，再试「Generate 3D」。若原图过白/过暗也可能导致分割失败。"

    try:
        sam3d = get_sam3d() # 获取 SAM-3D 模型
        seed = int(seed_num) if seed_num is not None else 42
        output = sam3d(image, mask, seed=seed)
        logger.debug(f"keys of output: {output.keys()}")

        with tempfile.NamedTemporaryFile(dir=TMPDIR, suffix=".ply", delete=False) as f:
            ply_path = f.name
        logger.debug(f"PLY 文件路径: {ply_path}")
        with tempfile.NamedTemporaryFile(dir=TMPDIR, suffix=".glb", delete=False) as f:
            glb_path = f.name
        logger.debug(f"GLB 文件路径: {glb_path}")
        
        FileUtils.save_ply(ply_path, output) # 保存 PLY 文件
        FileUtils.save_glb(glb_path, output) # 保存 GLB 文件
        
        return ply_path, glb_path, "3D 模型已生成，可在此查看或导出 PLY/GLB"
    except Exception as e:
        logger.error(f"生成 3D 模型失败: {e}")
        return None, None, f"生成 3D 模型失败: {e}"


def export_ply(ply_path: str) -> Optional[str]:
    """导出 PLY 文件"""
    if ply_path and os.path.isfile(ply_path):
        return ply_path
    return None


def export_glb(glb_path: str) -> Optional[str]:
    """导出 GLB 文件"""
    if glb_path and os.path.isfile(glb_path):
        return glb_path
    return None


def on_upload(img: Image.Image):
    """
    上传新图事件：只更新原图与状态，显示区用同一张图

    Returns:
        原始图像、点列表、显示图像、显示图像、当前 mask、PLY 路径、GLB 路径、是否正在生成、清除按钮状态、生成按钮状态
    """
    original_image, points, display_image = image_from_upload(img)
    display_image_np = np.array(display_image) if display_image is not None else None
    has_points = len(points) > 0
    return (
        original_image, points, display_image, display_image, None, None, None, False,
        gr.update(interactive=has_points),  # btn_clear_points
        gr.update(interactive=has_points),  # btn_3d
    )


def on_clear_image():
    """
    用户点击删除/清除上传的图片时调用，重置所有相关状态。
    """
    return (
        None,   # state_image
        [],     # state_points
        None,   # state_display
        None,   # img_input（清空图像组件显示）
        None,   # state_mask_np
        None,   # state_ply_path
        None,   # state_glb_path
        None,   # model3d（清空 3D 预览）
        "已清除图片，状态已重置。可重新上传图片。",  # msg
        False,  # state_is_generating
        gr.update(interactive=False),  # btn_clear_points
        gr.update(interactive=False),  # btn_3d
        gr.update(interactive=False),  # btn_export_ply
        gr.update(interactive=False),  # btn_export_glb
    )


def on_clear_all_points_confirm(
    state_image: Optional[Image.Image | np.ndarray],
    state_ply_path: Optional[str],
    state_glb_path: Optional[str]
):
    """
    一键清除所有点：恢复显示为原图（无点、无 mask），并清空点列表与 mask 状态
    同时清除已生成的 3D 模型

    Args:
        state_image: 原始图像
        state_ply_path: PLY 文件路径
        state_glb_path: GLB 文件路径

    Returns:
        显示图像、点列表、显示图像、当前 mask、PLY 路径、GLB 路径、3D 模型、消息、清除按钮状态、生成按钮状态、导出PLY状态、导出GLB状态
    """
    base = _base_pil(state_image, None)
    if base is None:
        logger.warning("当前无图像，无需清除")
        return (
            None, [], None, None, None, None, None, "当前无图像，无需清除",
            gr.update(interactive=False),  # btn_clear_points
            gr.update(interactive=False),  # btn_3d
            gr.update(interactive=False),  # btn_export_ply
            gr.update(interactive=False),  # btn_export_glb
        )
    # 无点时直接显示原图
    out_np = np.array(base)
    logger.debug("成功清除所有点和 3D 模型")
    
    # 删除已生成的 3D 模型文件
    if state_ply_path and os.path.isfile(state_ply_path):
        try:
            os.remove(state_ply_path)
            logger.debug(f"已删除 PLY 文件: {state_ply_path}")
        except Exception as e:
            logger.warning(f"删除 PLY 文件失败: {e}")
    
    if state_glb_path and os.path.isfile(state_glb_path):
        try:
            os.remove(state_glb_path)
            logger.debug(f"已删除 GLB 文件: {state_glb_path}")
        except Exception as e:
            logger.warning(f"删除 GLB 文件失败: {e}")
    
    return (
        out_np, [], out_np, None, None, None, None, "已清除所有点和 3D 模型，可重新添加点",
        gr.update(interactive=False),  # btn_clear_points
        gr.update(interactive=False),  # btn_3d
        gr.update(interactive=False),  # btn_export_ply
        gr.update(interactive=False),  # btn_export_glb
    )

def clear_with_confirm_dialog(
    state_image, state_ply_path, state_glb_path, state_points, state_display, state_mask_np, confirmed=None
):
    """
    带确认对话框的清除函数
    - JavaScript返回的确认结果会作为最后一个参数 confirmed 传递（追加到inputs列表末尾）
    - 如果用户取消（confirmed 为 "false"），则返回原始状态，不执行清除操作
    - 如果用户确认（confirmed 为 "true"），则执行清除操作
    
    Args:
        confirmed: JavaScript返回的确认结果，"true" 表示确认，"false" 表示取消（作为最后一个参数）
    """
    try:
        # 添加日志以便调试
        logger.debug(f"clear_with_confirm_dialog 接收到参数:")
        logger.debug(f"  - confirmed: {confirmed}, 类型: {type(confirmed)}")
        logger.debug(f"  - state_image: {state_image is not None}, state_display: {state_display is not None}")
        
        # 检查确认结果（处理None、字符串等情况）
        confirmed_str = str(confirmed).strip().lower() if confirmed is not None else "false"
        logger.debug(f"  - confirmed_str: {confirmed_str}")
        
        # 检查确认结果
        if confirmed_str == "true":
            # 用户确认了，执行清除操作
            logger.debug("用户确认清除操作，执行清除")
            result = on_clear_all_points_confirm(state_image, state_ply_path, state_glb_path)
            # 重置确认状态为false，以便下次使用
            return result + (gr.update(value="false"),)
        else:
            # 用户取消了确认，返回原始状态，不执行任何清除操作
            logger.debug("用户取消了清除操作，不执行清除")
            
            # 确保返回的图像是numpy数组格式（img_input需要numpy数组）
            # 如果state_display是None，尝试使用state_image
            display_img = state_display
            if display_img is None and state_image is not None:
                # 将state_image转换为numpy数组
                if isinstance(state_image, np.ndarray):
                    display_img = state_image
                elif isinstance(state_image, Image.Image):
                    display_img = np.array(state_image)
                else:
                    display_img = None
            
            # 确保display_img是numpy数组或None
            if display_img is not None and not isinstance(display_img, np.ndarray):
                try:
                    display_img = np.array(display_img)
                except Exception as e:
                    logger.warning(f"无法将display_img转换为numpy数组: {e}")
                    display_img = None
            
            return (
                display_img, state_points, display_img, state_mask_np,
                state_ply_path, state_glb_path, None, "操作已取消",
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(value="false")  # 重置确认状态
            )
    except Exception as e:
        logger.error(f"clear_with_confirm_dialog 执行出错: {e}", exc_info=True)
        # 出错时返回原始状态，避免界面崩溃
        display_img = state_display
        if display_img is None and state_image is not None:
            if isinstance(state_image, np.ndarray):
                display_img = state_image
            elif isinstance(state_image, Image.Image):
                display_img = np.array(state_image)
        return (
            display_img, state_points, display_img, state_mask_np,
            state_ply_path, state_glb_path, None, f"操作出错: {e}",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(value="false")
        )

def on_select(
    img: Image.Image,
    pts: list[tuple[int, int, int]],
    disp: Image.Image,
    mode_choice: str,
    evt: gr.SelectData,
    is_generating: bool = False,
):
    """
    点击图像：添加/取消点，并立即基于当前所有点进行分割（结果叠加到本图）
    
    Args:
        img: 原始图像
        pts: 当前点列表 [(x,y,label), ...]
        disp: 当前显示的图像
        mode_choice: 点模式 "Add +" 或 "Remove -"
        evt: 选择事件
        is_generating: 是否正在生成 3D 模型
    
    Returns:
        显示图像、点列表、显示图像、当前 mask、消息、清除按钮状态、生成按钮状态
    """
    if img is None:
        return (
            disp, pts, disp, None, "",
            gr.update(interactive=False),  # btn_clear_points
            gr.update(interactive=False),  # btn_3d
        )
    
    # 如果正在生成 3D 模型，不允许添加点
    if is_generating:
        return (
            disp, pts, disp, None, "正在生成 3D 模型，请稍候...",
            gr.update(interactive=False),  # btn_clear_points
            gr.update(interactive=False),  # btn_3d
        )
    
    x, y = evt.index[0], evt.index[1]
    mode = 1 if mode_choice == "Add +" else 0
    logger.debug(f"x: {x}, y: {y}, mode: {mode}")
    
    new_points = toggle_point(pts, x, y, mode)
    logger.debug(f"new_points: {new_points}")

    has_points = len(new_points) > 0
    
    if not has_points: # 无点时，恢复显示为原图
        logger.debug(f"无点时，恢复显示为原图")
        base = _base_pil(img, None)
        if base is None:
            return (
                disp, new_points, disp, None, "已移除所有点",
                gr.update(interactive=False),  # btn_clear_points
                gr.update(interactive=False),  # btn_3d
            )
        out_np = np.array(draw_points_on_image(base, new_points))
        return (
            out_np, new_points, out_np, None, "已移除所有点，可重新添加点",
            gr.update(interactive=False),  # btn_clear_points
            gr.update(interactive=False),  # btn_3d
        )
    
    disp_np, _, mask_image_np, message = run_generate_mask(img, new_points, disp)
    return (
        disp_np, new_points, disp_np, mask_image_np, message,
        gr.update(interactive=True),  # btn_clear_points
        gr.update(interactive=True),  # btn_3d
    )

def start_generating():
    """开始生成 3D 模型时，更新状态"""
    return True  # 设置 state_is_generating 为 True

def finish_generating(
    img: Optional[np.ndarray | Image.Image], 
    mask: Optional[np.ndarray], 
    seed: Optional[int], 
    points: list[tuple[int, int, int]]
):
    """完成 3D 模型生成后，更新状态和文件路径"""
    model_path, ply_p, glb_p, m, export_ply_enabled, export_glb_enabled, _ = do_3d_with_state(img, mask, seed)
    has_points = len(points) > 0
    return (
        model_path, ply_p, glb_p, m,
        gr.update(interactive=export_ply_enabled),  # btn_export_ply
        gr.update(interactive=export_glb_enabled),  # btn_export_glb
        False,  # state_is_generating = False
        gr.update(interactive=has_points),  # btn_clear_points
        gr.update(interactive=has_points),  # btn_3d
    )

def do_3d(
    img: Optional[np.ndarray | Image.Image], 
    mask_np: Optional[np.ndarray], 
    seed: Optional[int] = 42
):
    """
    生成 3D 模型
    
    Args:
        img: 原始图像
        mask_np: 当前 mask
        seed: 随机种子

    Returns:
        3D 模型文件路径及消息
    """
    ply_p, glb_p, m = run_generate_3d(img, mask_np, seed)
    # Model3D 需要文件路径，优先展示 glb（兼容性更好）
    return (glb_p or ply_p), ply_p, glb_p, m

def do_3d_with_state(
    img: Optional[np.ndarray | Image.Image], 
    mask_np: Optional[np.ndarray], 
    seed: Optional[int] = 42
):
    """
    生成 3D 模型（带状态更新）
    
    Args:
        img: 原始图像
        mask_np: 当前 mask
        seed: 随机种子

    Returns:
        3D 模型文件路径、PLY路径、GLB路径、消息、导出PLY按钮状态、导出GLB按钮状态、是否正在生成状态
    """
    ply_p, glb_p, m = run_generate_3d(img, mask_np, seed)
    model_path = glb_p or ply_p
    
    # 更新导出按钮状态
    export_ply_enabled = ply_p is not None and os.path.isfile(ply_p)
    export_glb_enabled = glb_p is not None and os.path.isfile(glb_p)
    
    return model_path, ply_p, glb_p, m, export_ply_enabled, export_glb_enabled, False  # False 表示生成完成


def update_button_states(
    points: list[tuple[int, int, int]],
    is_generating: bool = False
) -> tuple[bool, bool, bool, bool, bool]:
    """
    根据点的数量和生成状态，更新按钮的 interactive 状态
    
    Args:
        points: 当前点列表
        is_generating: 是否正在生成 3D 模型
    
    Returns:
        (清除所有点可用, 生成3D模型可用, Export PLY可用, Export GLB可用, 图像可交互)
    """
    has_points = len(points) > 0
    return (
        has_points and not is_generating,  # 清除所有点：有点且不在生成中
        has_points and not is_generating,  # 生成3D模型：有点且不在生成中
        False,  # Export PLY：初始状态，后续会根据是否有文件来更新
        False,  # Export GLB：初始状态，后续会根据是否有文件来更新
        not is_generating  # 图像可交互：不在生成中
    )


def update_export_button_states(
    ply_path: Optional[str],
    glb_path: Optional[str]
) -> tuple[bool, bool]:
    """
    根据文件路径，更新导出按钮的 interactive 状态
    
    Args:
        ply_path: PLY 文件路径
        glb_path: GLB 文件路径
    
    Returns:
        (Export PLY可用, Export GLB可用)
    """
    return (
        ply_path is not None and os.path.isfile(ply_path),
        glb_path is not None and os.path.isfile(glb_path)
    )


def build_ui() -> gr.Blocks:
    """构建 UI"""
    with gr.Blocks(title="Image to 3D Generator") as demo:
        gr.Markdown("# SAM-3D Objects 模型生成")
        gr.Markdown(
            """**使用说明**：
            上传图片后，选择「Add +」或「Remove -」再点击图中添加/排除点（点为带 +/− 的圆形标记）；
            分割结果会直接叠加到原始图像上；
            再次点击某点可进行取消操作；
            也可点击「清除所有点」一键清空所有点；
            最后点击「Generate 3D」生成 3D 模型，并可导出 PLY/GLB 文件。"""
        )

        # 状态
        state_image = gr.State(None) # 原始图 PIL or np
        state_points = gr.State([]) # [(x,y,label), ...]
        state_display = gr.State(None) # 当前显示的图（带 mask/点）
        state_mask_np = gr.State(None) # 当前 mask (H,W) 用于 3D
        state_ply_path = gr.State(None) # PLY 文件路径
        state_glb_path = gr.State(None) # GLB 文件路径
        point_mode = gr.State(1) # 1=Add, 0=Remove
        state_is_generating = gr.State(False) # 是否正在生成 3D 模型
        # 使用隐藏的Textbox来存储确认结果（JavaScript设置，Python读取）
        hidden_confirm_clear = gr.Textbox(value="false", visible=False, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                **点模式**：
                - 选 `Add +` 点击添加正向点（绿色圆内 +）
                - 选 `Remove -` 点击添加负向点（红色圆内 −）
                - 再次点击某点可取消该点，或点击「清除所有点」一键清空
                """)
                img_input = gr.Image(
                    label="图像（原图 + 分割叠加 + 点）",
                    type="numpy",
                    interactive=True,
                    sources=["upload"],
                    buttons=["download", "fullscreen"]
                )
                with gr.Row():
                    point_mode_radio = gr.Radio(
                        choices=["Add +", "Remove -"],
                        value="Add +",
                        label="",
                        scale=1,
                    )
                btn_clear_points = gr.Button("清除所有点（会清除 3D 模型）", variant="secondary", interactive=False)
                btn_mask = gr.Button("重新生成 Mask", variant="secondary", visible="hidden")
                btn_3d = gr.Button("生成 3D 模型", variant="primary", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("Your 3D models will appear here")
                model3d = gr.Model3D(
                    label="3D 模型",
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    # camera_position=[0, 0, 1.2],
                    camera_position=(0, 90, 1.2)
                    # show_download_button=False,
                )
                with gr.Row():
                    btn_export_ply = gr.Button("Export PLY", interactive=False)
                    btn_export_glb = gr.Button("Export GLB", interactive=False)
                download_file = gr.File(label="导出文件", interactive=False)
                seed_num = gr.Number(value=42, label="随机种子", precision=0)
                msg = gr.Markdown("")

        # 页面加载时预加载 SAM3 与 SAM-3D，后续操作无需再加载
        demo.load(
            fn=preload_models,
            inputs=[],
            outputs=[msg],
        )
        # 上传图片事件
        img_input.upload(
            fn=on_upload,
            inputs=[img_input],
            outputs=[
                state_image, state_points, state_display, img_input, state_mask_np, 
                state_ply_path, state_glb_path, state_is_generating,
                btn_clear_points, btn_3d
            ],
        )
        # 用户清除上传的图片时重置所有状态
        img_input.clear(
            fn=on_clear_image,
            inputs=[],
            outputs=[
                state_image,
                state_points,
                state_display,
                img_input,
                state_mask_np,
                state_ply_path,
                state_glb_path,
                model3d,
                msg,
                state_is_generating,
                btn_clear_points,
                btn_3d,
                btn_export_ply,
                btn_export_glb,
            ],
        )
        # 点击图像事件
        img_input.select(
            fn=on_select,
            inputs=[state_image, state_points, state_display, point_mode_radio, state_is_generating],
            outputs=[img_input, state_points, state_display, state_mask_np, msg, btn_clear_points, btn_3d],
        )
        # 点击 "清除所有点" 按钮事件（带确认对话框）
        # 使用两步流程：1) JavaScript更新确认状态 2) Python函数读取确认状态并执行清除
        # btn_clear_points.click(
        #     fn=None,  # 第一步：只执行JavaScript，不执行Python函数
        #     inputs=[],
        #     outputs=[hidden_confirm_clear],
        #     js="""
        #     () => { 
        #         const confirmed = confirm('确认清除所有点和 3D 模型？此操作不可撤销。');
        #         return confirmed ? "true" : "false";
        #     }
        #     """,
        # ).then(
        #     fn=clear_with_confirm_dialog,  # 第二步：读取确认状态并执行清除
        #     inputs=[state_image, state_ply_path, state_glb_path, state_points, state_display, state_mask_np, hidden_confirm_clear],
        #     outputs=[
        #         img_input, state_points, state_display, state_mask_np,
        #         state_ply_path, state_glb_path, model3d, msg,
        #         btn_clear_points, btn_3d, btn_export_ply, btn_export_glb,
        #         hidden_confirm_clear
        #     ],
        # )
        btn_clear_points.click(
            fn=on_clear_all_points_confirm,
            inputs=[state_image, state_ply_path, state_glb_path],
            outputs=[img_input, state_points, state_display, state_mask_np,
                state_ply_path, state_glb_path, model3d, msg,
                btn_clear_points, btn_3d, btn_export_ply, btn_export_glb,
            ],
        )
        # 仅用当前点重新生成 mask（不添加新点）
        btn_mask.click(
            run_generate_mask,
            inputs=[state_image, state_points, state_display],
            outputs=[img_input, state_display, state_mask_np, msg],
        )
        # 生成 3D 模型事件（带状态管理）
        btn_3d.click(
            fn=start_generating,
            inputs=[],
            outputs=[state_is_generating],
        ).then(
            fn=lambda: (
                gr.update(interactive=False),  # btn_clear_points
                gr.update(interactive=False),  # btn_3d
                gr.update(interactive=False),  # btn_export_ply
                gr.update(interactive=False),  # btn_export_glb
                gr.update(interactive=False),  # img_input - 立即禁用图像交互，防止添加点
                "正在生成 3D 模型，请稍候...",  # msg - 显示提示消息
            ),
            inputs=[],
            outputs=[btn_clear_points, btn_3d, btn_export_ply, btn_export_glb, img_input, msg],
        ).then(
            fn=finish_generating,
            inputs=[state_image, state_mask_np, seed_num, state_points],
            outputs=[
                model3d, state_ply_path, state_glb_path, msg,
                btn_export_ply, btn_export_glb,
                state_is_generating,
                btn_clear_points, btn_3d
            ],
        ).then(
            fn=lambda: gr.update(interactive=True),  # 恢复图像交互
            inputs=[],
            outputs=[img_input],
        )
        # 导出 PLY 文件事件
        btn_export_ply.click(
            export_ply,
            inputs=[state_ply_path],
            outputs=[download_file],
        )
        # 导出 GLB 文件事件
        btn_export_glb.click(
            export_glb,
            inputs=[state_glb_path],
            outputs=[download_file],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        # server_port=10003,
        server_port=10004,
        theme=gr.themes.Soft(primary_hue="slate"),
        allowed_paths=["/data/machine_learning/cpx/sam-3d-objects/tmp"],
    )

