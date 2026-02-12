# SAM 3 分割与 SAM-3D Objects 前端

基于 Gradio 的 Web 界面，实现：

1. **SAM 3 图像分割**：上传图片后，通过「Add +」/「Remove -」切换模式，在图上点击添加正向点或负向点，点击「Edit mask」生成并实时展示 Mask。
2. **SAM-3D Objects 3D 生成**：在得到 Mask 后点击「Generate 3D」生成 3D 模型，右侧展示并可交互查看。
3. **导出**：支持将当前 3D 模型导出为 **PLY** 或 **GLB** 格式并下载。

## 环境与依赖

- 与 `src/test_sam3d_inference.py` 相同：需安装项目依赖、CUDA、SAM3 与 SAM-3D 模型。
- 从**项目根目录**运行时，需把 `src` 加入 Python 路径；或先 `cd src` 再以模块方式运行 frontend。

## 运行方式

在项目根目录下（`sam-3d-objects/`）：

```bash
# 方式一：直接运行（会自动把 repo 根和 src 加入 path）
export CUDA_VISIBLE_DEVICES=2

python frontend/app.py
```

或指定配置与模型路径（与 `test_sam3d_inference` 一致）：

```bash
export SAM3D_CONFIG=/path/to/pipeline.yaml
export SAM3_MODEL_PATH=/path/to/sam3
python frontend/app.py
```

默认服务地址：`http://0.0.0.0:10003`。

## 使用流程

1. 上传一张图片。
2. 点击「Add +」，再在图中目标物体上点击，添加正向点；需要排除区域时点击「Remove -」再点击该区域。
3. 点击「Edit mask」运行 SAM 3，得到 Mask 并叠加显示在图上。
4. 点击「Generate 3D」运行 SAM-3D Objects，右侧会显示 3D 模型。
5. 点击「Export PLY」或「Export GLB」下载对应格式的 3D 文件。


## 相关变量

### 输入相关变量

`state_image`：当前图像

`state_points`：当前点列表，`[(x, y, label), ...]`

`state_display`：当前显示的图像

`image_input`：

`state_mask_np`：生成的 Mask（Numpy Array 格式）

`state_ply_path`：3D 模型 PLY 文件的保存路径

`state_glb_path`：3D 模型 GLB 文件的保存路径

`model3d`：3D 模型预览

### 状态相关变量

> 通过设置 Gradio Button 的 `interactive` 属性

> 通过 `gradio.update(interactive=True)` 或者 `gradio.update(interactive=False)` 更新状态值

`state_is_generating`：是否正在生成 ==bool 类型==

`btn_clear_points`："清除所有点" 按钮的状态（是否处于 "可用" 状态） ==bool 类型==

`btn_3d`："生成 3D 模型" 按钮的状态（是否处于 "可用" 状态） ==bool 类型==

`btn_export_ply`：（是否处于 "可用" 状态） ==bool 类型==

`btn_export_glb`：（是否处于 "可用" 状态） ==bool 类型==







