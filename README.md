# OpenCV Toolkit

基于 **Python 3.12+** 和 **OpenCV 4.x** 的计算机视觉工具包，提供人脸检测/比较、图像增强、线条检测、倾斜校正、印章检测等功能。

## 功能特性

| 模块 | 功能 | 类 |
|------|------|------|
| **人脸检测** | 通过摄像头实时识别人脸 | `FaceDetector` |
| **人脸比较** | 比较两张照片是否为同一人 | `FaceComparator` |
| **颜色移除** | 从图像中移除指定颜色区域 | `ColorRemover` |
| **图像增强** | 锐化 + 去噪处理 | `ImageEnhancer` |
| **线条检测** | 检测水平/垂直线并切割图像 | `LineDetector` |
| **倾斜校正** | 自动检测并校正文档倾斜 | `SkewCorrector` |
| **印章检测** | 检测矩形和圆形印章 | `StampDetector` |

## 演示视频

| 演示 | 文件路径 |
|------|----------|
| **Veo 生成演示** | [`demo_assets/veo_demo_final.mp4`](demo_assets/veo_demo_final.mp4) |
| **实拍演示** | [`demo_assets/real_demo_0.mp4`](demo_assets/real_demo_0.mp4) |

## 环境要求

- Python >= 3.12
- OpenCV >= 4.8
- 人脸检测功能需要 `dlib` 和 `face_recognition`

## 安装

```bash
# 克隆仓库
git clone https://github.com/user/OpenCV-Face_Detect.git
cd OpenCV-Face_Detect

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 或以可编辑模式安装为包
pip install -e .
```

> **注意：** `dlib` 的安装可能需要 CMake。macOS 用户可通过 `brew install cmake` 安装。

## 快速上手

### 人脸检测（摄像头实时识别）

```python
from opencv_toolkit.face import FaceDetector

detector = FaceDetector(scale_factor=0.25, frame_skip=2)
detector.register_face("张三", "reference_photo.jpg")
detector.run_webcam(camera_index=0)
```

### 人脸比较

```python
from opencv_toolkit.face import FaceComparator

comparator = FaceComparator(
    predictor_path="shape_predictor_68_face_landmarks.dat",
    face_rec_model_path="dlib_face_recognition_resnet_model_v1.dat",
    match_threshold=0.45,
)
result = comparator.compare("face1.jpg", "face2.jpg")
print(f"匹配: {result.is_match}, 距离: {result.distance:.4f}")
```

### 颜色移除

```python
from opencv_toolkit.image import ColorRemover
from opencv_toolkit.image.color_remover import RED_RANGES

remover = ColorRemover(color_ranges=RED_RANGES)
remover.process_file("input.jpg", "output_no_red.jpg")
```

### 图像增强（锐化 + 去噪）

```python
from opencv_toolkit.image import ImageEnhancer

enhancer = ImageEnhancer(kernel_size=9)
enhancer.enhance_file("noisy.jpg", "enhanced.jpg", grayscale=True)
```

### 线条检测与切割

```python
from opencv_toolkit.image import LineDetector

detector = LineDetector(hough_threshold=150, min_line_length=120)
result = detector.process_file("document.jpg", "output/", direction="both")
print(f"水平线: {len(result.horizontal_positions)}, 垂直线: {len(result.vertical_positions)}")
```

### 倾斜校正

```python
from opencv_toolkit.image import SkewCorrector

corrector = SkewCorrector()
result = corrector.correct_file("skewed.jpg", "corrected.jpg")
print(f"校正角度: {result.angle:.2f}°")
```

### 印章检测

```python
from opencv_toolkit.image import StampDetector

detector = StampDetector()
result = detector.detect_file("document.jpg", "annotated.jpg")
for stamp in result.stamps:
    print(f"{stamp.shape} at {stamp.center}")
```

## CLI 工具

项目附带命令行示例脚本：

```bash
# 实时人脸检测
python -m examples.face_detection_webcam --image ref.jpg --name "张三"

# 人脸比较
python -m examples.face_comparison \
    --predictor shape_predictor_68_face_landmarks.dat \
    --model dlib_face_recognition_resnet_model_v1.dat \
    --image1 face1.jpg --image2 face2.jpg

# 图像处理
python -m examples.image_processing color-remove input.jpg output.jpg
python -m examples.image_processing enhance input.jpg output.jpg
python -m examples.image_processing line-detect input.jpg output_dir/
python -m examples.image_processing skew-correct input.jpg output.jpg
python -m examples.image_processing stamp-detect input.jpg output.jpg
```

所有命令都支持 `-v` 参数输出详细日志。

## 项目结构

```
OpenCV-Face_Detect/
├── opencv_toolkit/            # 核心包
│   ├── __init__.py
│   ├── face/                  # 人脸处理模块
│   │   ├── __init__.py
│   │   ├── detector.py        # FaceDetector - 实时人脸检测
│   │   └── comparator.py      # FaceComparator - 人脸比较
│   └── image/                 # 图像处理模块
│       ├── __init__.py
│       ├── color_remover.py   # ColorRemover - 颜色移除
│       ├── enhancer.py        # ImageEnhancer - 锐化与去噪
│       ├── line_detector.py   # LineDetector - 线条检测与切割
│       ├── skew_corrector.py  # SkewCorrector - 倾斜校正
│       └── stamp_detector.py  # StampDetector - 印章检测
├── examples/                  # CLI 示例脚本
│   ├── face_detection_webcam.py
│   ├── face_comparison.py
│   └── image_processing.py
├── data/                      # 模型数据
│   └── haarcascade_frontalface_default.xml
├── demo_assets/               # 演示视频资源
├── requirements.txt           # Python 依赖
├── pyproject.toml             # 项目配置
├── .gitignore
└── README.md
```

## v2.0 重构亮点

本次重构将9年前的脚本式代码现代化为专业的 Python 包：

- **OOP 架构**：所有功能封装为独立的类，支持参数化配置
- **类型标注**：全量 type hints，兼容 `mypy --strict`
- **文档字符串**：所有类和方法都有完整的 Google 风格 docstring
- **跨平台兼容**：使用 `pathlib.Path` 替代硬编码 Windows 路径
- **OpenCV 4.x**：修复 `cv2.findContours` 返回值变更等兼容性问题
- **Python 3.12+**：使用 `from __future__ import annotations`、`dataclass`、`|` 类型联合等现代语法
- **Pillow 10+**：`Image.ANTIALIAS` → `Image.LANCZOS`
- **错误处理**：完善的异常抛出与日志记录
- **可配置性**：所有硬编码参数均可通过构造函数覆盖

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码检查
ruff check opencv_toolkit/
mypy opencv_toolkit/

# 运行测试
pytest
```

## 许可证

MIT License

## Web UI (Streamlit)

A web interface is available for testing face detection and landmark visualization.

### Usage

```bash
streamlit run app.py
```

### Features

- **Upload Image**: Supports JPG/PNG.
- **Model Selection**: Switch between HOG (CPU) and CNN (GPU/Accurate) models.
- **Visual Options**: Toggle bounding boxes and facial landmarks.
- **Customization**: Adjust colors and line thickness.
