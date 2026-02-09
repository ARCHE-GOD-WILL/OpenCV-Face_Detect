"""图像处理模块。"""

from opencv_toolkit.image.color_remover import ColorRemover
from opencv_toolkit.image.enhancer import ImageEnhancer
from opencv_toolkit.image.line_detector import LineDetector
from opencv_toolkit.image.skew_corrector import SkewCorrector
from opencv_toolkit.image.stamp_detector import StampDetector

__all__ = [
    "ColorRemover",
    "ImageEnhancer",
    "LineDetector",
    "SkewCorrector",
    "StampDetector",
]
