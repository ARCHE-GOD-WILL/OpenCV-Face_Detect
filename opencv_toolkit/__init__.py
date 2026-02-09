"""
OpenCV Toolkit — 基于 OpenCV 4.x 的计算机视觉工具包。

提供人脸检测/比较、图像增强、线条检测、倾斜校正、印章检测等功能。
"""

__version__ = "2.0.0"
__author__ = "OpenCV-Face_Detect Contributors"

from opencv_toolkit.face.detector import FaceDetector
from opencv_toolkit.face.comparator import FaceComparator
from opencv_toolkit.image.color_remover import ColorRemover
from opencv_toolkit.image.enhancer import ImageEnhancer
from opencv_toolkit.image.line_detector import LineDetector
from opencv_toolkit.image.skew_corrector import SkewCorrector
from opencv_toolkit.image.stamp_detector import StampDetector

__all__ = [
    "FaceDetector",
    "FaceComparator",
    "ColorRemover",
    "ImageEnhancer",
    "LineDetector",
    "SkewCorrector",
    "StampDetector",
]
