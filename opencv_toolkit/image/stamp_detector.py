"""印章检测模块 — 检测图像中的矩形和圆形印章区域。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DetectedStamp:
    """检测到的印章信息。

    Attributes:
        shape: 形状类型（'rectangle' 或 'circle'）。
        center: 中心坐标 (x, y)。
        bounding_box: 边界框 (x, y, w, h)。
        radius: 圆形印章的半径（仅圆形有值）。
        contour: 轮廓点（仅矩形有值）。
    """

    shape: str
    center: tuple[int, int]
    bounding_box: tuple[int, int, int, int]
    radius: int | None = None
    contour: NDArray[np.int32] | None = None


@dataclass
class StampDetectionResult:
    """印章检测结果。

    Attributes:
        stamps: 检测到的印章列表。
        annotated_image: 标注后的图像。
    """

    stamps: list[DetectedStamp] = field(default_factory=list)
    annotated_image: NDArray[np.uint8] | None = None


class StampDetector:
    """印章检测器 — 检测文档图像中的矩形和圆形印章。

    工作流程：锐化 → 灰度 → 高斯模糊 → 自适应阈值 →
    形态学操作 → 轮廓检测（矩形） / 霍夫圆检测（圆形）。

    Example:
        >>> detector = StampDetector()
        >>> result = detector.detect(image)
        >>> for stamp in result.stamps:
        ...     print(f"{stamp.shape} at {stamp.center}")
    """

    def __init__(
        self,
        *,
        sharpen_kernel_size: int = 100,
        gaussian_kernel: tuple[int, int] = (3, 3),
        adaptive_block_size: int = 11,
        adaptive_c: float = 3.5,
        morph_kernel_size: tuple[int, int] = (2, 2),
        morph_iterations: int = 1,
        min_vertices_for_rect: int = 4,
        contour_approx_factor: float = 0.02,
        hough_dp: float = 1.2,
        hough_min_dist: int = 100,
        rect_color: tuple[int, int, int] = (0, 255, 0),
        circle_color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """初始化印章检测器。

        Args:
            sharpen_kernel_size: 锐化核大小。
            gaussian_kernel: 高斯模糊核大小。
            adaptive_block_size: 自适应阈值的块大小。
            adaptive_c: 自适应阈值的常数 C。
            morph_kernel_size: 形态学操作核大小。
            morph_iterations: 形态学操作迭代次数。
            min_vertices_for_rect: 矩形检测的最小顶点数。
            contour_approx_factor: 轮廓近似精度因子。
            hough_dp: 霍夫圆检测的累加器分辨率反比。
            hough_min_dist: 检测到的圆心之间的最小距离。
            rect_color: 矩形标注颜色（BGR）。
            circle_color: 圆形标注颜色（BGR）。
        """
        self.sharpen_kernel_size = sharpen_kernel_size
        self.gaussian_kernel = gaussian_kernel
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.min_vertices_for_rect = min_vertices_for_rect
        self.contour_approx_factor = contour_approx_factor
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.rect_color = rect_color
        self.circle_color = circle_color

    def _sharpen(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """对图像进行锐化。

        Args:
            image: 输入图像。

        Returns:
            锐化后的图像。
        """
        k = self.sharpen_kernel_size
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, k // 2] = 2.0
        box = np.ones((k, k), dtype=np.float32) / (k * k)
        kernel = kernel - box
        return cv2.filter2D(image, -1, kernel)

    def _preprocess_for_circles(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """为圆形检测预处理图像。

        Args:
            image: BGR 输入图像。

        Returns:
            预处理后的灰度图像。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            int(self.adaptive_c),
        )
        kernel = np.ones(self.morph_kernel_size, dtype=np.uint8)
        gray = cv2.erode(gray, kernel, iterations=self.morph_iterations)
        gray = cv2.dilate(gray, kernel, iterations=self.morph_iterations)
        return gray

    def detect_rectangles(self, image: NDArray[np.uint8]) -> list[DetectedStamp]:
        """检测图像中的矩形印章。

        Args:
            image: BGR 格式的输入图像。

        Returns:
            检测到的矩形印章列表。
        """
        sharpened = self._sharpen(image)
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, 0)
        edges = cv2.Canny(blurred, 100, 300)

        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, struct)

        # OpenCV 4.x: findContours 返回 (contours, hierarchy)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stamps: list[DetectedStamp] = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.contour_approx_factor * peri, True)

            if len(approx) >= self.min_vertices_for_rect:
                x, y, w, h = cv2.boundingRect(approx)
                stamps.append(
                    DetectedStamp(
                        shape="rectangle",
                        center=(x + w // 2, y + h // 2),
                        bounding_box=(x, y, w, h),
                        contour=approx,
                    )
                )

        logger.info("检测到 %d 个矩形印章", len(stamps))
        return stamps

    def detect_circles(self, image: NDArray[np.uint8]) -> list[DetectedStamp]:
        """检测图像中的圆形印章。

        Args:
            image: BGR 格式的输入图像。

        Returns:
            检测到的圆形印章列表。
        """
        preprocessed = self._preprocess_for_circles(image)
        circles = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, self.hough_dp, self.hough_min_dist
        )

        stamps: list[DetectedStamp] = []
        if circles is not None:
            rounded = np.round(circles[0, :]).astype(int)
            for x, y, r in rounded:
                stamps.append(
                    DetectedStamp(
                        shape="circle",
                        center=(int(x), int(y)),
                        bounding_box=(int(x - r), int(y - r), int(2 * r), int(2 * r)),
                        radius=int(r),
                    )
                )

        logger.info("检测到 %d 个圆形印章", len(stamps))
        return stamps

    def detect(self, image: NDArray[np.uint8]) -> StampDetectionResult:
        """检测图像中的所有印章（矩形 + 圆形）。

        Args:
            image: BGR 格式的输入图像。

        Returns:
            StampDetectionResult 包含所有检测到的印章和标注图像。
        """
        annotated = image.copy()
        all_stamps: list[DetectedStamp] = []

        # 检测矩形
        rect_stamps = self.detect_rectangles(image)
        for stamp in rect_stamps:
            if stamp.contour is not None:
                cv2.drawContours(annotated, [stamp.contour], -1, self.rect_color, 4)
        all_stamps.extend(rect_stamps)

        # 检测圆形
        circle_stamps = self.detect_circles(image)
        for stamp in circle_stamps:
            if stamp.radius is not None:
                cv2.circle(annotated, stamp.center, stamp.radius, self.circle_color, 4)
        all_stamps.extend(circle_stamps)

        logger.info("共检测到 %d 个印章", len(all_stamps))
        return StampDetectionResult(stamps=all_stamps, annotated_image=annotated)

    def detect_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> StampDetectionResult:
        """检测图片文件中的印章。

        Args:
            input_path: 输入图片路径。
            output_path: 标注后图片的保存路径（可选）。

        Returns:
            StampDetectionResult 结果。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 无法读取图片。
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片: {path}")

        result = self.detect(image)

        if output_path and result.annotated_image is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), result.annotated_image)
            logger.info("已保存标注图像: %s", out)

        return result
