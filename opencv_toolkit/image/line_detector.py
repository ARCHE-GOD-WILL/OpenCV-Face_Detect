"""线条检测模块 — 检测图像中的水平/垂直线条并按线条切割图像。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DetectedLine:
    """检测到的线条。

    Attributes:
        x1: 起点 X 坐标。
        y1: 起点 Y 坐标。
        x2: 终点 X 坐标。
        y2: 终点 Y 坐标。
        orientation: 线条方向（'horizontal' 或 'vertical'）。
    """

    x1: int
    y1: int
    x2: int
    y2: int
    orientation: Literal["horizontal", "vertical", "diagonal"] = "diagonal"


@dataclass
class SliceResult:
    """图像切片结果。

    Attributes:
        slices: 切割后的图像片段列表。
        annotated_image: 标注了切割线的图像。
        horizontal_positions: 水平切割的 Y 坐标列表。
        vertical_positions: 垂直切割的 X 坐标列表。
    """

    slices: list[NDArray[np.uint8]] = field(default_factory=list)
    annotated_image: NDArray[np.uint8] | None = None
    horizontal_positions: list[int] = field(default_factory=list)
    vertical_positions: list[int] = field(default_factory=list)


class LineDetector:
    """基于 Hough 变换的线条检测器。

    检测图像中的水平/垂直线条，并支持按检测到的线条切割图像。

    Attributes:
        canny_low: Canny 边缘检测低阈值。
        canny_high: Canny 边缘检测高阈值。
        hough_threshold: Hough 变换累加器阈值。
        min_line_length: 最短线段长度（像素）。
        max_line_gap: 允许的最大线段间隙（像素）。
        merge_distance: 合并相近线条的最小间距（像素）。
    """

    def __init__(
        self,
        *,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 150,
        min_line_length: int = 120,
        max_line_gap: int = 5,
        merge_distance: int = 40,
        horizontal_angle_tolerance: int = 10,
        vertical_angle_tolerance: int = 10,
    ) -> None:
        """初始化线条检测器。

        Args:
            canny_low: Canny 低阈值。
            canny_high: Canny 高阈值。
            hough_threshold: Hough 变换阈值。
            min_line_length: 最短线段长度。
            max_line_gap: 最大线段间隙。
            merge_distance: 相近线条合并距离。
            horizontal_angle_tolerance: 水平线的倾斜容差（像素）。
            vertical_angle_tolerance: 垂直线的倾斜容差（像素）。
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.merge_distance = merge_distance
        self.horizontal_angle_tolerance = horizontal_angle_tolerance
        self.vertical_angle_tolerance = vertical_angle_tolerance

    def detect_lines(self, image: NDArray[np.uint8]) -> list[DetectedLine]:
        """检测图像中的线条。

        Args:
            image: BGR 格式的输入图像。

        Returns:
            检测到的线条列表。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high, apertureSize=3)

        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if raw_lines is None:
            logger.warning("未检测到线条")
            return []

        lines: list[DetectedLine] = []
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            dy, dx = abs(y1 - y2), abs(x1 - x2)

            if dy < self.horizontal_angle_tolerance and dx > self.horizontal_angle_tolerance:
                orientation: Literal["horizontal", "vertical", "diagonal"] = "horizontal"
            elif dy > self.vertical_angle_tolerance and dx < self.vertical_angle_tolerance:
                orientation = "vertical"
            else:
                orientation = "diagonal"

            lines.append(DetectedLine(x1=x1, y1=y1, x2=x2, y2=y2, orientation=orientation))

        logger.info("检测到 %d 条线段", len(lines))
        return lines

    @staticmethod
    def _merge_positions(positions: list[int], min_distance: int) -> list[int]:
        """合并相近的位置值。

        Args:
            positions: 排序后的位置列表。
            min_distance: 最小间距。

        Returns:
            合并后的位置列表。
        """
        if not positions:
            return []

        sorted_pos = sorted(positions)
        merged = [sorted_pos[0]]

        for pos in sorted_pos[1:]:
            if abs(pos - merged[-1]) > min_distance:
                merged.append(pos)

        return merged

    def detect_and_slice(
        self,
        image: NDArray[np.uint8],
        *,
        direction: Literal["horizontal", "vertical", "both"] = "horizontal",
        line_color: tuple[int, int, int] = (0, 0, 255),
    ) -> SliceResult:
        """检测线条并按线条切割图像。

        Args:
            image: BGR 格式的输入图像。
            direction: 检测方向。
            line_color: 标注线颜色（BGR）。

        Returns:
            SliceResult 包含切片、标注图和切割位置。
        """
        height, width = image.shape[:2]
        lines = self.detect_lines(image)
        annotated = image.copy()

        h_positions: list[int] = []
        v_positions: list[int] = []

        for line in lines:
            if direction in ("horizontal", "both") and line.orientation == "horizontal":
                h_positions.append(line.y1)
            if direction in ("vertical", "both") and line.orientation == "vertical":
                v_positions.append(line.x1)

        h_merged = self._merge_positions(h_positions, self.merge_distance)
        v_merged = self._merge_positions(v_positions, self.merge_distance)

        # 绘制标注线
        for y in h_merged:
            cv2.line(annotated, (0, y), (width, y), line_color, 2)
        for x in v_merged:
            cv2.line(annotated, (x, 0), (x, height), line_color, 2)

        # 切割图像
        slices: list[NDArray[np.uint8]] = []

        if direction == "both" and h_merged and v_merged:
            y_bounds = [0, *h_merged, height]
            x_bounds = [0, *v_merged, width]
            for i in range(len(y_bounds) - 1):
                for j in range(len(x_bounds) - 1):
                    crop = image[y_bounds[i] : y_bounds[i + 1], x_bounds[j] : x_bounds[j + 1]]
                    if crop.size > 0:
                        slices.append(crop)
        elif h_merged:
            y_bounds = [0, *h_merged, height]
            for i in range(len(y_bounds) - 1):
                crop = image[y_bounds[i] : y_bounds[i + 1], 0:width]
                if crop.size > 0:
                    slices.append(crop)
        elif v_merged:
            x_bounds = [0, *v_merged, width]
            for i in range(len(x_bounds) - 1):
                crop = image[0:height, x_bounds[i] : x_bounds[i + 1]]
                if crop.size > 0:
                    slices.append(crop)

        logger.info(
            "切割完成: %d 条水平线, %d 条垂直线, %d 个片段",
            len(h_merged),
            len(v_merged),
            len(slices),
        )

        return SliceResult(
            slices=slices,
            annotated_image=annotated,
            horizontal_positions=h_merged,
            vertical_positions=v_merged,
        )

    def process_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        direction: Literal["horizontal", "vertical", "both"] = "horizontal",
    ) -> SliceResult:
        """处理图片文件，检测线条并保存切割结果。

        Args:
            input_path: 输入图片路径。
            output_dir: 输出目录路径。
            direction: 检测方向。

        Returns:
            SliceResult 结果。

        Raises:
            FileNotFoundError: 输入文件不存在。
            ValueError: 无法读取图片。
        """
        path = Path(input_path)
        out_dir = Path(output_dir)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片: {path}")

        result = self.detect_and_slice(image, direction=direction)

        out_dir.mkdir(parents=True, exist_ok=True)

        # 保存标注图
        if result.annotated_image is not None:
            annotated_path = out_dir / f"{path.stem}_annotated{path.suffix}"
            cv2.imwrite(str(annotated_path), result.annotated_image)

        # 保存切片
        for idx, slice_img in enumerate(result.slices):
            slice_path = out_dir / f"{path.stem}_slice_{idx:03d}{path.suffix}"
            cv2.imwrite(str(slice_path), slice_img)

        logger.info("已保存 %d 个切片到 %s", len(result.slices), out_dir)
        return result
