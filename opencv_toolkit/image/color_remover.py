"""颜色移除模块 — 从图像中移除指定颜色区域。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HSVRange:
    """HSV 颜色范围。

    Attributes:
        lower: HSV 下限 [H, S, V]。
        upper: HSV 上限 [H, S, V]。
    """

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def to_numpy(self) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """转换为 numpy 数组。

        Returns:
            (lower_array, upper_array) 元组。
        """
        return np.array(self.lower, dtype=np.uint8), np.array(self.upper, dtype=np.uint8)


# 预定义常用颜色范围
RED_RANGES: list[HSVRange] = [
    HSVRange(lower=(0, 50, 50), upper=(10, 255, 255)),     # 低色相红色
    HSVRange(lower=(170, 50, 50), upper=(180, 255, 255)),  # 高色相红色
]

BLUE_RANGES: list[HSVRange] = [
    HSVRange(lower=(100, 50, 50), upper=(130, 255, 255)),
]

GREEN_RANGES: list[HSVRange] = [
    HSVRange(lower=(35, 50, 50), upper=(85, 255, 255)),
]


class ColorRemover:
    """从图像中移除指定颜色区域的处理器。

    支持通过 HSV 颜色范围精确指定要移除的颜色。
    匹配到的像素会被替换为指定的颜色（默认白色）。

    Example:
        >>> remover = ColorRemover(color_ranges=RED_RANGES)
        >>> result = remover.process(image)
    """

    def __init__(
        self,
        color_ranges: list[HSVRange] | None = None,
        *,
        replacement_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """初始化颜色移除器。

        Args:
            color_ranges: 要移除的 HSV 颜色范围列表。默认为红色。
            replacement_color: 替换颜色的 BGR 值，默认白色。
        """
        self.color_ranges = color_ranges or RED_RANGES
        self.replacement_color = replacement_color

    def _build_mask(self, hsv_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """构建颜色掩码。

        Args:
            hsv_image: HSV 格式的图像。

        Returns:
            合并后的二值掩码。
        """
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for color_range in self.color_ranges:
            lower, upper = color_range.to_numpy()
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))
        return mask

    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """处理图像，移除指定颜色区域。

        Args:
            image: BGR 格式的输入图像。

        Returns:
            处理后的图像，匹配颜色的区域被替换。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv)

        output = image.copy()
        output[mask != 0] = self.replacement_color
        return output

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> NDArray[np.uint8]:
        """处理图片文件。

        Args:
            input_path: 输入图片路径。
            output_path: 输出图片路径。为 None 时覆盖原文件。

        Returns:
            处理后的图像数组。

        Raises:
            FileNotFoundError: 输入文件不存在。
            ValueError: 无法读取图片。
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片: {path}")

        result = self.process(image)

        save_path = Path(output_path) if output_path else path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), result)
        logger.info("已保存: %s", save_path)

        return result
