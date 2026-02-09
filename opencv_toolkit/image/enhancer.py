"""图像增强模块 — 提供锐化和去噪功能。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """图像增强处理器，支持锐化和去噪。

    Example:
        >>> enhancer = ImageEnhancer(kernel_size=9)
        >>> sharpened = enhancer.sharpen(image)
        >>> denoised = enhancer.denoise(sharpened)
    """

    def __init__(
        self,
        *,
        kernel_size: int = 9,
        denoise_strength: int = 10,
        denoise_template_window: int = 7,
        denoise_search_window: int = 21,
    ) -> None:
        """初始化图像增强器。

        Args:
            kernel_size: 锐化卷积核大小（正奇数）。
            denoise_strength: 去噪滤波强度。值越大去噪越强，但细节损失也越多。
            denoise_template_window: 去噪模板窗口大小（奇数）。
            denoise_search_window: 去噪搜索窗口大小（奇数）。
        """
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError(f"kernel_size 必须为 ≥3 的奇数，当前值: {kernel_size}")

        self.kernel_size = kernel_size
        self.denoise_strength = denoise_strength
        self.denoise_template_window = denoise_template_window
        self.denoise_search_window = denoise_search_window

    def _build_sharpen_kernel(self) -> NDArray[np.float32]:
        """构建锐化卷积核。

        使用 identity * 2 - box_filter 的方法：
        将图像中心像素权重设为 2，减去均匀模糊核，
        达到锐化效果。

        Returns:
            锐化卷积核矩阵。
        """
        k = self.kernel_size
        center = k // 2

        # 恒等核 × 2
        identity = np.zeros((k, k), dtype=np.float32)
        identity[center, center] = 2.0

        # 均值模糊核
        box_filter = np.ones((k, k), dtype=np.float32) / (k * k)

        return identity - box_filter

    def sharpen(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """对图像进行锐化处理。

        Args:
            image: 输入图像（灰度或彩色）。

        Returns:
            锐化后的图像。
        """
        kernel = self._build_sharpen_kernel()
        return cv2.filter2D(image, -1, kernel)

    def denoise(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """对灰度图像进行去噪处理。

        使用 Non-Local Means Denoising 算法。

        Args:
            image: 灰度输入图像。

        Returns:
            去噪后的图像。
        """
        return cv2.fastNlMeansDenoising(
            image,
            None,
            self.denoise_strength,
            self.denoise_template_window,
            self.denoise_search_window,
        )

    def denoise_color(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """对彩色图像进行去噪处理。

        Args:
            image: 彩色（BGR）输入图像。

        Returns:
            去噪后的彩色图像。
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            self.denoise_strength,
            self.denoise_strength,
            self.denoise_template_window,
            self.denoise_search_window,
        )

    def enhance_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        grayscale: bool = True,
    ) -> NDArray[np.uint8]:
        """增强图片文件（锐化 + 去噪）。

        Args:
            input_path: 输入图片路径。
            output_path: 输出图片路径。
            grayscale: 是否以灰度模式处理。

        Returns:
            增强后的图像数组。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 无法读取图片。
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if grayscale:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(path))

        if image is None:
            raise ValueError(f"无法读取图片: {path}")

        sharpened = self.sharpen(image)
        result = self.denoise(sharpened) if grayscale else self.denoise_color(sharpened)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result)
        logger.info("已增强并保存: %s → %s", path.name, out)

        return result
