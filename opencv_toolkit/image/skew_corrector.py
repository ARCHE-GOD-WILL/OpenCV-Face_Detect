"""倾斜校正模块 — 自动检测并校正图像的旋转倾斜。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkewResult:
    """倾斜校正结果。

    Attributes:
        corrected_image: 校正后的图像。
        angle: 检测到的倾斜角度（度）。
    """

    corrected_image: NDArray[np.uint8]
    angle: float


class SkewCorrector:
    """图像倾斜自动校正器。

    使用最小面积矩形方法检测文档图像的旋转角度，
    然后通过仿射变换进行校正。

    Example:
        >>> corrector = SkewCorrector()
        >>> result = corrector.correct(image)
        >>> print(f"校正角度: {result.angle:.2f}°")
    """

    def __init__(
        self,
        *,
        threshold_method: int = cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        interpolation: int = cv2.INTER_CUBIC,
        border_mode: int = cv2.BORDER_REPLICATE,
    ) -> None:
        """初始化倾斜校正器。

        Args:
            threshold_method: 二值化方法标志。
            interpolation: 仿射变换插值方法。
            border_mode: 仿射变换边界处理方式。
        """
        self.threshold_method = threshold_method
        self.interpolation = interpolation
        self.border_mode = border_mode

    def detect_angle(self, image: NDArray[np.uint8]) -> float:
        """检测图像的倾斜角度。

        将图像转为灰度 → 反色 → 二值化 → 计算最小面积矩形。

        Args:
            image: BGR 格式的输入图像。

        Returns:
            倾斜角度（度），正值为逆时针倾斜。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, self.threshold_method)[1]

        coords = np.column_stack(np.where(thresh > 0))

        if coords.size == 0:
            logger.warning("未检测到前景像素，无法计算倾斜角度")
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]

        # cv2.minAreaRect 返回值范围为 [-90, 0)
        # 当矩形接近水平时角度趋近 0，接近垂直时趋近 -90
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        return float(angle)

    def correct(self, image: NDArray[np.uint8]) -> SkewResult:
        """检测并校正图像倾斜。

        Args:
            image: BGR 格式的输入图像。

        Returns:
            SkewResult 包含校正后图像和检测到的角度。
        """
        angle = self.detect_angle(image)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=self.interpolation,
            borderMode=self.border_mode,
        )

        logger.info("倾斜校正完成: 角度 = %.3f°", angle)
        return SkewResult(corrected_image=corrected, angle=angle)

    def correct_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        show: bool = False,
    ) -> SkewResult:
        """校正图片文件的倾斜。

        Args:
            input_path: 输入图片路径。
            output_path: 输出图片路径。
            show: 是否弹窗显示校正前后对比。

        Returns:
            SkewResult 结果。

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

        result = self.correct(image)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result.corrected_image)
        logger.info("已保存校正后图像: %s (角度: %.3f°)", out, result.angle)

        if show:
            cv2.imshow("Original", image)
            cv2.imshow("Corrected", result.corrected_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result
