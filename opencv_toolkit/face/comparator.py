"""人脸比较模块 — 使用 dlib 比较两张人脸图片的相似度。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import dlib
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.spatial import distance

logger = logging.getLogger(__name__)

# dlib 人脸特征点数量
_NUM_LANDMARKS = 68


@dataclass(frozen=True)
class ComparisonResult:
    """人脸比较结果。

    Attributes:
        distance: 两张人脸的欧氏距离。
        is_match: 是否为同一人。
        threshold: 判断阈值。
    """

    distance: float
    is_match: bool
    threshold: float

    def __str__(self) -> str:
        status = "Match" if self.is_match else "Not Match"
        return f"{status} (distance={self.distance:.4f}, threshold={self.threshold})"


class FaceComparator:
    """基于 dlib 的人脸比较器。

    使用 dlib 的 68 点特征检测和 ResNet 人脸识别模型，
    计算两张人脸图片之间的欧氏距离来判断是否为同一人。

    Attributes:
        predictor_path: 68 点特征检测器模型路径。
        face_rec_model_path: 人脸识别模型路径。
        match_threshold: 匹配阈值，距离小于该值判定为匹配。
        resize_width: 预处理时统一缩放到的宽度。
    """

    def __init__(
        self,
        predictor_path: str | Path,
        face_rec_model_path: str | Path,
        *,
        match_threshold: float = 0.45,
        resize_width: int = 800,
    ) -> None:
        """初始化人脸比较器。

        Args:
            predictor_path: dlib shape_predictor_68_face_landmarks.dat 的路径。
            face_rec_model_path: dlib dlib_face_recognition_resnet_model_v1.dat 的路径。
            match_threshold: 欧氏距离阈值，低于此值视为匹配。
            resize_width: 图片统一缩放宽度（像素）。

        Raises:
            FileNotFoundError: 模型文件不存在。
        """
        self.predictor_path = Path(predictor_path)
        self.face_rec_model_path = Path(face_rec_model_path)
        self.match_threshold = match_threshold
        self.resize_width = resize_width

        for p in (self.predictor_path, self.face_rec_model_path):
            if not p.exists():
                raise FileNotFoundError(f"模型文件不存在: {p}")

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(self.predictor_path))
        self._facerec = dlib.face_recognition_model_v1(str(self.face_rec_model_path))
        logger.info("人脸比较器已初始化")

    def _resize_image(self, image_path: Path) -> None:
        """等比例缩放图片到指定宽度。

        Args:
            image_path: 图片路径，缩放后就地保存。
        """
        with Image.open(image_path) as img:
            w_percent = self.resize_width / float(img.size[0])
            h_size = int(float(img.size[1]) * w_percent)
            # Pillow 10+ 移除了 ANTIALIAS，使用 LANCZOS
            resized = img.resize((self.resize_width, h_size), Image.LANCZOS)
            resized.save(image_path)
        logger.debug("图片已缩放: %s → 宽度 %d", image_path.name, self.resize_width)

    def _extract_face_descriptor(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.uint8]]:
        """从图像中提取人脸描述符并绘制特征点。

        Args:
            image: BGR 格式的图像数组。

        Returns:
            (face_descriptor, annotated_image) 元组。
            如果未检测到人脸，descriptor 为 None。
        """
        annotated = image.copy()
        faces = self._detector(image, 1)

        if not faces:
            logger.warning("图片中未检测到人脸")
            return None, annotated

        # 取第一张检测到的人脸
        face = faces[0]
        shape = self._predictor(image, face)
        descriptor = self._facerec.compute_face_descriptor(image, shape)

        # 绘制边框
        cv2.rectangle(
            annotated,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (255, 255, 255),
            2,
        )

        # 绘制 68 个特征点
        for i in range(_NUM_LANDMARKS):
            pt = (shape.part(i).x, shape.part(i).y)
            cv2.circle(annotated, pt, 5, (0, 255, 0), -1, cv2.LINE_8)
            cv2.putText(
                annotated,
                str(i),
                pt,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return np.array(descriptor), annotated

    def compare(
        self,
        image1_path: str | Path,
        image2_path: str | Path,
        *,
        resize: bool = True,
        show_result: bool = False,
    ) -> ComparisonResult:
        """比较两张图片中的人脸。

        Args:
            image1_path: 第一张图片路径。
            image2_path: 第二张图片路径。
            resize: 是否预先缩放图片。
            show_result: 是否弹窗显示标注结果。

        Returns:
            ComparisonResult 对象，包含距离、是否匹配和阈值。

        Raises:
            FileNotFoundError: 图片文件不存在。
            ValueError: 某张图片中未检测到人脸。
        """
        path1, path2 = Path(image1_path), Path(image2_path)
        for p in (path1, path2):
            if not p.exists():
                raise FileNotFoundError(f"图片不存在: {p}")

        if resize:
            self._resize_image(path1)
            self._resize_image(path2)

        img1 = cv2.imread(str(path1))
        img2 = cv2.imread(str(path2))

        if img1 is None:
            raise ValueError(f"无法读取图片: {path1}")
        if img2 is None:
            raise ValueError(f"无法读取图片: {path2}")

        desc1, annotated1 = self._extract_face_descriptor(img1)
        desc2, annotated2 = self._extract_face_descriptor(img2)

        if desc1 is None:
            raise ValueError(f"图片 1 中未检测到人脸: {path1}")
        if desc2 is None:
            raise ValueError(f"图片 2 中未检测到人脸: {path2}")

        dist = float(distance.euclidean(desc1, desc2))
        is_match = dist <= self.match_threshold
        result = ComparisonResult(distance=dist, is_match=is_match, threshold=self.match_threshold)

        logger.info("比较结果: %s", result)

        if show_result:
            label = str(result)
            cv2.putText(
                annotated2,
                f"Result: {label}",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Image 1", annotated1)
            cv2.imshow("Image 2", annotated2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result
