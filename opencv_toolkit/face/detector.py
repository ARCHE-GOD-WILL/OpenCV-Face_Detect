"""实时人脸检测模块 — 通过摄像头进行实时人脸识别。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import face_recognition
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class KnownFace:
    """已知人脸的数据类。

    Attributes:
        name: 人脸对应的名字。
        encoding: 128 维人脸编码向量。
    """

    name: str
    encoding: NDArray[np.float64]


class FaceDetector:
    """基于 face_recognition 库的实时人脸检测器。

    通过摄像头捕获视频帧，识别已知人脸并在画面中标注。

    Attributes:
        known_faces: 已注册的已知人脸列表。
        scale_factor: 缩放因子，用于加速人脸检测（默认 0.25）。
        frame_skip: 每隔多少帧执行一次检测（默认 2）。
        box_color: 人脸框的 BGR 颜色（默认红色）。
        label_font_scale: 标签字体大小。
    """

    def __init__(
        self,
        *,
        scale_factor: float = 0.25,
        frame_skip: int = 2,
        box_color: tuple[int, int, int] = (0, 0, 255),
        label_font_scale: float = 1.0,
    ) -> None:
        """初始化人脸检测器。

        Args:
            scale_factor: 检测时的帧缩放因子（0~1），值越小越快但精度降低。
            frame_skip: 每隔多少帧执行检测，用于性能优化。
            box_color: 人脸边框的 BGR 颜色。
            label_font_scale: 标签字号。
        """
        self.known_faces: list[KnownFace] = []
        self.scale_factor = scale_factor
        self.frame_skip = frame_skip
        self.box_color = box_color
        self.label_font_scale = label_font_scale

    def register_face(self, name: str, image_path: str | Path) -> None:
        """注册一张已知人脸图片。

        Args:
            name: 此人脸对应的名字。
            image_path: 人脸图片的文件路径。

        Raises:
            FileNotFoundError: 图片文件不存在。
            ValueError: 图片中未检测到人脸。
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")

        image = face_recognition.load_image_file(str(path))
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            raise ValueError(f"图片中未检测到人脸: {path}")

        self.known_faces.append(KnownFace(name=name, encoding=encodings[0]))
        logger.info("已注册人脸: %s (来源: %s)", name, path.name)

    def _identify_faces(
        self, frame: NDArray[np.uint8]
    ) -> list[tuple[tuple[int, int, int, int], str]]:
        """在单帧中识别人脸。

        Args:
            frame: BGR 格式的图像帧。

        Returns:
            列表，每个元素为 ((top, right, bottom, left), name) 的元组。
        """
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)

        results: list[tuple[tuple[int, int, int, int], str]] = []
        known_encodings = [kf.encoding for kf in self.known_faces]

        for encoding, location in zip(encodings, locations):
            name = "Unknown"
            if known_encodings:
                matches = face_recognition.compare_faces(known_encodings, encoding)
                for i, match in enumerate(matches):
                    if match:
                        name = self.known_faces[i].name
                        break

            # 还原到原始分辨率
            inv = int(1.0 / self.scale_factor)
            top, right, bottom, left = (
                location[0] * inv,
                location[1] * inv,
                location[2] * inv,
                location[3] * inv,
            )
            results.append(((top, right, bottom, left), name))

        return results

    def _draw_annotations(
        self,
        frame: NDArray[np.uint8],
        faces: Sequence[tuple[tuple[int, int, int, int], str]],
    ) -> NDArray[np.uint8]:
        """在帧上绘制人脸标注框和名字。

        Args:
            frame: 原始帧。
            faces: 人脸位置与名字列表。

        Returns:
            标注后的帧。
        """
        annotated = frame.copy()
        for (top, right, bottom, left), name in faces:
            cv2.rectangle(annotated, (left, top), (right, bottom), self.box_color, 2)
            cv2.rectangle(
                annotated, (left, bottom - 35), (right, bottom), self.box_color, cv2.FILLED
            )
            cv2.putText(
                annotated,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                self.label_font_scale,
                (255, 255, 255),
                1,
            )
        return annotated

    def run_webcam(
        self,
        camera_index: int = 0,
        window_name: str = "Face Detection",
        *,
        show_sample: str | Path | None = None,
    ) -> None:
        """启动摄像头进行实时人脸检测。

        按 'q' 键退出。

        Args:
            camera_index: 摄像头索引号（默认 0）。
            window_name: 显示窗口名称。
            show_sample: 可选，同时展示一张样本图片。
        """
        if show_sample is not None:
            sample_path = Path(show_sample)
            if sample_path.exists():
                sample_img = cv2.imread(str(sample_path))
                if sample_img is not None:
                    cv2.imshow("Sample Image", sample_img)

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("无法打开摄像头 %d", camera_index)
            return

        logger.info("摄像头已启动，按 'q' 退出")
        frame_count = 0
        current_faces: list[tuple[tuple[int, int, int, int], str]] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("无法读取视频帧")
                    break

                if frame_count % self.frame_skip == 0:
                    current_faces = self._identify_faces(frame)

                annotated = self._draw_annotations(frame, current_faces)
                cv2.imshow(window_name, annotated)

                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("摄像头已释放")
