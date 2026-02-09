#!/usr/bin/env python3
"""示例：使用摄像头进行实时人脸检测。

Usage:
    python -m examples.face_detection_webcam --image sample.jpg --name "John Doe"
"""

from __future__ import annotations

import argparse
import logging

from opencv_toolkit.face import FaceDetector


def main() -> None:
    """运行实时人脸检测示例。"""
    parser = argparse.ArgumentParser(description="实时人脸检测")
    parser.add_argument("--image", required=True, help="已知人脸的参考图片路径")
    parser.add_argument("--name", required=True, help="参考图片中人物的姓名")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引号 (默认: 0)")
    parser.add_argument("--scale", type=float, default=0.25, help="检测缩放因子 (默认: 0.25)")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    detector = FaceDetector(scale_factor=args.scale)
    detector.register_face(args.name, args.image)

    print(f"已注册人脸: {args.name}")
    print("启动摄像头... 按 'q' 退出")
    detector.run_webcam(camera_index=args.camera, show_sample=args.image)


if __name__ == "__main__":
    main()
