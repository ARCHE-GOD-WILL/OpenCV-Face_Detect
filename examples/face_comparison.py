#!/usr/bin/env python3
"""示例：比较两张图片中的人脸是否为同一人。

Usage:
    python -m examples.face_comparison \\
        --predictor shape_predictor_68_face_landmarks.dat \\
        --model dlib_face_recognition_resnet_model_v1.dat \\
        --image1 face1.jpg --image2 face2.jpg
"""

from __future__ import annotations

import argparse
import logging

from opencv_toolkit.face import FaceComparator


def main() -> None:
    """运行人脸比较示例。"""
    parser = argparse.ArgumentParser(description="人脸比较")
    parser.add_argument("--predictor", required=True, help="dlib 68 点特征检测模型路径")
    parser.add_argument("--model", required=True, help="dlib 人脸识别模型路径")
    parser.add_argument("--image1", required=True, help="第一张图片路径")
    parser.add_argument("--image2", required=True, help="第二张图片路径")
    parser.add_argument("--threshold", type=float, default=0.45, help="匹配阈值 (默认: 0.45)")
    parser.add_argument("--show", action="store_true", help="弹窗显示结果")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    comparator = FaceComparator(
        predictor_path=args.predictor,
        face_rec_model_path=args.model,
        match_threshold=args.threshold,
    )

    result = comparator.compare(
        args.image1, args.image2, show_result=args.show
    )

    print(f"比较结果: {result}")
    print(f"  欧氏距离: {result.distance:.4f}")
    print(f"  阈值: {result.threshold}")
    print(f"  匹配: {'是' if result.is_match else '否'}")


if __name__ == "__main__":
    main()
