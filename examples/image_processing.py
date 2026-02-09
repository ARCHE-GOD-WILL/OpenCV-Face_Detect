#!/usr/bin/env python3
"""示例：图像处理工具集（颜色移除、增强、线条检测、倾斜校正、印章检测）。

Usage:
    # 移除红色
    python -m examples.image_processing color-remove input.jpg output.jpg

    # 图像增强
    python -m examples.image_processing enhance input.jpg output.jpg

    # 线条检测
    python -m examples.image_processing line-detect input.jpg output_dir/

    # 倾斜校正
    python -m examples.image_processing skew-correct input.jpg output.jpg

    # 印章检测
    python -m examples.image_processing stamp-detect input.jpg output.jpg
"""

from __future__ import annotations

import argparse
import logging

from opencv_toolkit.image import (
    ColorRemover,
    ImageEnhancer,
    LineDetector,
    SkewCorrector,
    StampDetector,
)
from opencv_toolkit.image.color_remover import RED_RANGES


def cmd_color_remove(args: argparse.Namespace) -> None:
    """移除红色。"""
    remover = ColorRemover(color_ranges=RED_RANGES)
    remover.process_file(args.input, args.output)
    print(f"已保存: {args.output}")


def cmd_enhance(args: argparse.Namespace) -> None:
    """图像增强。"""
    enhancer = ImageEnhancer(kernel_size=args.kernel_size)
    enhancer.enhance_file(args.input, args.output, grayscale=args.grayscale)
    print(f"已保存: {args.output}")


def cmd_line_detect(args: argparse.Namespace) -> None:
    """线条检测。"""
    detector = LineDetector()
    result = detector.process_file(args.input, args.output, direction=args.direction)
    print(f"检测到水平线: {len(result.horizontal_positions)} 条")
    print(f"检测到垂直线: {len(result.vertical_positions)} 条")
    print(f"生成切片: {len(result.slices)} 个")


def cmd_skew_correct(args: argparse.Namespace) -> None:
    """倾斜校正。"""
    corrector = SkewCorrector()
    result = corrector.correct_file(args.input, args.output, show=args.show)
    print(f"检测到倾斜角度: {result.angle:.3f}°")
    print(f"已保存: {args.output}")


def cmd_stamp_detect(args: argparse.Namespace) -> None:
    """印章检测。"""
    detector = StampDetector()
    result = detector.detect_file(args.input, args.output)
    print(f"检测到 {len(result.stamps)} 个印章:")
    for i, stamp in enumerate(result.stamps, 1):
        print(f"  {i}. {stamp.shape} at {stamp.center}")


def main() -> None:
    """图像处理工具 CLI 入口。"""
    parser = argparse.ArgumentParser(description="OpenCV Toolkit 图像处理工具集")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 颜色移除
    p_color = subparsers.add_parser("color-remove", help="移除指定颜色")
    p_color.add_argument("input", help="输入图片路径")
    p_color.add_argument("output", help="输出图片路径")
    p_color.set_defaults(func=cmd_color_remove)

    # 图像增强
    p_enhance = subparsers.add_parser("enhance", help="锐化 + 去噪增强")
    p_enhance.add_argument("input", help="输入图片路径")
    p_enhance.add_argument("output", help="输出图片路径")
    p_enhance.add_argument("--kernel-size", type=int, default=9, help="锐化核大小 (默认: 9)")
    p_enhance.add_argument("--no-grayscale", dest="grayscale", action="store_false")
    p_enhance.set_defaults(func=cmd_enhance)

    # 线条检测
    p_line = subparsers.add_parser("line-detect", help="线条检测与切割")
    p_line.add_argument("input", help="输入图片路径")
    p_line.add_argument("output", help="输出目录路径")
    p_line.add_argument(
        "--direction",
        choices=["horizontal", "vertical", "both"],
        default="horizontal",
        help="检测方向 (默认: horizontal)",
    )
    p_line.set_defaults(func=cmd_line_detect)

    # 倾斜校正
    p_skew = subparsers.add_parser("skew-correct", help="倾斜校正")
    p_skew.add_argument("input", help="输入图片路径")
    p_skew.add_argument("output", help="输出图片路径")
    p_skew.add_argument("--show", action="store_true", help="显示校正对比")
    p_skew.set_defaults(func=cmd_skew_correct)

    # 印章检测
    p_stamp = subparsers.add_parser("stamp-detect", help="印章检测")
    p_stamp.add_argument("input", help="输入图片路径")
    p_stamp.add_argument("output", nargs="?", default=None, help="标注输出路径")
    p_stamp.set_defaults(func=cmd_stamp_detect)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
