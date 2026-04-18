#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据转换脚本

使用方法:
    # MAT文件转PNG图片
    python scripts/convert_data.py mat2images --input data.mat --output data/processed/images

    # NPY文件转MAT文件
    python scripts/convert_data.py npy2mat --input features.npy --output features.mat
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.converters import mat_to_images, npy_to_mat


def main():
    parser = argparse.ArgumentParser(description='Data conversion utilities')
    subparsers = parser.add_subparsers(dest='command', help='Conversion type')

    # mat2images 子命令
    mat2img_parser = subparsers.add_parser('mat2images', help='Convert MAT file to PNG images')
    mat2img_parser.add_argument('--input', '-i', required=True, help='Input MAT file path')
    mat2img_parser.add_argument('--output', '-o', required=True, help='Output directory')

    # npy2mat 子命令
    npy2mat_parser = subparsers.add_parser('npy2mat', help='Convert NPY file to MAT file')
    npy2mat_parser.add_argument('--input', '-i', required=True, help='Input NPY file path')
    npy2mat_parser.add_argument('--output', '-o', required=True, help='Output MAT file path')

    args = parser.parse_args()

    if args.command == 'mat2images':
        print("=" * 60)
        print("Converting MAT file to PNG images")
        print("=" * 60)
        mat_to_images(args.input, args.output)

    elif args.command == 'npy2mat':
        npy_to_mat(args.input, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
