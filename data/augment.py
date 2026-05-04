"""
数据增强 —— 基于 albumentations
针对座舱场景优化: 模拟夜间暗光、逆光、墨镜遮挡等情况
"""

import albumentations as A


def get_train_transforms(img_size: int = 640):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        # 模拟光照变化 (夜间暗光 / 逆光)
        A.RandomBrightnessContrast(
            brightness_limit=(-0.4, 0.3),
            contrast_limit=(-0.3, 0.3),
            p=0.6,
        ),
        # 模拟墨镜 / 遮挡
        A.CoarseDropout(
            max_holes=3, max_height=img_size // 8, max_width=img_size // 8,
            min_holes=1, min_height=img_size // 16, min_width=img_size // 16,
            fill_value=0, p=0.3,
        ),
        # 色调偏移 (IR 摄像头 / 不同车内光线)
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4,
        ),
        # 模糊 (运动模糊 / 对焦不清)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        # JPEG 压缩噪声
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    ])


def get_val_transforms(img_size: int = 640):
    """验证集不做增强."""
    return None
