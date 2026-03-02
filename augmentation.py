import albumentations as A
import cv2
import numpy as np

def get_augmentation_pipeline():
    """CCTV-realistic augmentation pipeline"""
    return A.Compose([
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=7, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        ], p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def augment_image(image, bboxes, labels):
    """Apply augmentation to image and bboxes"""
    transform = get_augmentation_pipeline()
    transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
    return transformed['image'], transformed['bboxes']
