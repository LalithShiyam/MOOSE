#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
#         Sebastian Gutschmayer
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 17.02.2023
# Version: 2.0.0
#
# Description:
# This module handles image processing for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform image processing.
#
# ----------------------------------------------------------------------------------------------------------------------

import SimpleITK
import os


# Reslice identity
def reslice_identity(reference_image: SimpleITK.Image, image: SimpleITK.Image,
                     output_image_path: str = None, is_label_image: bool = False) -> SimpleITK.Image:
    """
    Reslice an image to the same space as another image
    :param reference_image: The reference image
    :param image: The image to reslice
    :param output_image_path: Path to the resliced image
    :param is_label_image: Determines if the image is a label image. Default is False

    """
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)

    if is_label_image:
        resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(SimpleITK.sitkLinear)

    resampled_image = resampler.Execute(image)
    if output_image_path is not None:
        SimpleITK.WriteImage(resampled_image, output_image_path)
    return resampled_image


# Shift intensity
def shift_intensity(image: SimpleITK.Image, shift: int, output_image_path: str = None) -> SimpleITK.Image:
    """
    Shift the intensity of an image
    :param image: The image to shift
    :param shift: Amount to shift the image by
    :param output_image_path: Path to the shifted image
    """
    shifted_image = image + shift
    if output_image_path is not None:
        SimpleITK.WriteImage(shifted_image, output_image_path)
    return shifted_image


def sum_images(images: list, output_image_path: str = None) -> SimpleITK.Image:
    # Start with the first image
    summed_image = SimpleITK.ReadImage(images[0])

    # Sum all other images
    for image_index in range(1, len(images)):
        summed_image += SimpleITK.ReadImage(images[image_index])

    if output_image_path is not None:
        SimpleITK.WriteImage(summed_image, output_image_path)

    return summed_image


def scale_intensity(image: SimpleITK.Image, scale: int, output_image_path: str = None) -> SimpleITK.Image:
    """
    Scales the intensity of an image
    :param image: The image to shift
    :param scale: Amount to shift the image by
    :param output_image_path: Path to the shifted image
    """
    scaled_image = image * scale
    if output_image_path is not None:
        SimpleITK.WriteImage(scaled_image, output_image_path)
    return scaled_image


def binarize_image(image: SimpleITK.Image, threshold=1, output_image_path: str = None) -> SimpleITK.Image:
    """
    Binarizes an image according to the threshold
    :param image: The image to shift
    :param threshold: Threshold for binarizing
    :param output_image_path: Path to the shifted image
    """
    binarized_image = image > threshold
    if output_image_path is not None:
        SimpleITK.WriteImage(binarized_image, output_image_path)
    return binarized_image
