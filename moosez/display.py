#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 09.02.2023
# Version: 2.0.0
#
# Description:
# This module shows predefined display messages for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to show predefined display
# messages.
#
# ----------------------------------------------------------------------------------------------------------------------

import logging

import emoji
import pyfiglet

from moosez import constants
from moosez import resources

def logo():
    """
    Display MOOSE logo
    :return:
    """
    print(' ')
    logo_color_code = constants.ANSI_VIOLET
    slogan_color_code = constants.ANSI_VIOLET
    result = logo_color_code + pyfiglet.figlet_format(" MOOSE 2.0", font="smslant").rstrip() + "\033[0m"
    text = slogan_color_code + " A part of the ENHANCE community. Join us at www.enhance.pet to build the future of " \
                               "PET imaging together." + "\033[0m"
    print(result)
    print(text)
    print(' ')

def citation():
    """
        Display manuscript citation
        :return:
        """
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":scroll:")} CITATION:{constants.ANSI_RESET}')
    print(" ")
    print(
        " Shiyam Sundar LK, Yu J, Muzik O, et al. Fully-automated, semantic segmentation of whole-body 18F-FDG PET/CT "
        "images based on data-centric artificial intelligence. J Nucl Med. June 2022.")
    print(" Copyright 2022, Quantitative Imaging and Medical Physics Team, Medical University of Vienna")


def expectations(model_name: str) -> list:
    """
    Display expected modality for the model. This is used to check if the user has provided the correct modality.
    :param model_name: The name of the model.
    :return: list of modalities
    """
    model_info = resources.expected_modality(model_name)
    modality = model_info['Modality']

    # check for special case where 'FDG-PET-CT' should be split into 'FDG-PET' and 'CT'
    if modality == 'FDG-PET-CT':
        modalities = ['FDG-PET', 'CT']
    else:
        modalities = [modality]
    expected_prefix = [m.replace('-', '_') + "_" for m in modalities]

    print(
        f" Imaging: {model_info['Imaging']} |"
        f" Modality: {modality} | "
        f"Tissue of interest: {model_info['Tissue of interest']} | "
        f"nnUNet version: {model_info['nnUNet version']} ")
    print(
        f" Required modalities: {modalities} | "
        f" No. of modalities: {len(modalities)}"
        f" | Required prefix for non-DICOM files: {expected_prefix}")
    logging.info(f" Required modalities: {modalities} |  No. of modalities: {len(modalities)} "
                 f"| Required prefix for non-DICOM files: {expected_prefix} ")
    print(
        f"{constants.ANSI_ORANGE} Warning: Subjects which don't have the required modalities [check file prefix] "
        f"will be skipped. {constants.ANSI_RESET}")
    warning_message = " Skipping subjects without the required modalities (check file prefix).\n" \
                      " These subjects will be excluded from analysis and their data will not be used."
    logging.warning(warning_message)

    return modalities
