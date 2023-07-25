#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 13.02.2023
# Version: 2.0.0
#
# Description:
# This module contains the constants that are used in the moosez.
#
# Usage:
# The variables in this module can be imported and used in other modules within the moosez.
#
# ----------------------------------------------------------------------------------------------------------------------

import os

from moosez import file_utilities

project_root = file_utilities.get_virtual_env_root()

NNUNETV2_MODEL_FOLDER = os.path.join(project_root, 'models', 'nnunet_trained_models')
NNUNETV1_MODEL_FOLDER = os.path.join(NNUNETV2_MODEL_FOLDER, 'nnUNet', '3d_fullres')
ALLOWED_MODALITIES = ['CT', 'PT', 'MR']
TEMP_FOLDER = 'temp'

# COLOR CODES
ANSI_ORANGE = '\033[38;5;208m'
ANSI_GREEN = '\033[38;5;40m'
ANSI_VIOLET = '\033[38;5;141m'
ANSI_RESET = '\033[0m'

# SUPPORTED TRACERS (limited patch)

TRACER_FDG = 'FDG'

# FOLDER NAMES

SEGMENTATIONS_FOLDER = 'segmentations'
STATS_FOLDER = 'stats'

# PREPROCESSING PARAMETERS

MATRIX_THRESHOLD = 200 * 200 * 600
Z_AXIS_THRESHOLD = 200
MARGIN_PADDING = 20
INTERPOLATION = 'bspline'
CHUNK_THRESHOLD = 200

# POSTPROCESSING PARAMETERS

TUMOR_LABEL = 12

# FILE NAMES

RESAMPLED_IMAGE_FILE_NAME = 'resampled_image_0000.nii.gz'
RESAMPLED_MASK_FILE_NAME = 'resampled_mask.nii.gz'
CHUNK_FILENAMES = ["chunk01_0000.nii.gz", "chunk02_0000.nii.gz", "chunk03_0000.nii.gz"]
CHUNK_PREFIX = 'chunk'

# ORGAN INDICES

ORGAN_INDICES = {
    "clin_ct_organs_v2": {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "aorta",
        8: "inferior_vena_cava",
        9: "portal_vein_and_splenic_vein",
        10: "pancreas",
        11: "adrenal_gland_right",
        12: "adrenal_gland_left",
        13: "lung_upper_lobe_left",
        14: "lung_lower_lobe_left",
        15: "lung_upper_lobe_right",
        16: "lung_middle_lobe_right",
        17: "lung_lower_lobe_right"
    },
    "clin_ct_lungs_v2": {
        1: "lung_upper_lobe_left",
        2: "lung_lower_lobe_left",
        3: "lung_upper_lobe_right",
        4: "lung_middle_lobe_right",
        5: "lung_lower_lobe_right"
    },
    "clin_ct_body_v2": {
        1: "whole-body"
    },
    "clin_pt_fdg_tumor_v2": {
        1: "tumor"
    },
    "clin_ct_organs_v1": {
        1: 'Adrenal-glands',
        2: 'Aorta',
        3: 'Bladder',
        4: 'Brain',
        5: 'Heart',
        6: 'Kidneys',
        7: 'Liver',
        8: 'Pancreas',
        9: 'Spleen',
        10: 'Thyroid',
        11: 'Inferior-vena-cava',
        12: 'Skeleton',
        13: 'Lung'
    },
    "clin_ct_bones_v1": {
        1: 'Carpal',
        2: 'Clavicle',
        3: 'Femur',
        4: 'Fibula',
        5: 'Humerus',
        6: 'Metacarpal',
        7: 'Metatarsal',
        8: 'Patella',
        9: 'Pelvis',
        10: 'Phalanges-of-the-hand',
        11: 'Radius',
        12: 'Ribcage',
        13: 'Scapula',
        14: 'Skull',
        15: 'Spine',
        16: 'Sternum',
        17: 'Tarsal',
        18: 'Tibia',
        19: 'Phalanges-of-the-feet',
        20: 'Ulna'
    },
    "clin_ct_fat_muscles_v1": {
        1: 'Skeletal-muscle',
        2: 'Subcutaneous-fat',
        3: 'Torso-fat'
    },
    # More index-to-name dictionaries for other models...
}


