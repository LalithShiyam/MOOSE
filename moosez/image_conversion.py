#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
#         Sebastian Gutschmayer
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 09.02.2023
# Version: 2.0.0
#
# Description:
# This module handles image conversion for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform image conversion.
#
# ----------------------------------------------------------------------------------------------------------------------

import json
import os
from pathlib import Path

import SimpleITK
import dcm2niix
from rich.progress import Progress


def read_dicom_folder(folder_path: str) -> SimpleITK.Image:
    """
    Reads a folder of DICOM files and returns the image

    :param folder_path: str, Directory to get DICOM files from
    """
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)

    dicom_image = reader.Execute()
    return dicom_image


def non_nifti_to_nifti(input_path: str, output_directory: str = None) -> None:
    """
    Converts any image format known to ITK to NIFTI

    :param input_path: str, Directory OR filename to convert to nii.gz
    :param output_directory: str, optional output directory to write the image to.
    """
    subject_name = os.path.basename(os.path.dirname(input_path))
    output_image_basename = "output"
    output_image = None  # initialize output_image
    if os.path.isdir(input_path):
        dcm2niix_conversion(input_path)
        return
    elif os.path.isfile(input_path):
        if input_path.endswith('.nii.gz') or input_path.endswith('.nii'):
            return
        output_image = SimpleITK.ReadImage(input_path)
        output_image_basename = f"{os.path.splitext(os.path.basename(input_path))[0]}.nii"
    else:
        return

    if output_directory is None:
        output_directory = os.path.dirname(input_path)
    output_image_path = os.path.join(output_directory, output_image_basename)
    SimpleITK.WriteImage(output_image, output_image_path)


def standardize_to_nifti(parent_dir: str):
    """
    Converts all images in a parent directory to NIFTI
    """
    # go through the subdirectories
    subjects = os.listdir(parent_dir)
    # get only the directories
    subjects = [subject for subject in subjects if os.path.isdir(os.path.join(parent_dir, subject))]

    with Progress() as progress:
        task = progress.add_task("[white] Processing subjects...", total=len(subjects))
        for subject in subjects:
            subject_path = os.path.join(parent_dir, subject)
            if os.path.isdir(subject_path):
                images = os.listdir(subject_path)
                for image in images:
                    if os.path.isdir(os.path.join(subject_path, image)):
                        image_path = os.path.join(subject_path, image)
                        non_nifti_to_nifti(image_path)
                    elif os.path.isfile(os.path.join(subject_path, image)):
                        image_path = os.path.join(subject_path, image)
                        non_nifti_to_nifti(image_path)
            else:
                continue
            progress.update(task, advance=1, description=f"[white] Processing {subject}...")


def dcm2niix_conversion(input_path: str) -> None:
    """
    Converts DICOM images into Nifti images using dcm2niix
    :param input_path: Path to the folder with the dicom files to convert
    """
    output_dir = Path(input_path).parent  # One level up from the DICOM directory

    # Run dcm2niix
    dcm2niix.main([
        '-o', str(output_dir),  # Output directory
        input_path  # Input DICOM directory
    ])
    identify_and_cleanup(output_dir)


def identify_scan_type(json_file):
    # Load the JSON data
    with open(json_file) as f:
        json_data = json.load(f)

    # Check the modality
    modality = json_data.get("Modality")
    if modality == "PT":
        return "PT"
    elif modality == "CT":
        # Get the corresponding NIfTI file
        nifti_file = json_file.with_suffix('.nii')

        # Check for "Topogram" in "ImageType"
        if "Topogram" in json_data.get("ImageType", "") and nifti_file.is_file():
            # Get the directory containing the JSON file
            directory = json_file.parent

            # Get all other CT NIfTI files in the directory
            ct_files = [f for f in directory.glob('*.nii') if
                        f != nifti_file and identify_scan_type(f.with_suffix('.json')) == "CT"]

            # If the NIfTI file is smaller than all other CT files, it's a topogram
            if all(nifti_file.stat().st_size < other_file.stat().st_size for other_file in ct_files):
                return "Topogram"

        # If it's not a topogram, it's an original CT
        return "Original CT"
    else:
        return "Unknown"


def identify_and_cleanup(directory):
    # Iterate over all JSON files in the directory
    for json_file in Path(directory).glob('*.json'):
        # Identify the scan type
        scan_type = identify_scan_type(json_file)

        # Get the corresponding NIfTI file
        nifti_file = json_file.with_suffix('.nii')

        if scan_type == "PT":
            # Load the JSON data
            with open(json_file) as f:
                json_data = json.load(f)
            # Check the radiotracer name
            radiotracer = json_data.get("Radiopharmaceutical")
            if radiotracer == "Fluorodeoxyglucose":
                # Rename the NIfTI file
                nifti_file.rename(nifti_file.with_stem(f'FDG_PET_{nifti_file.stem}'))
            else:
                # Rename the NIfTI file
                nifti_file.rename(nifti_file.with_stem(f'PET_{nifti_file.stem}'))
        elif scan_type == "Original CT":
            # Rename the NIfTI file
            nifti_file.rename(nifti_file.with_stem(f'CT_{nifti_file.stem}'))
        else:
            # Delete the NIfTI file and the JSON file
            if nifti_file.is_file():
                nifti_file.unlink()
            json_file.unlink()
