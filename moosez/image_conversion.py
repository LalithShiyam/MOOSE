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

import contextlib
import io
import logging
import os
import re
import unicodedata
import glob
from typing import List, Dict
import json
import shutil
import tempfile
from datetime import datetime
import SimpleITK
import dicom2nifti
import pydicom
from rich.progress import Progress
from moosez import constants
from moosez import nifti2dicom
from moosez import file_utilities


def read_dicom_folder(folder_path: str) -> SimpleITK.Image:
    """
    Reads a folder of DICOM files and returns the image.

    :param folder_path: str
        The path to the folder containing the DICOM files.
    :type folder_path: str

    :return: SimpleITK.Image
        The image obtained from the DICOM files.
    :rtype: SimpleITK.Image
    """
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)

    dicom_image = reader.Execute()
    return dicom_image


    
def non_nifti_to_nifti(input_path: str, output_directory: str = None) -> None:
    """
    Converts any image format known to ITK to NIFTI

        :param input_path: The path to the directory or filename to convert to nii.gz.
        :type input_path: str
        
        :param output_directory: Optional. The output directory to write the image to. If not specified, the output image will be written to the same directory as the input image.
        :type output_directory: str
        
        :return: None
        :rtype: None
        
        :raises: FileNotFoundError if the input path does not exist.
        
        Usage:
        This function can be used to convert any image format known to ITK to NIFTI. If the input path is a directory, the function will convert all images in the directory to NIFTI format. If the input path is a file, the function will convert the file to NIFTI format. The output image will be written to the specified output directory, or to the same directory as the input image if no output directory is specified.
    """

    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist.")
        return

    # Processing a directory
    if os.path.isdir(input_path):
        dicom_info = create_dicom_lookup(input_path)
        nifti_dir = dcm2niix(input_path)
        rename_nifti_files(nifti_dir, dicom_info)
        return

    # Processing a file
    if os.path.isfile(input_path):
        # Ignore hidden or already processed files
        _, filename = os.path.split(input_path)
        if filename.startswith('.') or filename.endswith(('.nii.gz', '.nii')):
            return
        else:
            output_image = SimpleITK.ReadImage(input_path)
            output_image_basename = f"{os.path.splitext(filename)[0]}.nii"

    if output_directory is None:
        output_directory = os.path.dirname(input_path)

    output_image_path = os.path.join(output_directory, output_image_basename)
    SimpleITK.WriteImage(output_image, output_image_path)


def standardize_to_nifti(parent_dir: str) -> None:
    """
    Converts all non-NIfTI images in a parent directory and its subdirectories to NIfTI format.

    :param parent_dir: The path to the parent directory containing the images to convert.
    :type parent_dir: str
    :return: None
    """
    # Get a list of all subdirectories in the parent directory
    subjects = os.listdir(parent_dir)
    subjects = [subject for subject in subjects if os.path.isdir(os.path.join(parent_dir, subject))]

    # Convert all non-NIfTI images in each subdirectory to NIfTI format
    with Progress() as progress:
        task = progress.add_task("[white] Processing subjects...", total=len(subjects))
        for subject in subjects:
            subject_path = os.path.join(parent_dir, subject)
            if os.path.isdir(subject_path):
                images = os.listdir(subject_path)
                for image in images:
                    image_path = os.path.join(subject_path, image)
                    path_is_valid = os.path.isdir(image_path) | os.path.isfile(image_path)
                    path_is_valid = path_is_valid & ("moosez" not in os.path.basename(image_path))
                    if path_is_valid:
                        non_nifti_to_nifti(image_path)
            else:
                continue
            progress.update(task, advance=1, description=f"[white] Processing {subject}...")


def dcm2niix(input_path: str) -> str:
    """
    Converts DICOM images into NIfTI images using dcm2niix.

    :param input_path: The path to the folder containing the DICOM files to convert.
    :type input_path: str
    :return: The path to the folder containing the converted NIfTI files.
    :rtype: str
    """
    output_dir = os.path.dirname(input_path)

    # Redirect standard output and standard error to discard output
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dicom2nifti.convert_directory(input_path, output_dir, compression=False, reorient=True)

    return output_dir


def remove_accents(unicode_filename: str) -> str:
    """
    Removes accents and special characters from a Unicode filename.

    :param unicode_filename: The Unicode filename to clean.
    :type unicode_filename: str
    :return: The cleaned filename.
    :rtype: str
    """
    try:
        unicode_filename = str(unicode_filename).replace(" ", "_")
        cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore').decode('ASCII')
        cleaned_filename = re.sub(r'[^\w\s-]', '', cleaned_filename.strip().lower())
        cleaned_filename = re.sub(r'[-\s]+', '-', cleaned_filename)
        return cleaned_filename
    except:
        return unicode_filename


def is_dicom_file(filename: str) -> bool:
    """
    Checks if a file is a DICOM file.

    :param filename: The path to the file to check.
    :type filename: str
    :return: True if the file is a DICOM file, False otherwise.
    :rtype: bool
    """
    try:
        pydicom.dcmread(filename)
        return True
    except pydicom.errors.InvalidDicomError:
        return False

def create_dicom_lookup(dicom_dir: str) -> dict:
    """
    Create a lookup dictionary from DICOM files.

    :param dicom_dir: The directory where DICOM files are stored.
    :type dicom_dir: str
    :return: A dictionary where the key is the anticipated filename that dicom2nifti will produce and
             the value is the modality of the DICOM series.
    :rtype: dict
    """
    dicom_info = {}

    for filename in os.listdir(dicom_dir):
        full_path = os.path.join(dicom_dir, filename)
        if is_dicom_file(full_path):
            ds = pydicom.dcmread(full_path)

            series_number = ds.SeriesNumber if 'SeriesNumber' in ds else False
            series_description = ds.SeriesDescription if 'SeriesDescription' in ds else False
            sequence_name = ds.SequenceName if 'SequenceName' in ds else False
            protocol_name = ds.ProtocolName if 'ProtocolName' in ds else False
            series_instance_UID = ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else None

            if ds.Modality == 'PT':
                modality = 'PET'
            else:
                modality = ds.Modality

            if series_number is not False:
                base_filename = remove_accents(series_number)
                if series_description is not False:
                    anticipated_filename = f"{base_filename}_{remove_accents(series_description)}.nii"
                elif sequence_name is not False:
                    anticipated_filename = f"{base_filename}_{remove_accents(sequence_name)}.nii"
                elif protocol_name is not False:
                    anticipated_filename = f"{base_filename}_{remove_accents(protocol_name)}.nii"
            else:
                anticipated_filename = f"{remove_accents(series_instance_UID)}.nii"

            dicom_info[anticipated_filename] = modality

    return dicom_info


def rename_nifti_files(nifti_dir: str, dicom_info: dict) -> None:
    """
    Rename NIfTI files based on a lookup dictionary.

    :param nifti_dir: The directory where NIfTI files are stored.
    :type nifti_dir: str
    :param dicom_info: A dictionary where the key is the anticipated filename that dicom2nifti will produce and
                       the value is the modality of the DICOM series.
    :type dicom_info: dict
    """
    for filename in os.listdir(nifti_dir):
        if filename.endswith('.nii'):
            modality = dicom_info.get(filename, '')
            if modality:
                new_filename = f"{modality}_{filename}"
                old_filepath = os.path.join(nifti_dir, filename)
                new_filepath = os.path.join(nifti_dir, new_filename)

                # Move and overwrite the file if it already exists
                shutil.move(old_filepath, new_filepath)
                del dicom_info[filename]


def find_segmentations_folders(directory):
    moosez_folders = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name.startswith(constants.SEGMENTATIONS_FOLDER):
                moosez_folders.append(os.path.join(root, dir_name))
    return moosez_folders


def find_ct_dicom_folder(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            dicom_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.dcm')]
            if dicom_files:
                # Check the first DICOM file in the directory for modality
                first_dicom_file = pydicom.dcmread(os.path.join(dir_path, dicom_files[0]))
                if hasattr(first_dicom_file, 'Modality') and first_dicom_file.Modality == 'CT':
                    return dir_path
    return None


def is_label_present(segmentation, label_value):
    class_label_image = SimpleITK.BinaryThreshold(segmentation, lowerThreshold=label_value, upperThreshold=label_value)

    label_size_filter = SimpleITK.LabelShapeStatisticsImageFilter()
    label_size_filter.Execute(class_label_image)

    return label_size_filter.GetNumberOfLabels() > 0


def custom_sort_key(file_path):
    # custom sorting key function to sort based on ImagePositionPatient[2]
    ds = pydicom.dcmread(file_path)
    return float(ds.ImagePositionPatient[2])


def sorted_dicom_series(dicom_path: str):
    dcm_files = [os.path.join(dicom_path, file) for file in os.listdir(dicom_path) if
                 os.path.isfile(os.path.join(dicom_path, file))]
    # Sorting the list of DICOM file paths using the custom sorting key
    sorted_dcms = sorted(dcm_files, key=custom_sort_key)
    return sorted_dcms

def get_first_n_files(sorted_dcms: List[str], n: int):
    selected_files = []
    for i, file_path in enumerate(sorted_dcms):
        if os.path.isfile(file_path):
            selected_files.append(file_path)
            if len(selected_files) == n:
                break

    if len(selected_files) < n:
        print(f"Only {len(selected_files)} file(s) found in the list.")
    return selected_files

def nifti2dicom_process(subject_path: str, **kwargs: Dict):
    if os.path.isdir(subject_path):
        ct_dicom_dir = find_ct_dicom_folder(subject_path)
        sorted_dcms = sorted_dicom_series(ct_dicom_dir)

        segmentation_dir = find_segmentations_folders(subject_path)
        for i, seg_dir in enumerate(segmentation_dir):
            if os.path.exists(os.path.join(os.path.dirname(seg_dir), constants.DICOM_SEGS_FOLDER)):
                continue
            else:
                nii_files = glob.glob(os.path.join(seg_dir, "*.nii.gz"))
                dataset_json_path = os.path.join(seg_dir, 'dataset.json')
                for nii_file in nii_files:
                    original_seg = SimpleITK.ReadImage(nii_file)
                    # Load the JSON file containing label information
                    with open(dataset_json_path, "r") as json_file:
                        label_data = json.load(json_file)

                    labels = label_data.get("labels", {})

                    temp_directory = tempfile.mkdtemp()
                    nifti_path = os.path.join(temp_directory, 'niftis')
                    if not os.path.exists(nifti_path):
                        os.mkdir(nifti_path)

                    for class_name, class_label in labels.items():
                        if class_name == 'background':
                            continue

                        if not is_label_present(original_seg, class_label):
                            print(
                                f"Label '{class_name}' with label value {class_label} not found in the original segmentation. "
                                f"Skipping...")
                            continue

                        class_label_image = SimpleITK.BinaryThreshold(original_seg, lowerThreshold=class_label,
                                                                      upperThreshold=class_label)
                        class_name = class_name.lower().replace(" ", "_")
                        class_label_image.CopyInformation(original_seg)

                        if kwargs.get('crop', False):
                            # Calculate label statistics
                            copy_seg = SimpleITK.Cast(original_seg, SimpleITK.sitkInt16)
                            label_stats = SimpleITK.LabelShapeStatisticsImageFilter()
                            label_stats.Execute(copy_seg)

                            min_x, min_y, min_z, size_x, size_y, size_z = label_stats.GetRegion(class_label)
                            class_label_image = SimpleITK.RegionOfInterest(class_label_image, (size_x, size_y, size_z),
                                                                              (min_x, min_y, min_z))

                        output_nifti_dir = os.path.join(nifti_path, f"{class_name}.nii.gz")
                        SimpleITK.WriteImage(class_label_image, output_nifti_dir)

                    # Get n CT dcm seeds
                    dicom_seed = get_first_n_files(sorted_dcms, 1)

                    if kwargs.get('dicom_out_dir', False):
                        moosez_folder = os.path.basename(os.path.dirname(seg_dir))
                        subject_folder = os.path.basename(subject_path)
                        out_dir = kwargs['dicom_out_dir']
                        dicom_folder = os.path.join(out_dir, subject_folder + '-' +moosez_folder, constants.DICOM_SEGS_FOLDER)
                    else:
                        dicom_folder = os.path.join(os.path.dirname(seg_dir), constants.DICOM_SEGS_FOLDER)
                    if not os.path.exists(dicom_folder):
                        file_utilities.create_directory(dicom_folder)

                    if kwargs.get('crop', False):
                        nifti2dicom.convert_nifti_to_dicom_seg(nifti_path, dicom_seed, dicom_folder,
                                                               cropping=kwargs.get('crop', False))
                    else:
                        nifti2dicom.convert_nifti_to_dicom_seg(nifti_path, dicom_seed, dicom_folder)
                    shutil.rmtree(temp_directory)
