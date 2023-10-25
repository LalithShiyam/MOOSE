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
# This module contains the functions for performing file operations for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform file operations.
#
# ----------------------------------------------------------------------------------------------------------------------
import pydicom
import SimpleITK
import glob
import os
import shutil
import sys
from datetime import datetime
from multiprocessing import Pool
from typing import List
from moosez import constants


def create_directory(directory_path: str) -> None:
    """
    Creates a directory at the specified path.
    
    :param directory_path: The path to the directory.
    :type directory_path: str
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def get_virtual_env_root() -> str:
    """
    Returns the root directory of the virtual environment.
    
    :return: The root directory of the virtual environment.
    :rtype: str
    """
    python_exe = sys.executable
    virtual_env_root = os.path.dirname(os.path.dirname(python_exe))
    return virtual_env_root


def get_files(directory: str, wildcard: str) -> list:
    """
    Returns the list of files in the directory with the specified wildcard.
    
    :param directory: The directory path.
    :type directory: str
    
    :param wildcard: The wildcard to be used.
    :type wildcard: str
    
    :return: The list of files.
    :rtype: list
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(wildcard):
            files.append(os.path.join(directory, file))
    return files


def moose_folder_structure(parent_directory: str, model_name: str, modalities: list) -> tuple:
    """
    Creates the moose folder structure.
    
    :param parent_directory: The path to the parent directory.
    :type parent_directory: str
    
    :param model_name: The name of the model.
    :type model_name: str
    
    :param modalities: The list of modalities.
    :type modalities: list
    
    :return: A tuple containing the paths to the moose directory, input directories, output directory, and stats directory.
    :rtype: tuple
    """
    moose_dir = os.path.join(parent_directory,
                             'moosez-' + model_name + '-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    create_directory(moose_dir)
    input_dirs = []
    for modality in modalities:
        input_dirs.append(os.path.join(moose_dir, modality))
        create_directory(input_dirs[-1])

    output_dir = os.path.join(moose_dir, constants.SEGMENTATIONS_FOLDER)
    stats_dir = os.path.join(moose_dir, constants.STATS_FOLDER)
    create_directory(output_dir)
    create_directory(stats_dir)
    return moose_dir, input_dirs, output_dir, stats_dir


def copy_file(file: str, destination: str) -> None:
    """
    Copies a file to the specified destination.
    
    :param file: The path to the file to be copied.
    :type file: str
    
    :param destination: The path to the destination directory.
    :type destination: str
    """
    shutil.copy(file, destination)


def copy_files_to_destination(files: list, destination: str) -> None:
    """
    Copies the files inside the list to the destination directory in a parallel fashion.
    
    :param files: The list of files to be copied.
    :type files: list
    
    :param destination: The path to the destination directory.
    :type destination: str
    """
    with Pool(processes=len(files)) as pool:
        pool.starmap(copy_file, [(file, destination) for file in files])


def select_files_by_modality(moose_compliant_subjects: list, modality_tag: str) -> list:
    """
    Selects the files with the selected modality tag from the moose-compliant folders.
    
    :param moose_compliant_subjects: The list of moose-compliant subjects paths.
    :type moose_compliant_subjects: list
    
    :param modality_tag: The modality tag to be selected.
    :type modality_tag: str
    
    :return: The list of selected files.
    :rtype: list
    """
    selected_files = []
    for subject in moose_compliant_subjects:
        files = os.listdir(subject)
        for file in files:
            if file.startswith(modality_tag) and (file.endswith('.nii') or file.endswith('.nii.gz')):
                selected_files.append(os.path.join(subject, file))
    return selected_files


def organise_files_by_modality(moose_compliant_subjects: list, modalities: list, moose_dir: str) -> None:
    """
    Organises the files by modality.
    
    :param moose_compliant_subjects: The list of moose-compliant subjects paths.
    :type moose_compliant_subjects: list
    
    :param modalities: The list of modalities.
    :type modalities: list
    
    :param moose_dir: The path to the moose directory.
    :type moose_dir: str
    """
    for modality in modalities:
        files_to_copy = select_files_by_modality(moose_compliant_subjects, modality)
        copy_files_to_destination(files_to_copy, os.path.join(moose_dir, modality))


def find_pet_file(folder: str) -> str:
    """
    Finds the PET file in the specified folder.
    
    :param folder: The path to the folder.
    :type folder: str
    
    :return: The path to the PET file.
    :rtype: str
    """
    # Searching for files with 'PET' in their name and ending with either .nii or .nii.gz
    pet_files = glob.glob(os.path.join(folder, 'PET*.nii*')) # Files should start with PET

    if len(pet_files) == 1:
        return pet_files[0]
    elif len(pet_files) > 1:
        raise ValueError("More than one PET file found in the directory.")
    else:
        return None


def find_segmentations_folders(directory: str):
    """
    Finds the segmentation folders' directory

    :param directory: The directory to be searched for "segmentation" folder.
    :type directory: str

    """
    moosez_folders = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name.startswith(constants.SEGMENTATIONS_FOLDER):
                moosez_folders.append(os.path.join(root, dir_name))
    return moosez_folders


def find_ct_dicom_folder(root_directory: str):
    """
    Finds the directory contaning the CT DICOM series

    :param root_directory: The directory to be searched for CT DICOMS files.
    :type root_directory: str

    """
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


def is_label_present(segmentation, label_value: int):
    """
    Finds whether a class label exists in a segmentation

    :param root_directory: The directory to be searched for CT DICOMS files.
    :type root_directory: str

    """
    class_label_image = SimpleITK.BinaryThreshold(segmentation, lowerThreshold=label_value, upperThreshold=label_value)

    label_size_filter = SimpleITK.LabelShapeStatisticsImageFilter()
    label_size_filter.Execute(class_label_image)

    return label_size_filter.GetNumberOfLabels() > 0


def custom_sort_key(file_path: str):
    """
    custom sorting key function to sort dcms based on ImagePositionPatient[2]

    :param file_path: the path to a DICOM series directory
    :type file_path: str
    """
    ds = pydicom.dcmread(file_path)
    return float(ds.ImagePositionPatient[2])


def sorted_dicom_series(dicom_path: str):
    """
    Sorts DICOM series based on a key

    :param dicom_path: the path to a DICOM series directory to be sorted
    :type dicom_path: str
    """
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