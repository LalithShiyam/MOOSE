# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""

import os
import shutil
import tempfile
import unittest

import dicom2nifti
import tests.test_data as test_data
from tests.test_tools import assert_compare_nifti, ground_thruth_filenames


class TestConversionDicom(unittest.TestCase):
    def test_main_function(self):
        tmp_output_dir = tempfile.mkdtemp()
        try:
            results = dicom2nifti.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.SIEMENS_ANATOMICAL)[0])

            results = dicom2nifti.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.SIEMENS_ANATOMICAL)[1])

            results = dicom2nifti.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL_IMPLICIT,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.SIEMENS_ANATOMICAL_IMPLICIT)[0])

            results = dicom2nifti.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL_IMPLICIT,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.SIEMENS_ANATOMICAL_IMPLICIT)[1])

            results = dicom2nifti.dicom_series_to_nifti(test_data.GENERIC_ANATOMICAL,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_ANATOMICAL)[0])

            results = dicom2nifti.dicom_series_to_nifti(test_data.GENERIC_ANATOMICAL,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_ANATOMICAL)[1])

            results = dicom2nifti.dicom_series_to_nifti(test_data.GENERIC_ANATOMICAL_IMPLICIT,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_ANATOMICAL_IMPLICIT)[0])

            results = dicom2nifti.dicom_series_to_nifti(test_data.GENERIC_ANATOMICAL_IMPLICIT,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_ANATOMICAL_IMPLICIT)[1])

            results = dicom2nifti.dicom_series_to_nifti(test_data.GENERIC_COMPRESSED,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_COMPRESSED)[0])

            results = dicom2nifti.dicom_series_to_nifti(test_data.GENERIC_COMPRESSED,
                                                        os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                        True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_COMPRESSED)[1])

        finally:
            shutil.rmtree(tmp_output_dir)

    def test_convert_directory(self):

        tmp_output_dir = tempfile.mkdtemp()
        try:
            dicom2nifti.convert_directory(test_data.GENERIC_ANATOMICAL, tmp_output_dir)

        finally:
            shutil.rmtree(tmp_output_dir)


if __name__ == '__main__':
    unittest.main()
