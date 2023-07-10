# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""

import os
import shutil
import tempfile
import unittest

import dicom2nifti.convert_dicom as convert_dicom
import dicom2nifti.settings as settings
import nibabel
import tests.test_data as test_data
from dicom2nifti.common import read_dicom_directory
from tests.test_tools import assert_compare_nifti, ground_thruth_filenames


class TestConversionDicom(unittest.TestCase):
    def test_anatomical(self):
        tmp_output_dir = tempfile.mkdtemp()
        try:
            results = convert_dicom.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL,
                                                          None,
                                                          False)
            self.assertTrue(results.get('NII_FILE') is None)
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.SIEMENS_ANATOMICAL)[0])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.SIEMENS_ANATOMICAL)[1])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL_IMPLICIT,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.SIEMENS_ANATOMICAL_IMPLICIT)[0])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.SIEMENS_ANATOMICAL_IMPLICIT,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.SIEMENS_ANATOMICAL_IMPLICIT)[1])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.GENERIC_ANATOMICAL,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_ANATOMICAL)[0])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.GENERIC_ANATOMICAL,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_ANATOMICAL)[1])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.GENERIC_ANATOMICAL_IMPLICIT,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_ANATOMICAL_IMPLICIT)[0])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.GENERIC_ANATOMICAL_IMPLICIT,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_ANATOMICAL_IMPLICIT)[1])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.GENERIC_NON_ISOTROPIC,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_NON_ISOTROPIC)[0])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.GENERIC_COMPRESSED,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          False)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_COMPRESSED)[0])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.GENERIC_COMPRESSED,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.GENERIC_COMPRESSED)[1])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_dicom.dicom_series_to_nifti(test_data.HITACHI_ANATOMICAL,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          True)
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.HITACHI_ANATOMICAL)[1])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

        finally:
            shutil.rmtree(tmp_output_dir)

    def test_are_imaging_dicoms(self):
        assert convert_dicom.are_imaging_dicoms(read_dicom_directory(test_data.SIEMENS_ANATOMICAL))

    def test_get_vendor(self):
        assert convert_dicom._get_vendor(
            read_dicom_directory(test_data.SIEMENS_ANATOMICAL)) == convert_dicom.Vendor.SIEMENS
        assert convert_dicom._get_vendor(read_dicom_directory(test_data.GE_ANATOMICAL)) == convert_dicom.Vendor.GE
        assert convert_dicom._get_vendor(
            read_dicom_directory(test_data.PHILIPS_ANATOMICAL)) == convert_dicom.Vendor.PHILIPS
        assert convert_dicom._get_vendor(
            read_dicom_directory(test_data.PHILIPS_ENHANCED_ANATOMICAL)) == convert_dicom.Vendor.PHILIPS
        assert convert_dicom._get_vendor(
            read_dicom_directory(test_data.GENERIC_ANATOMICAL)) == convert_dicom.Vendor.GENERIC


class TestConversionGantryTilted(unittest.TestCase):
    def setUp(self):
        settings.disable_validate_orthogonal()
        settings.enable_resampling()
        settings.set_resample_padding(-1000)
        settings.set_resample_spline_interpolation_order(1)

    def tearDown(self):
        settings.disable_resampling()
        settings.enable_validate_orthogonal()

    def test_resampling(self):
        tmp_output_dir = tempfile.mkdtemp()
        try:

            results = convert_dicom.dicom_series_to_nifti(test_data.FAILING_ORHTOGONAL,
                                                          os.path.join(tmp_output_dir, 'test.nii.gz'),
                                                          False)
            self.assertTrue(os.path.isfile(results['NII_FILE']))

        finally:
            shutil.rmtree(tmp_output_dir)


def _count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(name)])


if __name__ == '__main__':
    unittest.main()
