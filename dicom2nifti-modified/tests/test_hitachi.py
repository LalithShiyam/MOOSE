# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""

import os
import shutil
import tempfile
import unittest

import dicom2nifti.common as common
import dicom2nifti.convert_hitachi as convert_hitachi
import nibabel
import tests.test_data as test_data
from dicom2nifti.common import read_dicom_directory
from tests.test_tools import assert_compare_nifti, ground_thruth_filenames


class TestConversionHitachi(unittest.TestCase):
    def test_anatomical(self):
        tmp_output_dir = tempfile.mkdtemp()
        try:
            results = convert_hitachi.dicom_to_nifti(read_dicom_directory(test_data.HITACHI_ANATOMICAL),
                                                     None)
            self.assertTrue(results.get('NII_FILE') is None)
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_hitachi.dicom_to_nifti(read_dicom_directory(test_data.HITACHI_ANATOMICAL),
                                                     os.path.join(tmp_output_dir, 'test.nii.gz'))
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.HITACHI_ANATOMICAL)[0])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))

            results = convert_hitachi.dicom_to_nifti(read_dicom_directory(test_data.HITACHI_ANATOMICAL_IMPLICIT),
                                                     os.path.join(tmp_output_dir, 'test.nii.gz'))
            assert_compare_nifti(results['NII_FILE'],
                                 ground_thruth_filenames(test_data.HITACHI_ANATOMICAL_IMPLICIT)[0])
            self.assertTrue(isinstance(results['NII'], nibabel.nifti1.Nifti1Image))
        finally:
            shutil.rmtree(tmp_output_dir)

    def test_is_hitachi(self):
        assert not common.is_hitachi(read_dicom_directory(test_data.SIEMENS_ANATOMICAL))
        assert not common.is_hitachi(read_dicom_directory(test_data.GE_ANATOMICAL))
        assert not common.is_hitachi(read_dicom_directory(test_data.PHILIPS_ANATOMICAL))
        assert not common.is_hitachi(read_dicom_directory(test_data.GENERIC_ANATOMICAL))
        assert common.is_hitachi(read_dicom_directory(test_data.HITACHI_ANATOMICAL))


if __name__ == '__main__':
    unittest.main()
