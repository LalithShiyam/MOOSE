# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""

import os
import shutil
import tempfile
import unittest

import dicom2nifti.convert_dir as convert_directory
import tests.test_data as test_data


class TestConversionDirectory(unittest.TestCase):

    def test_convert_directory(self):

        tmp_output_dir = tempfile.mkdtemp()
        try:
            convert_directory.convert_directory(test_data.GENERIC_ANATOMICAL, tmp_output_dir)
            assert os.path.isfile(os.path.join(tmp_output_dir, '4_dicom2nifti.nii.gz'))

        finally:
            shutil.rmtree(tmp_output_dir)

    def test_remove_accents(self):

        assert convert_directory._remove_accents(u'êén_ölîfānt@') == 'een_olifant'
        assert convert_directory._remove_accents(')(*&^%$#@!][{}\\"|,./?><') == ''


if __name__ == '__main__':
    unittest.main()
