# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""

import os
import shutil
import sys
import tempfile
import unittest

import tests.test_data as test_data


class TestConversionDicom(unittest.TestCase):
    def test_main_function(self):
        tmp_output_dir = tempfile.mkdtemp()
        script_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'scripts',
                                   'dicom2nifti')
        assert os.path.isfile(script_file)

        try:
            if sys.version_info > (3, 0):
                from importlib.machinery import SourceFileLoader
                dicom2nifti_module = SourceFileLoader("dicom2nifti_script", script_file).load_module()
            else:
                import imp
                dicom2nifti_module = imp.load_source('dicom2nifti_script', script_file)
            dicom2nifti_module.main([test_data.SIEMENS_ANATOMICAL, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii.gz"))

        finally:
            shutil.rmtree(tmp_output_dir)

    def test_gantry_option(self):
        tmp_output_dir = tempfile.mkdtemp()
        script_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'scripts',
                                   'dicom2nifti')
        assert os.path.isfile(script_file)

        try:
            if sys.version_info > (3, 0):
                from importlib.machinery import SourceFileLoader
                dicom2nifti_module = SourceFileLoader("dicom2nifti_script", script_file).load_module()
            else:
                import imp
                dicom2nifti_module = imp.load_source('dicom2nifti_script', script_file)
            dicom2nifti_module.main(['-G', test_data.FAILING_ORHTOGONAL, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii.gz"))
            dicom2nifti_module.main(['--allow-gantry-tilting', test_data.FAILING_ORHTOGONAL, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii.gz"))

        finally:
            shutil.rmtree(tmp_output_dir)

    def test_gantry_resampling(self):
        tmp_output_dir = tempfile.mkdtemp()
        script_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'scripts',
                                   'dicom2nifti')
        assert os.path.isfile(script_file)

        try:
            if sys.version_info > (3, 0):
                from importlib.machinery import SourceFileLoader
                dicom2nifti_module = SourceFileLoader("dicom2nifti_script", script_file).load_module()
            else:
                import imp
                dicom2nifti_module = imp.load_source('dicom2nifti_script', script_file)
            dicom2nifti_module.main(
                ['-G', '-r', '-o', '1', '-p', '-1000', test_data.FAILING_ORHTOGONAL, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii.gz"))
            dicom2nifti_module.main(['--allow-gantry-tilting',
                                     '--resample',
                                     '--resample-order', '1',
                                     '--resample-padding', '-1000',
                                     test_data.FAILING_ORHTOGONAL,
                                     tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii.gz"))

        finally:
            shutil.rmtree(tmp_output_dir)

    def test_multiframe_option(self):
        tmp_output_dir = tempfile.mkdtemp()
        script_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'scripts',
                                   'dicom2nifti')
        assert os.path.isfile(script_file)

        try:
            if sys.version_info > (3, 0):
                from importlib.machinery import SourceFileLoader
                dicom2nifti_module = SourceFileLoader("dicom2nifti_script", script_file).load_module()
            else:
                import imp
                dicom2nifti_module = imp.load_source('dicom2nifti_script', script_file)
            dicom2nifti_module.main(['-M', test_data.PHILIPS_ENHANCED_ANATOMICAL_IMPLICIT, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "301_dicom2nifti.nii.gz"))
            dicom2nifti_module.main(
                ['--allow-multiframe-implicit', test_data.PHILIPS_ENHANCED_ANATOMICAL_IMPLICIT, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "301_dicom2nifti.nii.gz"))

        finally:
            shutil.rmtree(tmp_output_dir)

    def test_compression_function(self):
        tmp_output_dir = tempfile.mkdtemp()
        script_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'scripts',
                                   'dicom2nifti')
        assert os.path.isfile(script_file)

        try:
            if sys.version_info > (3, 0):
                from importlib.machinery import SourceFileLoader
                dicom2nifti_module = SourceFileLoader("dicom2nifti_script", script_file).load_module()
            else:
                import imp
                dicom2nifti_module = imp.load_source('dicom2nifti_script', script_file)
            dicom2nifti_module.main(['-C', test_data.SIEMENS_ANATOMICAL, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii"))
            dicom2nifti_module.main(['--no-compression', test_data.SIEMENS_ANATOMICAL, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii"))

        finally:
            shutil.rmtree(tmp_output_dir)

    def test_reorientation_function(self):
        tmp_output_dir = tempfile.mkdtemp()
        script_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'scripts',
                                   'dicom2nifti')
        assert os.path.isfile(script_file)

        try:
            if sys.version_info > (3, 0):
                from importlib.machinery import SourceFileLoader
                dicom2nifti_module = SourceFileLoader("dicom2nifti_script", script_file).load_module()
            else:
                import imp
                dicom2nifti_module = imp.load_source('dicom2nifti_script', script_file)
            dicom2nifti_module.main(['-R', test_data.SIEMENS_ANATOMICAL, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii.gz"))
            dicom2nifti_module.main(['--no-reorientation', test_data.SIEMENS_ANATOMICAL, tmp_output_dir])
            assert os.path.isfile(os.path.join(tmp_output_dir, "4_dicom2nifti.nii.gz"))

        finally:
            shutil.rmtree(tmp_output_dir)


if __name__ == '__main__':
    unittest.main()
