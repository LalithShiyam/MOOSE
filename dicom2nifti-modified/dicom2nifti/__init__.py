# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""

import dicom2nifti.patch_pydicom_encodings as patch_pydicom_encodings

patch_pydicom_encodings.apply()
