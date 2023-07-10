# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""


class ConversionValidationError(Exception):
    """
    Custom error type to distinguish between know validations and script errors
    """

    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(ConversionValidationError, self).__init__(message)


class ConversionError(Exception):
    """
    Custom error type to distinguish between know validations and script errors
    """

    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(ConversionError, self).__init__(message)
