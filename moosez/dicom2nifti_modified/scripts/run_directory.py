# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""

import logging

import dicom2nifti.convert_dir as convert_directory


def main():
    print('TEST')

    convert_directory.convert_directory("/Users/abrys/Downloads/Agfa",
                                        "/Users/abrys/Downloads/Agfa")


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s %(levelname)-8s %(message)s')
    main()

    # unittest.main()
