#!/usr/bin/env bash
# update the api docs
sphinx-apidoc -f -o ../rst ../../dicom2nifti
rm -Rf ../../../dicom2nifti-docs/*
sphinx-build -b html ../rst ../../../dicom2nifti-docs/
touch ../../../dicom2nifti-docs/.nojekyll