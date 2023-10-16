import argparse
from datetime import datetime
import math
import nibabel as nib
import numpy as np
import os
import pydicom
import sys
import tempfile
import logging
import shutil
import json

# The machine ID is platform-dependent
if os.name == "nt":
    import wmi

    c = wmi.WMI()
    csProduct = c.Win32_ComputerSystemProduct()[0]
    machineID = csProduct.UUID
else:
    with open("/etc/machine-id") as file:
        machineID = file.readline().strip()


def CreateSegmentation(img, seed, siuid: str, cropping: bool):
    # Build dataset from Nifti img and DICOM seed
    ds = pydicom.Dataset()

    AssertMatchingInput(img, seed)
    CreatePatientModule(seed, ds)
    CreateGeneralStudyModule(seed, ds)
    CreatePatientStudyModule(seed, ds)
    CreateGeneralSeriesModule(seed, ds, siuid)
    CreateSegmentationSeriesModule(ds)
    CreateFrameOfReferenceModule(seed, ds)
    CreateGeneralEquipmentModule(ds)
    CreateImagePixelModule(img, ds, seed, cropping)
    CreateSopCommonModule(seed, ds)
    CreateSegmentationImageModule(ds)
    CreateMultiFrameFunctionGroupsModule(img, ds, seed)
    CreateMultiFrameDimensionModule(ds)
    # private tags
    CreateRegionVisibilityModule(seed, ds)

    data = img.get_fdata()

    # ensure that we only have values 0 and 1
    data[data < 0] = 0
    data[data > 0] = 1

    # shape is (x,y,z) where z changes most rapidly but we want
    # shape (z,y,x) where x changes most rapidly, so order 'F'
    (x_dim, y_dim, z_dim) = data.shape
    array = data.flatten().reshape(z_dim, y_dim, x_dim, order="F")

    # Flip Y axis
    array = np.flip(array, axis=1)

    # pack binary values
    ds.PixelData = pydicom.pixel_data_handlers.numpy_handler.pack_bits(array)

    return ds


def AssertMatchingInput(img, seed):
    # The SEG file should have same image orientation, image position, pixel size,
    # slice spacing, and matrix size (x,y,z) as the CT seed.
    data = img.get_fdata()
    error = []
    # if data.shape[0] != seed.Columns:
    #     error.append("Mismatch in Columns")
    # if data.shape[1] != seed.Rows:
    #     error.append("Mismatch in Rows")

    (mat, vec) = nib.affines.to_matvec(img.affine)
    dir_x = mat[:, 0]
    dir_y = mat[:, 1]
    dir_z = mat[:, 2]
    len_x = np.linalg.norm(dir_x)
    len_y = np.linalg.norm(dir_y)
    len_z = np.linalg.norm(dir_z)
    (dy, dx) = seed.PixelSpacing

    tolerance = 0.000001
    if abs(dx - len_x) > tolerance:
        error.append(f"Mismatch in PixelSpacing x")
    if abs(dy - len_y) > tolerance:
        error.append(f"Mismatch in PixelSPacing y")

    voxelSize = img.header.get_zooms()
    if abs(len_x - voxelSize[0]) > tolerance:
        error.append(f"Mismatch in voxel size x")
    if abs(len_y - voxelSize[1]) > tolerance:
        error.append(f"Mismatch in voxel size y")
    if abs(len_z - voxelSize[2]) > tolerance:
        error.append(f"Mismatch in voxel size z")

    # Nifti directions, normalize, flip X axis to compare with DICOM
    v1 = img.affine[:, 0]
    v2 = img.affine[:, 1]
    v1 = -v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    for i in range(3):
        if abs(v1[i] - seed.ImageOrientationPatient[i]) > tolerance:
            error.append("Mismatch in image orientation")
            break
        if abs(v2[i] - seed.ImageOrientationPatient[i + 3]) > tolerance:
            error.append("Mismatch in image orientation")
            break

    # Only transverse DICOM
    expectedOrientation = [1, 0, 0, 0, 1, 0]
    if not all(
        math.isclose(a, b, abs_tol=tolerance)
        for a, b in zip(seed.ImageOrientationPatient, expectedOrientation)
    ):
        error.append(R"Image orientation is not 1\0\0\0\1\0")

    if len(error) > 0:
        print("Error:", error)
        sys.exit("The Nifti segmentation is not related to the DICOM CT")
    return


def CreateCodeSequence(value, scheme, meaning):
    item = pydicom.Dataset()
    item.CodeValue = value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return [item]


def CreateFrameContentSequence(u):
    item = pydicom.Dataset()
    item.DimensionIndexValues = u
    return [item]


def CreateFrameOfReferenceModule(seed, ds):
    ds.FrameOfReferenceUID = seed.FrameOfReferenceUID
    ds.PositionReferenceIndicator = None


def CreateGeneralEquipmentModule(ds):
    manufacturer = "Hermes Medical Solutions AB"
    modelName = "MOOSE"
    softwareVersion = "nifti2dicomseg.py-1.0.0"
    # These are required in Enhanced General Equipment Module
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = modelName
    ds.DeviceSerialNumber = GetUuid()
    ds.SoftwareVersions = softwareVersion


def CreateGeneralSeriesModule(seed, ds, siuid):
    seriesDescription = "Segmentation: " + str(seed.SeriesDescription)
    ds.SeriesDescription = seriesDescription
    ds.SeriesInstanceUID = siuid
    ds.SeriesNumber = None
    now = datetime.now()
    ds.SeriesDate = now.date()
    ds.SeriesTime = now.time()


def CreateGeneralStudyModule(seed, ds):
    ds.StudyInstanceUID = seed.StudyInstanceUID
    ds.StudyDate = seed.StudyDate
    ds.StudyTime = seed.StudyTime
    ds.ReferringPhysicianName = seed.ReferringPhysicianName
    ds.StudyID = seed.StudyID
    ds.AccessionNumber = seed.AccessionNumber
    ds.StudyDescription = seed.StudyDescription


def CreateImagePixelModule(img, ds, seed, cropping):
    if cropping:
        ds.Columns = img.shape[0]
        ds.Rows = img.shape[1]
    else:
        ds.Columns = seed.Columns
        ds.Rows = seed.Rows


def CreateMultiFrameDimensionModule(ds):
    dimensionOrganizationUid = pydicom.uid.generate_uid()

    item = pydicom.Dataset()
    item.DimensionOrganizationUID = dimensionOrganizationUid
    ds.DimensionOrganizationSequence = [item]

    item = pydicom.Dataset()
    item.DimensionOrganizationUID = dimensionOrganizationUid
    item.DimensionIndexPointer = pydicom.tag.Tag("ImagePositionPatient")
    item.FunctionalGroupPointer = pydicom.tag.Tag("PlanePositionSequence")
    ds.DimensionIndexSequence = [item]


def readJsonCIELabValues(segment_label: str):
    with open("CIELabValues.json", "r") as json_file:
        display_data = json.load(json_file)

    # Extract the labels dictionary from the JSON data
    cielab_values = display_data.get("DisplayValues", {})
    return cielab_values[segment_label]


def CreateMultiFrameFunctionGroupsModule(img, ds, seed):
    ds.InstanceNumber = 1
    now = datetime.now()
    ds.ContentDate = now.date()
    ds.ContentTime = now.time()
    ds.NumberOfFrames = img.shape[2]  # size Z

    # Segment Sequence
    item = pydicom.Dataset()
    segmentNumber = 1
    item.SegmentNumber = 1
    item.SegmentLabel = GetSegmentLabel(img)
    item.SegmentAlgorithmType = "AUTOMATIC"  # when not MANUAL, set algorithm name
    item.SegmentAlgorithmName = "nnUNET"
    # convert RGA to CIELab or leave it to get any default color.
    item.RecommendedDisplayCIELabValue = readJsonCIELabValues(item.SegmentLabel)

    # These should perhaps get different values for different segmented organs
    item.SegmentedPropertyCategoryCodeSequence = CreateCodeSequence(
        "M-01000", "SRT", "Morphologically Altered Structure"
    )
    item.SegmentedPropertyTypeCodeSequence = CreateCodeSequence(
        "M-03000", "SRT", "Mass"
    )
    ds.SegmentSequence = [item]

    # Content Identification Macro Attributes
    ds.ContentLabel = "VOI"
    ds.ContentDescription = None
    ds.ContentCreatorName = None

    # Shared Functional Groups
    voxelSize = img.header.get_zooms()

    pixelMeasuresSequence = CreatePixelMeasuresSequence(voxelSize)

    sharedItem = pydicom.Dataset()
    sharedItem.PlaneOrientationSequence = CreatePlaneOrientationSequence(seed)
    sharedItem.PixelMeasuresSequence = pixelMeasuresSequence
    sharedItem.SegmentIdentificationSequence = CreateSegmentIdentificationSequence(
        segmentNumber
    )
    ds.SharedFunctionalGroupsSequence = [sharedItem]

    # Per-frame Functional Groups
    v1 = img.affine[:, 0]
    v2 = img.affine[:, 1]
    v3 = img.affine[:, 2]
    v4 = img.affine[:, 3]
    new_x = -v4[0]
    new_y = -(v4 + v2 * (img.shape[1] - 1))[1]
    new_z = v4[2]
    imagePosition = [new_x, new_y, new_z]

    dir_z = v3[:3]

    list = []
    for i in range(ds.NumberOfFrames):
        framePosition = imagePosition + dir_z * i
        item = pydicom.Dataset()
        item.PlanePositionSequence = CreatePlanePositionSequence(framePosition)
        item.FrameContentSequence = CreateFrameContentSequence(i + 1)
        list.append(item)

    ds.PerFrameFunctionalGroupsSequence = list


def CreatePatientModule(seed, ds):
    ds.PatientName = seed.PatientName
    ds.PatientID = seed.PatientID
    ds.PatientBirthDate = seed.PatientBirthDate
    ds.PatientSex = seed.PatientSex


def CreatePatientStudyModule(seed, ds):
    if "PatientSize" in seed:
        ds.PatientSize = seed.PatientSize
    if "PatientWeight" in seed:
        ds.PatientWeight = seed.PatientWeight
    if "PatientAge" in seed:
        ds.PatientAge = seed.PatientAge


def CreatePixelMeasuresSequence(voxelSize):
    # Convert and round to avoid implicit conversion when creating
    # the DS attribute, which looses the float32 rounding.
    pixelSpacingX = round(np.float64(voxelSize[0]), 6)
    pixelSpacingY = round(np.float64(voxelSize[1]), 6)
    sliceThickness = round(np.float64(voxelSize[2]), 6)
    # Row spacing and column spacing in the order defined in DICOM.
    pixelSpacing = [pixelSpacingY, pixelSpacingX]
    item = pydicom.Dataset()
    item.SliceThickness = sliceThickness
    item.PixelSpacing = pixelSpacing
    return [item]


def CreatePlaneOrientationSequence(seed):
    item = pydicom.Dataset()
    item.ImageOrientationPatient = seed.ImageOrientationPatient
    return [item]


def CreatePlanePositionSequence(vector):
    vector = [round(vector[0], 6), round(vector[1], 6), round(vector[2], 6)]
    item = pydicom.Dataset()
    item.ImagePositionPatient = vector
    return [item]


def CreateReferencedSeries(ds):
    pass  # Maybe


def CreateRegionVisibilityModule(seed, ds):
    item = pydicom.Dataset()
    item.SeriesInstanceUID = seed.SeriesInstanceUID
    # Add the private HMS Region Visibility Sequence
    block = ds.private_block(0x7017, "HMS", create=True)
    block.add_new(0x11, "SQ", [item])


def CreateSegmentIdentificationSequence(id):
    item = pydicom.Dataset()
    item.ReferencedSegmentNumber = id
    return [item]


def CreateSegmentationImageModule(ds):
    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0  # unsigned
    ds.BitsAllocated = 1
    ds.BitsStored = 1
    ds.HighBit = 0
    ds.LossyImageCompression = "00"
    ds.SegmentationType = "BINARY"


def CreateSegmentationSeriesModule(ds):
    ds.Modality = "SEG"
    ds.SeriesNumber = 1


def CreateSopCommonModule(seed, ds):
    ds.SOPClassUID = pydicom.uid.SegmentationStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    now = datetime.now()
    ds.InstanceCreationDate = now.date()
    ds.InstanceCreationTime = now.time()
    # timezoneOffsetFromUtc? Should consider also for SeriesDate/Time and ContentDate/Time


def GetSegmentLabel(img):
    filename = os.path.basename(img.get_filename())
    filename = os.path.splitext(filename)[0]
    filename = os.path.splitext(filename)[0]
    return filename


def GetUuid():
    return machineID


def convert_nifti_to_dicom_seg(nifti_path, dicom_seed_path, output_path, cropping=False):
    # Create DICOM conformant date and time
    pydicom.config.datetime_conversion = True
    seriesinstanceUID = pydicom.uid.generate_uid()

    seed = pydicom.read_file(dicom_seed_path)

    if os.path.isdir(nifti_path):
        nifti_list = [
            os.path.join(nifti_path, file)
            for file in os.listdir(nifti_path)
        ]
    elif os.path.isfile(nifti_path):
        nifti_list = [nifti_path]
    else:
        sys.exit("No such file: " + nifti_path)

    if not os.path.isfile(dicom_seed_path):
        sys.exit("No such file: " + dicom_seed_path)

    if os.path.isdir(output_path):
        pass
    elif os.path.isfile(output_path):
        pass  # overwrite
    else:
        # a file ends with ".dcm"
        if output_path.endswith(".dcm"):
            dir = os.path.dirname(output_path)
        else:
            dir = output_path

        # create the output dir (or parent dir of the output file)
        if len(dir) > 0 and not os.path.isdir(dir):
            os.makedirs(dir)

    if len(nifti_list) > 1 and output_path.endswith(".dcm"):
        sys.exit("Incompatible args, multiple nifti inputs requires output to dir")

    temp_directory = tempfile.mkdtemp()
    for nifti in nifti_list:
        print(nifti)

        img = nib.load(nifti)

        ds = CreateSegmentation(img, seed, seriesinstanceUID, cropping)
        ds.ensure_file_meta()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        output_file = temp_directory

        if os.path.isdir(output_file):
            output_file = os.path.join(output_file, ds.SOPInstanceUID + ".dcm")

        ds.save_as(output_file, False)

    shutil.copytree(temp_directory, output_path, dirs_exist_ok=True)
    shutil.rmtree(temp_directory)
