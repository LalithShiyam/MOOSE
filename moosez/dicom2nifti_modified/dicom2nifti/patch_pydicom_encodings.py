"""
Importing this file will extend the pydicom encoding mapping fixing issues when reading less frequently used encodings
"""

try:
    import pydicom.charset as pydicom_charset
except ImportError:
    import dicom.charset as pydicom_charset


def apply():
    pydicom_charset.python_encoding.update({
        '': 'iso8859',  # default character set for DICOM
        'ISO_IR 6': 'iso8859',  # alias for latin_1 too (iso_ir_6 exists as an alias to 'ascii')
        'ISO_IR 13': 'shift_jis',
        'ISO_IR 58': 'GB2312',
        'ISO_IR 87': 'iso2022_jp',
        'ISO_IR 100': 'latin_1',  # these also have iso_ir_1XX aliases in python 2.7
        'ISO_IR 101': 'iso8859_2',
        'ISO_IR 109': 'iso8859_3',
        'ISO_IR 110': 'iso8859_4',
        'ISO_IR 126': 'iso_ir_126',  # Greek
        'ISO_IR 127': 'iso_ir_127',  # Arabic
        'ISO_IR 138': 'iso_ir_138',  # Hebrew
        'ISO_IR 144': 'iso_ir_144',  # Russian
        'ISO_IR 148': 'iso_ir_148',  # Turkish
        'ISO_IR 149': 'euc_kr',  # needs cleanup via clean_escseq from valuerep
        'ISO_IR 159': 'iso-2022-jp',
        'ISO_IR 166': 'iso_ir_166',  # Thai
        'ISO_IR 192': 'UTF8',  # from Chinese example, 2008 PS3.5 Annex J p1-4

        'ISO 2022 IR 6': 'iso8859',  # alias for latin_1 too
        'ISO 2022 IR 13': 'shift_jis',
        'ISO 2022 IR 58': 'GB2312',
        'ISO 2022 IR 87': 'iso2022_jp',
        'ISO 2022 IR 100': 'latin_1',
        'ISO 2022 IR 101': 'iso8859_2',
        'ISO 2022 IR 109': 'iso8859_3',
        'ISO 2022 IR 110': 'iso8859_4',
        'ISO 2022 IR 126': 'iso_ir_126',
        'ISO 2022 IR 127': 'iso_ir_127',
        'ISO 2022 IR 138': 'iso_ir_138',
        'ISO 2022 IR 144': 'iso_ir_144',
        'ISO 2022 IR 148': 'iso_ir_148',
        'ISO 2022 IR 149': 'euc_kr',  # needs cleanup via clean_escseq()
        'ISO 2022 IR 159': 'iso-2022-jp',
        'ISO 2022 IR 166': 'iso_ir_166',
        'ISO 2022 IR 192': 'UTF8',  # from Chinese example, 2008 PS3.5 Annex J p1-4

        'GB18030': 'GB18030',
        'ISO 2022 GBK': 'GBK',  # from DICOM correction CP1234
        'ISO 2022 58': 'GB2312',  # from DICOM correction CP1234
        'GBK': 'GBK',  # from DICOM correction CP1234
    })
