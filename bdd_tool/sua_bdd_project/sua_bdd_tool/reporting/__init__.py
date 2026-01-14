from .pdf_generator import *

EXPORTER_MAP = {
    0: PDFExporterBasic,
    1: PDFExporterDetailed,
    2: PDFExporterMeasurement,
    3: PDFExporterCompact,
    32: PDFExporterCompactAuxImage,
    4: PDFExporterWithContext,
    42: PDFExporterWithContextAuxImage,
}