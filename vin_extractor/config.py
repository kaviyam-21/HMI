import os

# Region of Interest (ROI) configuration
# Format: [ymin, xmin, ymax, xmax] as fractions of image size (0.0 to 1.0)
# Defaulting to middle section of the image
DEFAULT_ROI = {
    "ymin": 0.3,
    "xmin": 0.1,
    "ymax": 0.7,
    "xmax": 0.9
}

# OCR Configuration
PADLLE_OCR_LANG = 'en'
RECOGNITION_CONFIDENCE_THRESHOLD = 0.5

# Blur Detection Threshold
# Laplacian variance below this value is considered too blurry
BLUR_THRESHOLD = 100.0

# VIN Validation
VIN_REGEX = r"^[A-HJ-NPR-Z0-9]{17}$"

# Character corrections for prohibited characters in VINs (ISO 3779)
# I, O, Q are never allowed and are common OCR errors for 1, 0, 0
OCR_CORRECTION_MAP = {
    'O': '0',
    'I': '1',
    'Q': '0'
}
