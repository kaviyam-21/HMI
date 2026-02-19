import cv2
import numpy as np
import logging
from .preprocessing import PreProcessor
from .ocr_engine import OCREngine
from .validator import VINValidator
from .config import DEFAULT_ROI, BLUR_THRESHOLD

class VINExtractionPipeline:
    def __init__(self, use_gpu=False):
        self.ocr_engine = OCREngine(use_gpu=use_gpu)
        self.logger = logging.getLogger(__name__)

    def process_image(self, image_path: str, custom_roi: dict = None) -> str:
        """Adaptive pipeline with dynamic region detection."""
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Could not read image path."

        # 1. Image Quality Check (Done on original resolution)
        gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
        
        if variance < 20.0:
            return "Image too blurry"

        # 2. Image Pre-normalization (Upscale after blur check)
        h_orig, w_orig = image.shape[:2]
        target_w = 1600
        if w_orig < target_w:
            scale = target_w / w_orig
            image = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_LANCZOS4)

        # 3. Dynamic Region Localization
        region_candidates = PreProcessor.detect_vin_candidates(image)
        
        # Add the manual ROI as a fallback
        roi_config = custom_roi if custom_roi else DEFAULT_ROI
        manual_roi = PreProcessor.get_roi(image, roi_config)
        region_candidates.append(manual_roi)

        # 4. Process each candidate region
        best_format_match = None
        
        for region in region_candidates:
            variants = PreProcessor.get_all_variants(region)
            for variant in variants:
                raw_text = self.ocr_engine.extract_text(variant)
                is_valid, final_vin, is_checksum = VINValidator.process_and_validate(raw_text)
                
                if is_checksum:
                    return f"VIN Detected: {final_vin}"
                
                if is_valid and best_format_match is None:
                    best_format_match = final_vin

        # 5. Final fallback
        full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raw_full = self.ocr_engine.extract_text(full_gray)
        is_valid, final_vin, is_checksum = VINValidator.process_and_validate(raw_full)
        
        if is_checksum:
            return f"VIN Detected: {final_vin}"
            
        if is_valid and best_format_match is None:
            best_format_match = final_vin

        if best_format_match:
            return f"VIN Detected: {best_format_match}"

        return "Invalid VIN"
