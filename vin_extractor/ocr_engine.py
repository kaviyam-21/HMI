from paddleocr import PaddleOCR
import logging
import os

# Suppress PaddleOCR logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_USE_MKLDNN'] = '0'
os.environ['PADDLE_DISABLE_DNNL'] = '1'
os.environ['MKLDNN_ENABLED'] = 'OFF'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
logging.getLogger("ppocr").setLevel(logging.ERROR)

class OCREngine:
    def __init__(self, use_gpu=False):
        # Initialize PaddleOCR with parameters optimized for small engraved text
        # det_db_unclip_ratio: helps expand the detection box to capture full character height
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            use_gpu=use_gpu, 
            show_log=False,
            det_db_thresh=0.3, # More sensitive detection
            det_db_box_thresh=0.5, # Keep boxes with lower confidence
            det_db_unclip_ratio=1.8 # Expansion ratio
        )

    def extract_text(self, image_np) -> str:
        """
        Extract text from image using PaddleOCR.
        Returns the combined string of high-confidence results.
        """
        result = self.ocr.ocr(image_np, cls=True)
        
        if not result or not result[0]:
            return ""
            
        full_text = []
        for line in result[0]:
            # result format: [ [[coords], (text, confidence)], ... ]
            text, confidence = line[1]
            if confidence > 0.3:
                # Keep only alphanumeric and uppercase for VIN consistency
                cleaned = "".join(c for c in text if c.isalnum()).upper()
                full_text.append(cleaned)
                
        return "".join(full_text)
