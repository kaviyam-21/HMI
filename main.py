import os
# Required for PaddlePaddle stability on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_USE_MKLDNN'] = '0'
os.environ['PADDLE_DISABLE_DNNL'] = '1'

import sys
import argparse
import logging
from vin_extractor.pipeline import VINExtractionPipeline

# Configure minimal logging for production
logging.basicConfig(level=logging.ERROR, format='%(message)s')

def main():
    parser = argparse.ArgumentParser(description="Industrial VIN Extraction System")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR if available")
    
    args = parser.parse_args()

    try:
        pipeline = VINExtractionPipeline(use_gpu=args.gpu)
        result = pipeline.process_image(args.image_path)
        print(result)
    except Exception:
        print("Invalid VIN")

if __name__ == "__main__":
    main()
