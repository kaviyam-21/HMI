import cv2
import numpy as np

class PreProcessor:
    @staticmethod
    def is_blurry(image: np.ndarray, threshold: float = 60.0) -> bool:
        """Adaptive blur detection using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Correct image tilt/skew using minAreaRect on the largest text contours."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def remove_glare(gray: np.ndarray) -> np.ndarray:
        """Use Morphological Hat transforms to isolate text from metallic glare."""
        struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, struct_elem)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, struct_elem)
        return cv2.add(cv2.subtract(gray, tophat), blackhat)

    @staticmethod
    def adjust_contrast_brightness(gray: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Dynamically adjust contrast (alpha) and brightness (beta).
        Formula: g(x) = alpha*f(x) + beta
        """
        return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    @staticmethod
    def edge_enhancement_sobel(gray: np.ndarray) -> np.ndarray:
        """Amplify engraved strokes using Sobel operators."""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        return cv2.convertScaleAbs(magnitude)

    @staticmethod
    def get_roi(image: np.ndarray, roi_config: dict) -> np.ndarray:
        h, w = image.shape[:2]
        return image[
            int(roi_config['ymin'] * h):int(roi_config['ymax'] * h),
            int(roi_config['xmin'] * w):int(roi_config['xmax'] * w)
        ]

    @staticmethod
    def strategy_engraved_sobel_adaptive(roi: np.ndarray) -> np.ndarray:
        """Strategy: Sobel Edge -> Adaptive Threshold (Best for deep engravings)."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edge = PreProcessor.edge_enhancement_sobel(gray)
        # Combine back for better context
        combined = cv2.addWeighted(gray, 0.7, edge, 0.3, 0)
        return cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    @staticmethod
    def strategy_gamma(roi: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Adjust gamma to highlight faint engravings."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.LUT(gray, table)

    @staticmethod
    def strategy_high_contrast_otsu(roi: np.ndarray) -> np.ndarray:
        """Strategy: Contrast Boost -> CLAHE -> Otsu (Best for flat low-contrast)."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        boosted = PreProcessor.adjust_contrast_brightness(gray, alpha=1.8, beta=-40)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(boosted)
        _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def detect_vin_candidates(image: np.ndarray) -> list[np.ndarray]:
        """Detect potential VIN regions using scale-aware edge detection."""
        h_orig, w_orig = image.shape[:2]
        
    @staticmethod
    def strategy_thinning(roi: np.ndarray) -> np.ndarray:
        """Strategy: Thin strokes to prevent character merging (O vs C, B vs 8)."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Erase some boundary pixels to separate letters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thinned = cv2.erode(thresh, kernel, iterations=1)
        return thinned

    @staticmethod
    def detect_vin_candidates(image: np.ndarray) -> list[np.ndarray]:
        """Detect potential VIN regions using scale-aware edge detection."""
        h_orig, w_orig = image.shape[:2]
        
        target_w = 2000 
        if w_orig < target_w:
            scale = target_w / w_orig
            # Switch to LANCZOS4 for higher quality upscaling
            image = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_LANCZOS4)
            img_h, img_w = image.shape[:2]
        else:
            img_h, img_w = h_orig, w_orig

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        k_size = max(3, int(img_w / 300))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        close_w = int(img_w / 15)
        close_h = int(img_h / 40)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, max(1, close_h)))
        
        thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        
        cnts, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h) if h > 0 else 0
            area_ratio = (w * h) / float(img_w * img_h)
            
            if 3.5 < aspect_ratio < 25.0 and 0.001 < area_ratio < 0.25:
                pad = int(h * 0.2)
                y1, y2 = max(0, y - pad), min(img_h, y + h + pad)
                x1, x2 = max(0, x - pad), min(img_w, x + w + pad)
                candidates.append(image[y1:y2, x1:x2])
        
        if not candidates:
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if 4.0 < (w/h) < 20.0 and 0.001 < (w*h)/(img_w*img_h) < 0.3:
                    candidates.append(image[y:y+h, x:x+w])

        candidates.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
        return candidates

    @staticmethod
    def strategy_shadow_normalization(roi: np.ndarray) -> np.ndarray:
        """Strategy: Local Normalization -> Deep Shadow Enhancement."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Calculate local mean and std dev
        mean = cv2.GaussianBlur(gray, (21, 21), 0)
        std = cv2.GaussianBlur(np.sqrt(np.square(gray - mean)), (21, 21), 0)
        # Normalize: (x - mean) / std
        normalized = cv2.divide(gray - mean, std + 1e-5, scale=128) + 128
        return cv2.convertScaleAbs(np.clip(normalized, 0, 255))

    @staticmethod
    def strategy_unsharp_mask(roi: np.ndarray) -> np.ndarray:
        """Strategy: Unsharp masking to boost high-frequency stroke edges."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gaussian_3 = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray, 2.5, gaussian_3, -1.5, 0)
        return unsharp

    @staticmethod
    def strategy_extreme_contrast(roi: np.ndarray) -> np.ndarray:
        """Strategy: High-gain CLAHE followed by iterative denoising."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4)).apply(gray)
        denoised = cv2.fastNlMeansDenoising(clahe, None, 10, 7, 21)
        return denoised

    @staticmethod
    def strategy_character_deepening(roi: np.ndarray) -> np.ndarray:
        """Strategy: Sharpen the internal strokes of engravings using Laplacian of Gaussian."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Calculate Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian_8u = cv2.convertScaleAbs(laplacian)
        # Sharpen: Original - weight * Laplacian
        deepened = cv2.addWeighted(gray, 1.5, laplacian_8u, -0.5, 0)
        return deepened

    @staticmethod
    def strategy_thinning(roi: np.ndarray) -> np.ndarray:
        """Strategy: Thin strokes to prevent character merging (O vs C, B vs 8)."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thinned = cv2.erode(thresh, kernel, iterations=1)
        return thinned

    @staticmethod
    def strategy_high_pass(roi: np.ndarray) -> np.ndarray:
        """Strategy: High-pass filter to isolate engraved stroke textures."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        high_pass = cv2.subtract(gray, blurred)
        return cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def get_all_variants(roi: np.ndarray) -> list:
        """Generator for different preprocessing variants tailored for metal engravings."""
        roi_clean = PreProcessor.deskew(roi)
        gray = cv2.cvtColor(roi_clean, cv2.COLOR_BGR2GRAY)
        
        # 1. High-frequency detail boost
        unsharp = PreProcessor.strategy_unsharp_mask(roi_clean)
        
        variants = [
            roi_clean,
            gray,
            unsharp,
            PreProcessor.strategy_character_deepening(roi_clean),
            PreProcessor.strategy_shadow_normalization(roi_clean),
            PreProcessor.strategy_high_pass(roi_clean),
        ]
        
        # 2. Add lighting robust variants
        for gamma in [0.5, 0.8, 1.2, 1.8]:
            variants.append(PreProcessor.strategy_gamma(roi_clean, gamma=gamma))
            
        # 3. Dedicated contrast-level binary variants
        for alpha in [1.3, 1.8, 2.3]:
            # Apply some denoising before contrast boost
            denoised = cv2.medianBlur(gray, 3)
            boosted = PreProcessor.adjust_contrast_brightness(denoised, alpha=alpha, beta=-30)
            
            # Small grid CLAHE for high-frequency detail
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(boosted)
            _, t_otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(t_otsu)
            
            # 4. Thinning/Dilation variants
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            variants.append(cv2.erode(t_otsu, kernel, iterations=1))
            variants.append(cv2.dilate(t_otsu, kernel, iterations=1))

        return variants
