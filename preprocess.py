"""
preprocess.py
Cleans up noisy side-scan sonar images before they go into YOLOv8.
Two steps: (1) remove speckle noise, (2) boost contrast so faint
debris/anomaly shapes stand out more.
"""

import cv2
import numpy as np


def enhance_sonar_image(image: np.ndarray) -> np.ndarray:
    """
    Takes a raw sonar image (as a numpy array, e.g. loaded with cv2.imread)
    and returns a cleaned-up version ready for YOLO detection.
    """

    # 1. Convert to grayscale if the image is in color (sonar is usually
    #    single-channel anyway, but uploaded files may be RGB).
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Denoise (median blur handles the "speckle" noise typical of sonar).
    denoised = cv2.medianBlur(gray, 5)

    # 3. Contrast enhancement using CLAHE (Contrast Limited Adaptive
    #    Histogram Equalization) — makes faint objects/shadows pop out
    #    without blowing out bright areas.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. Convert back to 3-channel BGR so YOLOv8 (which expects color
    #    images) can process it normally.
    output = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return output


if __name__ == "__main__":
    # Quick manual test: run `python preprocess.py sample.jpg`
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Could not read image:", sys.argv[1])
        sys.exit(1)

    result = enhance_sonar_image(img)
    cv2.imwrite("enhanced_output.jpg", result)
    print("Saved enhanced_output.jpg")
