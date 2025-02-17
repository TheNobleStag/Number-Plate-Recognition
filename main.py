import os
import torch
import logging
import contextlib
import numpy as np
from PIL import Image
from transformers import pipeline
from paddleocr import PaddleOCR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.getLogger("ppocr").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"

@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield

def initialize_models():
    checkpoint = "google/owlv2-base-patch16-ensemble"
    with suppress_print():
        detector = pipeline(
            model=checkpoint,
            task="zero-shot-object-detection",
            device=device,
            use_fast=True
        )
    with suppress_print():
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return detector, ocr

def process_image(image_path, detector, ocr):
    original_image = Image.open(image_path)

    with suppress_print():
        prediction = detector(
            original_image,
            candidate_labels=["license plate"],
        )[0]

    box = prediction["box"]
    cropped_image = original_image.crop(list(box.values()))

    cropped_numpy_image = np.array(cropped_image)
    cropped_numpy_image_rgb = cropped_numpy_image[:, :, ::-1].copy()

    with suppress_print():
        result = ocr.ocr(cropped_numpy_image_rgb, cls=True)

    if result:
        detected_text = sorted(result[0], key=lambda x: x[0][2][1])
        
        license_plate_number = " ".join([text[1][0] for text in detected_text])
    else:
        license_plate_number = "UNKNOWN"

    return license_plate_number

if __name__ == "__main__":
    detector, ocr = initialize_models()

    while True:
        image_path = input("Enter the path to the image (or 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break
        try:
            license_plate_number = process_image(image_path, detector, ocr)
            print(f"{license_plate_number}")
        except Exception as e:
            print(f"Error: {e}")