import argparse
import logging
import os
from typing import Any

import cv2
import supervision as sv
from tqdm import tqdm  # type: ignore
from yaya_tools.helpers.dataset import load_directory_images_annotatations  # type: ignore

from model_face.detector.yolov8_face_detector import YOLOv8FaceDetection
from model_face.helpers.transformations import blur_box  # type: ignore

logger = logging.getLogger(__name__)


def anonymize_file(detector: YOLOv8FaceDetection, image_name: str, source_directory: str) -> tuple[str, bool]:
    """Threded Resize image"""
    try:
        source_path = os.path.join(source_directory, image_name)
        image = cv2.imread(source_path)
        if image is None:
            logger.error(f"Could not read image {image_name}")
            return image_name, False

        detections = detector.detect(image)

        # Logging : How many faces detected
        if detections == sv.Detections.empty():
            logger.info("No faces on %s image.", image_name)
        else:
            logger.info(
                "Detected %u faces inside %s image. Average conf %2.2f.",
                len(detections),
                image_name,
                detections.confidence.mean(),  # type: ignore
            )

        # Anonymize
        anonymized_im = image
        for xyxy in detections.xyxy:
            anonymized_im = blur_box(image=anonymized_im, box=xyxy)

        #  Save : Only if inplace is set and found faces
        if detections != sv.Detections.empty():
            cv2.imwrite(source_path, anonymized_im)
            logger.info(f"Saved anonymized image to {source_path}")

        return image_name, True

    except Exception as e:
        logger.error(f"Error anonymizing image {image_name}: {e}")
        return image_name, False


def multiprocess_anonymize(
    detector: YOLOv8FaceDetection,
    source_directory: str,
    images_names: list[str],
    pool_size: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Anonymizes images using a single loop instead of multiprocessing,
    since the detector object cannot be pickled.
    """
    sucess_files: list[str] = []
    failed_files: list[str] = []
    for image_name in tqdm(images_names, desc="Anonymizing images"):
        result = anonymize_file(detector, image_name, source_directory)
        if result[1]:
            sucess_files.append(image_name)
        else:
            failed_files.append(image_name)

    return sucess_files, failed_files


def configure_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main_anonymize() -> None:
    """Main function to run the detection."""
    # Configure logging
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="images/", help="Directory of images to anonymize in place")
    parser.add_argument(
        "--modelpath",
        type=str,
        default="zoo/yolov8n-face.onnx",
        help="onnx filepath",
    )
    parser.add_argument("--confThreshold", default=0.20, type=float, help="class confidence")
    parser.add_argument("--nmsThreshold", default=0.40, type=float, help="nms iou thresh")
    parser.add_argument("--pixelate", action="store_true", help="pixelate the face instead of blurring")
    parser.add_argument("--padding", default=5, type=int, help="Bounding box padding size in px")
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    images_annotated: dict[str, Any] = load_directory_images_annotatations(args.images)

    # Detector : Create
    detector = YOLOv8FaceDetection(
        args.modelpath,
        conf_thres=args.confThreshold,
        iou_thres=args.nmsThreshold,
        padding=args.padding,
    )

    # Multiprocess : Anonymize
    logger.info("Anonymizing images...")
    sucess_files, failed_files = multiprocess_anonymize(
        detector=detector,
        source_directory=args.images,
        images_names=list(images_annotated.keys()),
        pool_size=5,
    )
    logger.info("Anonymization completed.")


if __name__ == "__main__":
    main_anonymize()
