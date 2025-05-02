import argparse
import logging

import cv2
import supervision as sv

from model_face.detector.yolov8_face_detector import YOLOv8FaceDetection
from model_face.helpers.transformations import blur_box, pixelate_box  # type: ignore

logger = logging.getLogger(__name__)


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
    source = cv2.imread(args.imgpath)

    # Detect Objects
    detector = YOLOv8FaceDetection(
        args.modelpath,
        conf_thres=args.confThreshold,
        iou_thres=args.nmsThreshold,
        padding=args.padding,
    )
    detections = detector.detect(source)

    # Logging : How many faces detected
    if detections == sv.Detections.empty():
        logger.warning("No faces detected.")
    else:
        logger.info(f"Detected {len(detections)} faces.")

    # Anonymize
    anonymized_im = source.copy()
    for xyxy in detections.xyxy:
        anonymized_im = (
            pixelate_box(image=anonymized_im, box=xyxy) if args.pixelate else blur_box(image=anonymized_im, box=xyxy)
        )

    #  Save : Only if inplace is set and found faces
    if detections != sv.Detections.empty():
        cv2.imwrite(args.imgpath, anonymized_im)
        logger.info(f"Saved anonymized image to {args.imgpath}")


if __name__ == "__main__":
    main_anonymize()
