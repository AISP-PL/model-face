import argparse
import logging

import cv2
import supervision as sv

from model_face.detector.yolov8_face_detector import YOLOv8FaceDetection
from model_face.helpers.transformations import blur_box, pixelate_box  # type: ignore

logger = logging.getLogger(__name__)


def main_detect() -> None:
    """Main function to run the detection."""
    # Basic logging config to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgpath", type=str, default="images/1.jpg", help="image path"
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        default="zoo/yolov8n-face.onnx",
        help="onnx filepath",
    )
    parser.add_argument(
        "--confThreshold", default=0.20, type=float, help="class confidence"
    )
    parser.add_argument(
        "--nmsThreshold", default=0.40, type=float, help="nms iou thresh"
    )
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    source = cv2.imread(args.imgpath)

    # Detect Objects
    detector = YOLOv8FaceDetection(
        args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold
    )
    detections = detector.detect(source)

    # Draw detections
    annotated = source.copy()
    annotator = sv.BoxAnnotator()
    scene = annotator.annotate(scene=annotated, detections=detections)
    cv2.imshow("YOLOv8 Face Detection", scene)
    cv2.waitKey(0)


def main_anonymize() -> None:
    """Main function to run the detection."""
    # Basic logging config to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgpath", type=str, default="images/1.jpg", help="image path"
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        default="zoo/yolov8n-face.onnx",
        help="onnx filepath",
    )
    parser.add_argument(
        "--confThreshold", default=0.20, type=float, help="class confidence"
    )
    parser.add_argument(
        "--nmsThreshold", default=0.40, type=float, help="nms iou thresh"
    )
    parser.add_argument(
        "--pixelate", action="store_true", help="pixelate the face instead of blurring"
    )
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    source = cv2.imread(args.imgpath)

    # Detect Objects
    detector = YOLOv8FaceDetection(
        args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold
    )
    detections = detector.detect(source)

    # Anonymize
    scene = source.copy()
    for xyxy in detections.xyxy:
        scene = (
            pixelate_box(image=scene, box=xyxy)
            if args.pixelate
            else blur_box(image=scene, box=xyxy)
        )

    cv2.imshow("YOLOv8 Face Detection", scene)
    cv2.waitKey(0)


if __name__ == "__main__":
    main_detect()
