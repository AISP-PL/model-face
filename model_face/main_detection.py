import argparse
import logging

import cv2
import supervision as sv

from model_face.detector.yolov8_face_detector import YOLOv8FaceDetection

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main_detect() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default="images/1.jpg", help="image path")
    parser.add_argument(
        "--modelpath",
        type=str,
        default="zoo/yolov8n-face.onnx",
        help="onnx filepath",
    )
    parser.add_argument("--confThreshold", default=0.20, type=float, help="class confidence")
    parser.add_argument("--nmsThreshold", default=0.40, type=float, help="nms iou thresh")
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    source = cv2.imread(args.imgpath)

    # Detect Objects
    detector = YOLOv8FaceDetection(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    detections = detector.detect(source)

    # Draw detections
    annotated = source.copy()
    annotator = sv.BoxAnnotator()
    scene = annotator.annotate(scene=annotated, detections=detections)
    cv2.imshow("YOLOv8 Face Detection", scene)
    cv2.waitKey(0)


if __name__ == "__main__":
    main_detect()
