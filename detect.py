from ultralytics import YOLO
import sys
import argparse
import os

'''
python detect.py --path "F:\\TrainingData\\Human\\COD\\20250523-PVP\\images\\Desktop 2025.05.23 - 21.25.06.09_5681.973750832498_m.jpg" --weights "F:\\Project\\FPSHelper\\yolov12\\COD\\2025-05-23_yolo11l-AttnP2_1280_100\\weights\\best.pt" --show

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Save training results")
    parser.add_argument("--weights", default=r"F:\\Project\\FPSHelper\\yolov12\\COD\\2025-05-21_pre1500_yolo11l_1280_300\\weights\\best.onnx", type=str, help="Path to the weight file")
    parser.add_argument("--path", default=r"G:\\Videos\\Call of Duty  Black Ops 6\\PVP\\Desktop 2025.04.08 - 21.49.00.03~1.mp4", type=str, help="Path of input")
    parser.add_argument("--half", action="store_true", help="Use half precision")
    
    args = parser.parse_args()
    
    
    model = YOLO(args.weights)
    source = args.path
    
    results = model.predict(
        source=source,
        conf=0.25,
        iou=0.45,
        save=True,
        show=args.show,
        imgsz=[1280,1280],
        half=args.half,
        device="cuda:0",
        stream=False,
        project="output",
        name=rf"{args.weights.rsplit(r'\\weights',1)[0].rsplit(r'\\\\',1)[-1]}_o",
        exist_ok=True,
    )
    # for result in results:
    #     print(result.boxes)