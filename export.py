from ultralytics import YOLO


import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="Path to the weight file")
    # parser.add_argument("--int8", action="store_true", help="Use int8 quantization")
    parser.add_argument("--half", action="store_true", help="Use half precision")

    args = parser.parse_args()
    model = YOLO(model=rf"{args.weights}", task="detect", verbose=True)
    
    model.export(
        format="onnx",
        imgsz = [960,960],
        simplify = True,
        data=r"F:\Project\FPSHelper\yolov58\data\COD.yaml",
        # int8=args.int8,
        half=args.half,
    )