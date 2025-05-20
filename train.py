from ultralytics import YOLO

model = YOLO(model=r"F:\Project\FPSHelper\yolov12\FPS.yaml",task="detect",verbose=True)

model.train(
    data=r"F:\Project\FPSHelper\yolov12\FPS.yaml",
    cfg=r"F:\Project\FPSHelper\yolov12\config.yaml",
    
)