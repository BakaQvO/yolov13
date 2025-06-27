from ultralytics import YOLO
import sys
import argparse
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--exsist_ok", action="store_true", help="Overwrite existing training results")
    parser.add_argument("--cfg", type=str, default=None, help="Path to the model configuration file")

    args = parser.parse_args()
    
    if args.resume:
        model = YOLO(r"F:\Project\FPSHelper\yolov12\COD\2025-05-30_yolo11l-AttnP2_960_150\weights\last.pt")
        model.train(
            resume=True,
            workers=16,
        )
        
        exit(0)
    
    
    
    # model = YOLO(model=r"F:\Project\FPSHelper\yolov12\yolov12l.yaml",task="detect",verbose=True)
    # model = YOLO(model=r"yolov12s.pt",task="detect",verbose=True)
    model_size = "s"
    model_cfg = args.cfg or f"yolo11{model_size}-AttnP2"
    model = YOLO(
        model=rf"F:\Project\FPSHelper\yolov12\{model_cfg}.yaml",
        # model=rf"F:\Project\FPSHelper\yolov12\ultralytics\cfg\models\v8\yolov8{model_size}-p6.yaml",
        task="detect",
        verbose=True
    )
    
    model = model.load(
        rf"F:\Project\FPSHelper\yolov12\COD\2025-05-30_yolo11s-AttnP2_960_100\weights\best.pt"
    )
    
    detect = model.model.model[-1]
    print("stride  :", detect.stride)   # 预期 tensor([4., 8., 16.])
    print("anchors :", detect.anchors)  # 预期 tensor([], size=(0,))

    epochs = 150
    imgsz = 960
    
    model.train(
        # model=r"F:\Project\FPSHelper\yolov12\COD\pre1500_2025-05-21_l_640\weights\best.pt",
        # model = r"yolov12s.pt",
        # data=r"F:\Project\FPSHelper\yolov12\ultralytics\cfg\datasets\coco8.yaml",
        # data=r"F:\Project\FPSHelper\yolov12\Datasets-COD-pre1500.yaml",
        data=r"F:\Project\FPSHelper\yolov12\Datasets-COD-2025-05-25.yaml",
        # cfg=r"F:\Project\FPSHelper\yolov12\config.yaml",
        epochs = epochs,
        
        # batch = -1,
        # batch = 9, # s
        batch = 6, # l
        # batch = 2, 
        imgsz = imgsz,
        workers = 12,
        cache = "disk",
        # amp = False,
        
        # optimizer = "Adamax",
        optimizer = "SGD",
        # optimizer = "AdamW",
        
        lr0 = 0.004,
        lrf = 0.1,
        
        warmup_epochs = 5,
        warmup_bias_lr = 0.01,
        
        box = 6.0,
        obj = 0.3,
        cls = 1.0,
        
        hsv_h = 0.08,
        hsv_s = 0.8,
        hsv_v = 0.5,
        perspective = 0.0005,
        # copy_paste = 0.3,
        scale = 0.5,
        shear = 10.0,
        
        auto_augment = "augmix",
        
        
        name=f"2025-05-30_{model_cfg}_{imgsz}_{epochs}",
        exist_ok = args.exsist_ok,
    )