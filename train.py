from ultralytics import YOLO

model = YOLO('yolov8m.pt')

def train(filepath, model = YOLO('yolov8m.pt')):

    results = model.train(
        data = filepath,
        epochs=50,                # Number of training epochs
        imgsz=640,                # Image size
        batch=32,                 # Batch size
        name="castle_test",      # Custom name for the training run
        device='cpu',
        workers=12
    )

if __name__ == '__main__':
    train('the yaml filepath here plz (MAKE IT A RAWSTRING)') #WHEN YOU RUN THIS IT MAKES A FILE OF THE RUNS AND THEN YOU USE IT LATER OKEY?

