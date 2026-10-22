import os
from pathlib import Path
from ultralytics import YOLO
import torch

def train_model(
    model_variant: str = "yolo11n.pt",
    data_yaml: str = "data.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "cards_detector",
):
    # Determine best compute device (mps on Mac Apple Silicon, cuda on GPU, or cpu)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "0"
    else:
        device = "cpu"
    print(f"Using compute device: {device}")

    current_dir = Path(__file__).parent.resolve()
    data_path = current_dir / data_yaml

    # Initialize base model
    model = YOLO(model_variant)

    print(f"Starting training for {epochs} epochs with {model_variant} on {data_path}...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(current_dir / project),
        name=name,
        patience=15,
        save=True,
        plots=True,
    )
    print("Training finished successfully!")
    print(f"Best model weights saved at: {current_dir / project / name / 'weights' / 'best.pt'}")
    return results

if __name__ == "__main__":
    train_model()
