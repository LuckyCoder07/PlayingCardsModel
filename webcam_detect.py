import cv2
import time
from pathlib import Path
from ultralytics import YOLO

def run_webcam(
    weights_path: str = "runs/detect/cards_detector/weights/best.pt",
    conf_threshold: float = 0.5,
    camera_index: int = 0
):
    current_dir = Path(__file__).parent.resolve()
    model_file = current_dir / weights_path

    # Fallback to pretrained model or last.pt if best.pt is not found
    if not model_file.exists():
        fallback_model = current_dir / "yolo11n.pt"
        print(f"[WARN] Trained weights '{model_file}' not found. Falling back to base model '{fallback_model}'...")
        model = YOLO(str(fallback_model) if fallback_model.exists() else "yolo11n.pt")
    else:
        print(f"[INFO] Loading trained weights from: {model_file}")
        model = YOLO(str(model_file))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        return

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    print("\n--- Card Detection Running ---")
    print("Controls:")
    print("  'q' : Quit application")
    print("  '+' / '=' : Increase confidence threshold by 0.05")
    print("  '-' : Decrease confidence threshold by 0.05")
    print("  's' : Save current frame snapshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        # Inference
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Extract detected cards
        detected_names = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            card_name = model.names[cls_id]
            detected_names.append(card_name)

        # Draw HUD info overlay
        hud_text = f"FPS: {fps:.1f} | Conf: {conf_threshold:.2f} | Detected: {len(detected_names)} cards"
        cv2.rectangle(annotated_frame, (10, 10), (550, 50), (20, 20, 20), -1)
        cv2.putText(
            annotated_frame,
            hud_text,
            (20, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 127),
            2,
            cv2.LINE_AA,
        )

        if detected_names:
            cards_summary = ", ".join(detected_names[:8])
            if len(detected_names) > 8:
                cards_summary += f" +{len(detected_names)-8} more"
            cv2.putText(
                annotated_frame,
                f"Cards: {cards_summary}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Playing Card Real-Time Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            conf_threshold = min(0.95, round(conf_threshold + 0.05, 2))
            print(f"Confidence threshold set to: {conf_threshold}")
        elif key == ord("-"):
            conf_threshold = max(0.10, round(conf_threshold - 0.05, 2))
            print(f"Confidence threshold set to: {conf_threshold}")
        elif key == ord("s"):
            snapshot_path = current_dir / f"card_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(snapshot_path), annotated_frame)
            print(f"[INFO] Snapshot saved to: {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time Playing Card Detection via Webcam")
    parser.add_argument("--weights", type=str, default="runs/detect/cards_detector/weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    run_webcam(weights_path=args.weights, conf_threshold=args.conf, camera_index=args.camera)
