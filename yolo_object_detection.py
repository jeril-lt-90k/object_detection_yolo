import torch
import multiprocessing

from ultralytics import YOLO


if __name__ == "__main__":

    # Ensure the multiprocessing start method is set to 'spawn' to avoid issues with certain platforms like macOS
    multiprocessing.set_start_method('spawn', force = True)

    # Load the pre-trained YOLO model from a local file (YOLOv11 in this case)
    model = YOLO("./yolo11n.pt")

    # Train the model (ensure you have the correct path to the dataset YAML)
    train_results = model.train(
        data = "./coco8.yaml",   # Path to the dataset YAML file
        epochs = 5,  # Number of training epochs
        imgsz = 160,  # Image size used for training (resize all images to this size)
        device = "cuda",  # Use GPU for training; use "cpu" if GPU is unavailable
    )

    # Evaluate the trained model on the validation set and get performance metrics
    metrics = model.val()
    print("Validation complete. Metrics:", metrics)

    # Perform object detection on a test image
    results = model("./images/<test_image>.jpg")

    # Show the image with bounding boxes and labels for detected objects
    results[0].show()

    # Save the trained model weights to a file ('yolo_model.pt')
    torch.save(model.state_dict(), './yolo_model.pt')