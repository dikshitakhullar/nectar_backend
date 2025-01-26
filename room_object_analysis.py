from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from ultralytics import YOLO

def analyze_room_objects(image_path):
    # Load the object detection model (YOLO)
    object_detector = YOLO("yolov8x.pt")  # Use a lightweight YOLOv8 model

    # Load the style and texture classification model (CLIP)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define styles, textures/materials, and colors for objects
    styles = ["modern", "imperial", "rustic", "minimalist", "industrial", "vintage", "luxury"]
    materials = ["wood", "metal", "fabric", "glass", "plastic", "marble", "ceramic"]
    colors = ["red", "blue", "green", "yellow", "black", "white", "gray", "beige", "brown"]

    # Load and process the room image
    image = Image.open(image_path)

    # Detect objects in the image using YOLO
    results = object_detector(image_path)
    objects = results[0].boxes.data.cpu().numpy()  # Extract bounding boxes and class information

    # Initialize results
    detected_objects = []

    # Process each detected object
    for obj in objects:
        # Extract bounding box and crop the object from the image
        x1, y1, x2, y2, conf, class_id = obj
        cropped_object = image.crop((x1, y1, x2, y2))

        # Generate text descriptions for style, material, and color
        descriptions = [
            f"{style} {material} object in {color} color"
            for style in styles
            for material in materials
            for color in colors
        ]

        # Compute text embeddings for the descriptions
        text_inputs = processor(text=descriptions, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

        # Compute image embeddings for the cropped object
        object_inputs = processor(images=cropped_object, return_tensors="pt")
        with torch.no_grad():
            object_features = model.get_image_features(**object_inputs)
        object_features /= object_features.norm(dim=-1, keepdim=True)  # Normalize

        # Find the best matching style, material, and color
        similarities = torch.matmul(object_features, text_features.T)
        best_match_idx = torch.argmax(similarities, dim=1)
        best_description = descriptions[best_match_idx.item()]

        # Store the detected object and its attributes
        detected_objects.append({
            "bounding_box": (x1, y1, x2, y2),
            "confidence": conf,
            "class_id": class_id,
            "attributes": best_description
        })

    # Return the detected objects and their attributes
    for obj in detected_objects:
        print(f"Object: {obj['bounding_box']}, Confidence: {obj['confidence']:.2f}, Attributes: {obj['attributes']}")

    return detected_objects


