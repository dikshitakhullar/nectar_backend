from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from models.room import Room, RoomStyle, RoomMetadata, MaterialFinish, RoomFurniture, FurniturePiece, FurnitureMaterial, RoomType, FixtureMaterials, Lighting, LightingType, FurnitureType, RoomType

def analyze_room_objects(image_path):
    # Load the object detection model (YOLO)
    object_detector = YOLO("yolov8x.pt")  # Use a lightweight YOLOv8 model
    # object_detector = YOLO('runs/detect/yolov8_home_decor8/weights/best.pt')  # Update with the correct path

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
        class_name = results[0].names[int(class_id)]  # Get the object name from YOLO
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
            "class_name": class_name,
            "attributes": best_description
        })

    # Return the detected objects and their attributes
    for obj in detected_objects:
        print(f"Object ({obj['class_name']}): {obj['bounding_box']}, "
              f"Confidence: {obj['confidence']:.2f}, "
              f"Attributes: {obj['attributes']}")

    return detected_objects


def create_room_from_analysis(detected_objects, room_style: RoomStyle, room_type: RoomType) -> Room:
    """
    Creates a basic Room object from the analysis results with detailed descriptions.
    
    Args:
        detected_objects (list): List of detected objects
        room_style (RoomStyle): Style of the room
        room_type (RoomType): Type of the room
    
    Returns:
        Room: A Room object with basic information
    """
    # Create detailed description based on detected objects
    description = f"A {room_style.value} {room_type.value.lower()} featuring: "
    
    # Create detailed descriptions for each object
    object_descriptions = []
    for obj in detected_objects:
        obj_desc = f"a {obj['attributes'].replace(' object in', '')} {obj['class_name']}"
        object_descriptions.append(obj_desc)
    
    description += ", ".join(object_descriptions)
    
    # Create the room object with minimal information
    room = Room(
        room_type=room_type,
        style=room_style,
        title=f"{str(room_style.value).title()} {room_type.value}",
        description=description,
        image_url="",  # This will be set later
        is_original=True
    )
    
    return room
