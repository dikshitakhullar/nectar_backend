from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from models.room import RoomStyle, RoomType


def get_clip_embeddings(image_path):

    # Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define room types with styles as candidate text descriptions
    room_types = [
        "modern living room",
        "imperial living room",
        "rustic living room",
        "minimalist dining room",
        "vintage dining room",
        "modern bedroom",
        "industrial bedroom",
        "traditional kitchen",
        "modern kitchen",
        "rustic bathroom",
        "modern bathroom",
        "luxury study room",
        "industrial study room",
        "modern outdoor patio",
        "rustic outdoor patio",
        "vintage hallway",
        "modern hallway",
        "traditional garden",
        "modern garden",
        "luxury home office",
        "modern home office",
        "rustic library",
        "luxury library",
        "modern nursery",
        "vintage nursery",
        "rustic bedroom",
        "luxury bedroom"
    ]

    # Compute text embeddings for room types
    text_inputs = processor(text=room_types, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

    image = Image.open(image_path)
    image_inputs = processor(images=image, return_tensors="pt")

    # Compute image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

    # Find the closest room type and style to the image
    similarities = torch.matmul(image_features, text_features.T)
    best_match_idx = torch.argmax(similarities, dim=1)
    best_match = room_types[best_match_idx.item()]

    # Split into style and room type
    parts = best_match.split()
    style = parts[0]  # First word is the style
    room_type = " ".join(parts[1:])  # Rest is the room type

    # Try to map to enums
    try:
        room_style = RoomStyle(style.lower())
    except ValueError:
        # Default to modern if style not found in enum
        room_style = RoomStyle.MODERN

    try:
        # Remove spaces and convert to uppercase for enum matching
        room_type_key = room_type.replace(" ", "_").upper()
        room_type = RoomType[room_type_key]
    except KeyError:
        # Default to living room if type not found in enum
        room_type_enum = RoomType.LIVING_ROOM

    print(f"Detected Style: {room_style.value}, Room Type: {room_type.value}")
    return room_style, room_type