from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


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
        "traditional media room",
        "modern media room",
        "rustic guest room",
        "luxury guest room"
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
    best_room_type = room_types[best_match_idx.item()]

    print("Best Matching Room Type and Style:", best_room_type)
