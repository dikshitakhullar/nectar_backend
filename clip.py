from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


def get_clip_embeddings():

    # Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load candidate text descriptions from the file
    with open("home_decor_descriptions.txt", "r") as file:
        text_descriptions = file.read().splitlines()

    # Precompute and save text embeddings
    precomputed_file = "text_features.pt"
    try:
        # Attempt to load precomputed embeddings
        text_descriptions, text_features = torch.load(precomputed_file)
        print("Loaded precomputed embeddings.")
    except FileNotFoundError:
        print("Precomputed embeddings not found. Computing now...")
        batch_size = 100  # Process descriptions in batches for memory efficiency
        text_features_list = []
        for i in range(0, len(text_descriptions), batch_size):
            batch = text_descriptions[i:i+batch_size]
            text_inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                batch_features = model.get_text_features(**text_inputs)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)  # Normalize
            text_features_list.append(batch_features)

        # Combine all batches
        text_features = torch.cat(text_features_list, dim=0)

        # Save precomputed embeddings
        torch.save((text_descriptions, text_features), precomputed_file)
        print("Precomputed embeddings saved.")

    # Load the image and preprocess it
    image = Image.open(save_path)
    image_inputs = processor(images=image, return_tensors="pt")

    # Compute image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

    # Find the closest text descriptions to the image
    similarities = torch.matmul(image_features, text_features.T)
    best_match_idx = torch.argmax(similarities, dim=1)
    best_description = text_descriptions[best_match_idx.item()]

    print("Best Matching Description:", best_description)