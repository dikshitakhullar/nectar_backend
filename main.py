from datetime import datetime
import logging
from config import config
from firebase_admin import credentials, initialize_app
from firebase_operations.firebase_manager import FirebaseManager
from pinterest_utils import download_pinterest_image
from clip import get_clip_embeddings
from room_object_analysis import analyze_room_objects
from visualize_objects import draw_bounding_boxes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main script function for extracting and processing Pinterest images.
    """
    try:
        # Step 1: Provide Pinterest URL
        pinterest_url = input("Enter the Pinterest Pin URL: ").strip()
        # Step 2: Extract and Download Image
        image_path = "downloaded_image.jpg"
        logger.info("Downloading image...")
        download_result = download_pinterest_image(pinterest_url, image_path)
        logger.info("Image downloaded successfully.")
        get_clip_embeddings(image_path)
        detected_objects = analyze_room_objects(image_path)
        draw_bounding_boxes(image_path, detected_objects)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()