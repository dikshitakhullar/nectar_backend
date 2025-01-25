from datetime import datetime
import logging
from config import config
from firebase_admin import credentials, initialize_app
from firebase_operations.firebase_manager import FirebaseManager
from pinterest_utils import download_pinterest_image
from clip import get_clip_embeddings

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
        save_path = "downloaded_image.jpg"
        logger.info("Downloading image...")
        download_result = download_pinterest_image(pinterest_url, save_path)

        if download_result:
            logger.info("Image downloaded successfully.")
            get_clip_embeddings(save_path)
        else:
            logger.error("Failed to download image.")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()