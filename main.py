from datetime import datetime
import logging
import asyncio
from config import config
from firebase_admin import credentials, initialize_app, storage
from firebase_operations.firebase_manager import FirebaseManager
from pinterest_utils import download_pinterest_image
from stable_diffusion.img2img_service import StableDiffusionImg2Img
from clip import get_clip_embeddings
from room_object_analysis import analyze_room_objects, create_room_from_analysis
from visualize_objects import draw_bounding_boxes
from services.similar_images_service import SimilarImagesService
from models.room import Room, RoomStyle, RoomType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def async_main():
    """
    Async main script function for extracting and processing Pinterest images.
    """
    cred = credentials.Certificate(config.FIREBASE_CREDENTIALS)
    app = initialize_app(cred, {'storageBucket': config.FIREBASE_STORAGE_BUCKET})
        
    firebase_manager = FirebaseManager()
    sd_service = StableDiffusionImg2Img(
        api_key=config.STABLE_DIFFUSION_API_KEY
    )
        
    similar_service = SimilarImagesService(
        firebase_manager=firebase_manager,
        sd_service=sd_service
    )

    try:
        # Step 1: Provide Pinterest URL
        pinterest_url = input("Enter the Pinterest Pin URL: ").strip()

        # Step 2: Extract and Download Image
        image_path = "downloaded_image.jpg"
        logger.info("Downloading image...")
        download_result = download_pinterest_image(pinterest_url, image_path)
        logger.info("Image downloaded successfully.")

        # Step 3: Analyze Room Objects
        room_style, room_type = get_clip_embeddings(image_path)
        detected_objects = analyze_room_objects(image_path)
        draw_bounding_boxes(image_path, detected_objects)
        room = create_room_from_analysis(detected_objects, room_style, room_type)
        print("Room: ", room)

        public_url = firebase_manager.upload_image(image_path)
        
        # Step 4: Generate similar images
        similar_rooms = await similar_service.generate_similar_images_from_pinterest(
            room=room,
            pinterest_url=public_url
        )
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

def main():
    """
    Entry point that runs the async main function.
    """
    asyncio.run(async_main())

if __name__ == "__main__":
    main()