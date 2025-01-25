import asyncio
import logging
from config import config as app_config
from firebase_admin import credentials, initialize_app
from firebase_operations.firebase_manager import FirebaseManager
from stable_diffusion.img2img_service import StableDiffusionImg2Img, Img2ImgConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# currently tests the img2img service
async def test_api():
    try:
        # Initialize Firebase
        cred = credentials.Certificate(app_config.FIREBASE_CREDENTIALS)
        initialize_app(cred, {'storageBucket': app_config.FIREBASE_STORAGE_BUCKET})
        
        # Get a real image URL from Firebase
        firebase_manager = FirebaseManager()
        room_id = "2UCLU71rZc3NEPWvb4qj"  # Replace with your room ID
        room = await firebase_manager.get_room(room_id)
        
        if not room:
            raise ValueError("Room not found")
            
        logger.info(f"Using image URL: {room.image_url}")

        # Initialize SD service
        sd_service = StableDiffusionImg2Img(
            api_key=app_config.STABLE_DIFFUSION_API_KEY
        )

        # Create test config with the real image URL
        test_config = Img2ImgConfig(
            init_image=room.image_url,
            prompt="change the flooring of this room, use marble instead",
            negative_prompt=StableDiffusionImg2Img.default_negative_prompt()
        )

        # Try to generate images
        logger.info("Attempting to generate images...")
        images = await sd_service.generate_similar_images(test_config)
        
        logger.info(f"Successfully generated {len(images)} images:")
        for i, url in enumerate(images, 1):
            logger.info(f"Image {i}: {url}")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    asyncio.run(test_api())