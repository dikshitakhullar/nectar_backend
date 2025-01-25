import asyncio
import logging
from config import config
from firebase_admin import credentials, initialize_app
from firebase_operations.firebase_manager import FirebaseManager
from stable_diffusion.img2img_service import StableDiffusionImg2Img
from services.similar_images_service import SimilarImagesService
from models.room_relationship import (
    RoomChange, ChangeType, MaterialType, ColorPalette
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# currently tests the similar images service -> generate similar image and upload the room
# and store the room relationship (parent id + similar room ids)
async def generate_similar_rooms(room_id: str):
    try:
        # Initialize Firebase if not already initialized
        # if not len(firebase_admin._apps):
        cred = credentials.Certificate(config.FIREBASE_CREDENTIALS)
        initialize_app(cred, {'storageBucket': config.FIREBASE_STORAGE_BUCKET})
        
        firebase_manager = FirebaseManager()
        sd_service = StableDiffusionImg2Img(
            api_key=config.STABLE_DIFFUSION_API_KEY
        )
        
        similar_service = SimilarImagesService(
            firebase_manager=firebase_manager,
            sd_service=sd_service
        )
        
        # Create proper RoomChange objects
        changes = [
            RoomChange(
                change_type=ChangeType.STYLE.value,
                to_value='industrial'
            ),
            RoomChange(
                change_type=ChangeType.MATERIAL.value,
                to_value='polished concrete',
                material_type=MaterialType.FLOOR
            )
        ]
        
        similar_rooms = await similar_service.generate_similar_images(
            room_id=room_id,
            changes=changes,
            strength=0.75
        )
        
        logger.info(f"Generated {len(similar_rooms)} similar rooms")
        for room in similar_rooms:
            logger.info(f"Room ID: {room.id} - {room.image_url}")
            
        return similar_rooms
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise

async def main():
    await generate_similar_rooms("261DVtfJCNh0gZJMLuHU")

if __name__ == "__main__":
    asyncio.run(main())