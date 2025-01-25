from datetime import datetime
import logging
from config import config
from firebase_admin import credentials, initialize_app
from models.room import *
from firebase_operations.firebase_manager import FirebaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# currently only tests uploading a room to firebase (image and metadata)
class RoomUploader:

    def __init__(self):
        cred = credentials.Certificate(config.FIREBASE_CREDENTIALS)
        initialize_app(cred, {'storageBucket': config.FIREBASE_STORAGE_BUCKET})
        self.firebase = FirebaseManager()
        # initialize.initialize_firebase()

    def upload_room(self, image_path: str, room: Room) -> tuple[str, str]:
        try:
            doc_id, image_url = self.firebase.upload_room(
                image_path=image_path,
                metadata=room.model_dump(exclude_none=True)
            )
            logger.info(f"Uploaded room {room.title} with ID: {doc_id}")
            return doc_id, image_url
        except Exception as e:
            logger.error(f"Error uploading room: {str(e)}")
            raise

def main():
    uploader = RoomUploader()
    # Example usage
    # Create room metadata (hardcoded for this example)
    # room_data = Room(
    #     id="2",
    #     metadata=RoomMetadata(
    #     tags=["rustic", "cozy", "exposed beams", "leather", "earth tones"],
    #     ceiling_height=10.0,
    #     color_palette=[ColorPalette.EARTH_TONES, ColorPalette.WARM_NEUTRALS],
    #     lighting=Lighting(
    #         type=LightingType.MIXED,
    #         intensity="warm",
    #         natural_light="south-facing",
    #         fixtures=["recessed lights", "floor lamps"],
    #         features=["large windows", "accent lighting"]
    #     ),
    #     floor=MaterialFinish(
    #         material="hardwood",
    #         color="walnut",
    #         texture="hand-scraped",
    #         pattern="wide plank",
    #         finish="matte"
    #     ),
    #     walls=MaterialFinish(
    #         material="plaster",
    #         color="warm beige",
    #         texture="smooth",
    #         pattern=None,
    #         finish="eggshell"
    #     ),
    #     ceiling=MaterialFinish(
    #         material="wood",
    #         color="walnut",
    #         texture="exposed beams",
    #         pattern="parallel beams",
    #         finish="matte"
    #     ),
    #     furniture=RoomFurniture(
    #         pieces=[
    #             FurniturePiece(
    #                 furniture_type=FurnitureType.SOFA,
    #                 material=FurnitureMaterial(
    #                     material_type="leather",
    #                     color="cognac",
    #                     finish="distressed",
    #                     texture="soft",
    #                     pattern=None
    #                 ),
    #                 quantity=1,
    #                 notes="3-seater Chesterfield style"
    #             ),
    #             FurniturePiece(
    #                 furniture_type=FurnitureType.ARM_CHAIR,
    #                 material=FurnitureMaterial(
    #                     material_type="velvet",
    #                     color="forest green",
    #                     finish=None,
    #                     texture="plush",
    #                     pattern=None
    #                 ),
    #                 quantity=2,
    #                 notes="wingback design"
    #             )
    #         ]
    #     ),
    #     fixtures=FixtureMaterials(
    #         ceiling_lights=None,
    #         wall_lights=None,
    #         floor_lamps="brass floor lamp",
    #         faucets=None,
    #         sink=None,
    #         door_handles=None,
    #         cabinet_hardware=None,
    #         chandeliers=None,
    #         pendant_lights=None,
    #         track_lighting=None,
    #         sconces="antique brass"
    #     ),
    #     windows=[
    #         Window(
    #             type=WindowType.PICTURE,
    #             count=2,
    #             treatment=WindowTreatment.DRAPES,
    #             treatment_material="linen",
    #             treatment_color="natural"
    #         )
    #     ],
    # ),
    # room_type=RoomType.LIVING_ROOM,
    # style=RoomStyle.RUSTIC,
    # title="Rustic Modern Living Room",
    # description="A warm, inviting living room that blends rustic elements with modern comfort",
    # image_url="",
    # generation_prompt="Create a warm and inviting living room interior featuring exposed wooden ceiling beams in walnut finish. The room has walnut hardwood flooring with wide planks and a hand-scraped texture. The walls are painted in warm beige with an eggshell finish. The focal point is a cognac leather Chesterfield sofa and two forest green velvet wingback chairs. Two large picture windows are dressed with natural linen drapes. The room is lit by brass floor lamps and wall sconces, creating a cozy atmosphere. The style combines rustic elements with modern comfort, emphasizing natural materials and earth tones.",
    # is_original=True,  
    # timestamp=datetime.now(),
    # )
    room_data = Room(
    id="123abc",
    room_type=RoomType.DINING_ROOM,
    metadata=RoomMetadata(
        ceiling_height=12.0,
        tags=["formal dining", "classical", "elegant", "symmetrical", "traditional", "ornate moldings"],
        color_palette=[ColorPalette.WARM_NEUTRALS],
        lighting=Lighting(
            type=LightingType.MIXED,
            intensity="moderate",
            natural_light=None,
            fixtures=["wall sconces", "table lamps"],
            features=["ornate chandelier with crystal details"]
        ),
        floor=MaterialFinish(
            material="marble",
            color="beige",
            texture="polished",
            pattern="subtle veining",
            finish="glossy"
        ),
        walls=MaterialFinish(
            material="plaster",
            color="off-white",
            texture="smooth",
            pattern="panel moldings",
            finish="matte"
        ),
        ceiling=MaterialFinish(
            material="plaster",
            color="white",
            texture="ornate",
            pattern="decorative moldings and medallion",
            finish="matte"
        ),
        furniture=RoomFurniture(
            pieces=[
                FurniturePiece(
                    furniture_type=FurnitureType.DINING_TABLE,
                    material=FurnitureMaterial(
                        material_type="wood",
                        color="black",
                        finish="lacquered",
                        texture="smooth",
                        pattern="round pedestal base"
                    ),
                    quantity=1,
                    notes="Large round pedestal dining table"
                ),
                FurniturePiece(
                    furniture_type=FurnitureType.DINING_CHAIR,
                    material=FurnitureMaterial(
                        material_type="upholstered wood",
                        color="black and cream",
                        finish="lacquered wood",
                        texture="fabric upholstery",
                        pattern="curved back design"
                    ),
                    quantity=6,
                    notes="Upholstered armchairs with brass feet"
                ),
                FurniturePiece(
                    furniture_type=FurnitureType.SIDEBOARD,
                    material=FurnitureMaterial(
                        material_type="wood",
                        color="black",
                        finish="lacquered",
                        texture="smooth",
                        pattern=None
                    ),
                    quantity=1,
                    notes="Console table with decorative hardware"
                )
            ]
        ),
        fixtures=FixtureMaterials(
            ceiling_lights="crystal and brass chandelier",
            wall_lights="brass sconces",
            floor_lamps=None,
            faucets=None,
            sink=None,
            door_handles=None,
            cabinet_hardware="brass",
            chandeliers=None,
            pendant_lights=None,
            track_lighting=None,
            sconces=None
        ),
        windows=[],
    ),
    title="Neoclassical Formal Dining Room",
    description="An elegant formal dining room featuring classical architectural details including ornate crown moldings, panel walls, and a decorative ceiling medallion. The space is anchored by a round black pedestal dining table surrounded by six upholstered chairs in black and cream. A crystal chandelier and brass wall sconces provide warm lighting.",
    style=RoomStyle.NEOCLASSICAL,
    image_url="",
    generation_prompt=None,
    is_original=True,  
    timestamp=datetime.now(),
    )
    try:
        doc_id, url = uploader.upload_room("neo_classical_dining.jpg", room_data)
        logger.info(f"Success - Doc ID: {doc_id}, URL: {url}")
    except Exception as e:
        logger.error(f"Failed: {str(e)}")

if __name__ == "__main__":
    main()
