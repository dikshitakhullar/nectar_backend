from typing import List, Optional, Dict
from models.room import Room
from firebase_operations.firebase_manager import FirebaseManager
from stable_diffusion.img2img_service import StableDiffusionImg2Img, Img2ImgConfig
from stable_diffusion.text2img_service import StableDiffusionText2Img, Text2ImgConfig
from models.room_relationship import RoomChange, ChangeType
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimilarImagesService:
    # Room-specific enhancers as a class attribute
    ROOM_SPECIFIC_ENHANCERS: Dict[str, List[str]] = {
        'Kitchen': [
            "professional kitchen layout",
            "premium appliances",
            "functional workspace",
            "efficient kitchen design",
            "modern appliances",
            "elegant cabinetry",
            "high-end countertops",
            "task lighting",
            "kitchen island",
            "cooking space"
        ],
        'Living Room': [
            "comfortable seating area",
            "harmonious arrangement",
            "focal point",
            "conversation area",
            "statement furniture",
            "layered lighting",
            "accent pieces",
            "balanced decor",
            "entertainment setup",
            "cozy atmosphere"
        ],
        'Bedroom': [
            "luxurious bedding",
            "serene atmosphere",
            "relaxing environment",
            "peaceful setting",
            "comfortable bed",
            "elegant headboard",
            "nightstands",
            "ambient lighting",
            "plush textiles",
            "restful design"
        ],
        'Bathroom': [
            "spa-like features",
            "luxury fixtures",
            "pristine surfaces",
            "premium fittings",
            "elegant vanity",
            "modern fixtures",
            "clean aesthetic",
            "sophisticated tiles",
            "well-lit mirror",
            "sleek design"
        ],
        'Home Office': [
            "productive workspace",
            "ergonomic setup",
            "organized desk",
            "professional environment",
            "efficient layout",
            "storage solutions",
            "task lighting",
            "comfortable seating",
            "clean workspace",
            "functional design"
        ],
        'Dining Room': [
            "elegant dining setup",
            "formal dining space",
            "statement chandelier",
            "dining table arrangement",
            "sophisticated setting",
            "place settings",
            "dining chairs",
            "centerpiece",
            "table styling",
            "entertainment ready"
        ],
        'Coffee Shop': [
            "cafe atmosphere",
            "coffee bar setup",
            "seating arrangement",
            "barista station",
            "coffee equipment",
            "cozy nooks",
            "ambient cafe lighting",
            "counter design",
            "welcoming atmosphere",
            "social space"
        ],
        'Office': [
            "professional environment",
            "corporate setting",
            "collaborative space",
            "modern workstations",
            "meeting areas",
            "office lighting",
            "business atmosphere",
            "productivity focused",
            "organized workspace",
            "professional layout"
        ]
    }

    def __init__(
        self,
        firebase_manager: FirebaseManager,
        sd_service: StableDiffusionImg2Img,
        text2img_service: StableDiffusionText2Img
    ):
        self.firebase_manager = firebase_manager
        self.sd_service = sd_service
        self.text2img_service = text2img_service

    async def generate_similar_images(
        self,
        room_id: str,
        changes: Optional[List[RoomChange]] = None,
        strength: float = 0.7
    ) -> List[Room]:
        # Logging the start of the method
        logger.info(f"Generating similar images for room ID: {room_id}")
        logger.info(f"Changes: {changes}")
        
        try:
            # Get original room
            original_room = await self.firebase_manager.get_room(room_id)
            logger.info(f"Original room retrieved: {original_room.title}")
            
            # Prepare modified metadata from changes
            modified_metadata = {}
            for change in changes or []:
                if change.change_type == ChangeType.STYLE.value:
                    modified_metadata['style'] = change.to_value
                    logger.info(f"Style change to: {change.to_value}")
                elif change.change_type == ChangeType.MATERIAL.value and change.material_type:
                    modified_metadata[f'{change.material_type.value}_material'] = change.to_value
                    logger.info(f"Material change for {change.material_type.value} to: {change.to_value}")

            # Construct prompt
            base_prompt = self._construct_prompt(original_room, modified_metadata)
            logger.info(f"Generated base prompt: {base_prompt}")
            
            # Generate images
            config = Img2ImgConfig(
                init_image=original_room.image_url,
                prompt=base_prompt,
                negative_prompt=self._get_negative_prompt(),
                strength=strength,
                samples="1"
            )
            
            logger.info("Attempting to generate similar images...")
            image_urls = await self.sd_service.generate_similar_images(config)
            logger.info(f"Generated {len(image_urls)} image URLs")
            
            # Store similar rooms and their relationships
            similar_rooms = []
            for url in image_urls:
                logger.info(f"Processing image URL: {url}")
                
                new_room = self._create_similar_room(original_room, url, modified_metadata)
                
                # Upload room to get a valid ID
                logger.info("Uploading new room...")
                uploaded_room = await self.firebase_manager.upload_room(
                    image_path=url,  # Ensure this is handled correctly
                    metadata=new_room.dict()
                )
                logger.info(f"Uploaded room ID: {uploaded_room.id}")
                
                # Create room relationship
                logger.info("Creating room relationship...")
                await self.firebase_manager.create_room_relationship(
                    parent_id=original_room.id,
                    similar_id=uploaded_room.id,
                    changes=changes,
                    prompt=base_prompt
                )
                
                similar_rooms.append(uploaded_room)
            
            logger.info(f"Generated {len(similar_rooms)} similar rooms")
            return similar_rooms
        
        except Exception as e:
            logger.error(f"Error in generate_similar_images: {str(e)}")
            logger.error(f"Full traceback:", exc_info=True)
            raise

    async def generate_img2img_from_pinterest(
        self,
        room: Room,
        pinterest_url: str,
        strength: float = 0.7
    ) -> List[Room]:
        """Generate similar images based on a Room object and Pinterest image URL."""
        try:
            # Clean up room style and type
            style = room.style.lower() if isinstance(room.style, str) else room.style.value.lower()
            room_type = room.room_type.lower() if isinstance(room.room_type, str) else room.room_type.value.lower()

            # Create a meaningful prompt
            prompt = (
                f"Generate a {style} style {room_type} interior, "
                f"with {room.description}. "
                "High quality interior design photo, natural lighting, professional photography."
            )
            logger.info(f"Generating with prompt: {prompt}")

            # Configure stable diffusion request
            config = Img2ImgConfig(
                init_image=pinterest_url,
                prompt=prompt,
                negative_prompt=self.sd_service.get_default_negative_prompt()  # Use instance method
            )

            # Generate images
            image_urls = await self.sd_service.generate_similar_images(config)
            logger.info(f"Generated {len(image_urls)} images")

            # Create similar rooms
            similar_rooms = []
            for url in image_urls:
                new_room = Room(
                    room_type=room.room_type,
                    style=room.style,
                    title=room.title,
                    description=room.description,
                    image_url=url,
                    is_original=False,
                    metadata=room.metadata
                )
                similar_rooms.append(new_room)

            return similar_rooms

        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
            raise

    async def generate_text2img_from_pinterest(
        self,
        room: Room,
        num_samples: str = "1"
    ) -> List[Room]:
        """
        Generate new room images based on a Room object using text-to-image generation.
        
        Args:
            room (Room): The room object containing description and metadata
            num_samples (str): Number of images to generate
            
        Returns:
            List[Room]: List of generated rooms
        """
        logger.info(f"Generating {num_samples} images based on room: {room.title}")
        
        try:
            # Construct comprehensive prompt using room metadata
            base_prompt = f"Generate a {room.style} style {room.room_type} with description {room.description} . It room should have large windows letting in plenty of natural light. The room should feature a warm and cheerful color palette with earthy terracotta, mustard yellow, burnt orange, and deep teal as the primary tones."
            enhancers = self._get_enhancers(room.room_type)

            logger.info(f"Generated prompt: {base_prompt}")

            
            # Add room-specific details and materials if available
            # details = []
            # if room.materials:
            #     if room.materials.floor:
            #         details.append(f"with {room.materials.floor} flooring")
            #     if room.materials.walls:
            #         details.append(f"{room.materials.walls} walls")
            #     if room.materials.fixtures and room.materials.fixtures.ceiling_lights:
            #         details.append(f"{room.materials.fixtures.ceiling_lights} lighting")
            
            # Add colors if available
            # if room.colors:
            #     details.append(f"color scheme featuring {', '.join(room.colors)}")
            
            # Combine all prompt elements
            # detail_text = ", ".join(details) if details else ""
            full_prompt = f"{base_prompt}, {enhancers}"
            
            logger.info(f"Generated prompt: {full_prompt}")
            
            # Configure text2img request
            config = Text2ImgConfig(
                prompt=full_prompt,
                negative_prompt=self._get_negative_prompt(),
                width="1024",  # Higher resolution for better quality
                height="1024",
                samples=num_samples,
                num_inference_steps="40",  # More steps for better quality
                guidance_scale=7.5,
                enhance_prompt="yes",
                safety_checker="yes"
            )
            
            # Generate images
            logger.info("Attempting to generate images...")
            image_urls = await self.text2img_service.generate_images(config)
            logger.info(f"Generated {len(image_urls)} image URLs")
            
            # Store generated rooms
            similar_rooms = []
            for url in image_urls:
                logger.info(f"Processing image URL: {url}")
                
                # Create new room using original room's data
                new_room = self._create_similar_room(
                    original_room=room,
                    image_url=url,
                    modified_metadata={
                        'generation_prompt': full_prompt,
                        'is_original': False,
                        'parent_room_id': room.id
                    }
                )

                 # Upload room to get a valid ID
                # logger.info("Uploading new room...")
                # doc_id, image_url  = await self.firebase_manager.upload_room(
                #     image_path="downloaded_image.jpg",  # Placeholder for actual image paths
                #     metadata=new_room.dict()
                # )
                # logger.info(f"Uploaded room ID: {doc_id} and image URL: {image_url}")
                
                # new_room.id = doc_id
                # new_room.image_url = image_url
                
                similar_rooms.append(new_room)
            
            logger.info(f"Successfully generated {len(similar_rooms)} rooms")
            return similar_rooms
            
        except Exception as e:
            logger.error(f"Error in generate_text2img_from_pinterest: {str(e)}")
            logger.error(f"Full traceback:", exc_info=True)
            raise

    def _get_negative_prompt(self) -> str:
        """Get negative prompt for interior design."""
        negative_elements = [
            # Technical issues
            "blur", "distortion", "low quality", "text", "watermark",
            "signature", "deformed", "bad proportions", "duplicate",
            "out of frame", "unclear", "cropped", "low resolution",
            "pixelated", "compression artifacts", "noise",
            
            # Architectural/Design issues
            "broken architecture", "floating furniture",
            "impossible architecture", "incorrect lighting",
            "inconsistent lighting", "bad shadows",
            
            # Quality issues
            "amateur", "unprofessional", "unfinished",
            "cartoon", "anime", "3d render", "simplified",
            "poor quality", "oversaturated", "undersaturated",
            
            # Unwanted elements
            "people", "persons", "humans", "animals",
            "text overlay", "timestamp", "border", "frame",
            "collage", "stock photo watermark"
        ]
        
        return ", ".join(negative_elements)

    def _get_enhancers(self, room_type: str) -> str:
        """Get enhancers for interior design, including room-specific ones."""
        general_enhancers = [
            # Photography quality
            "interior design", "professional architectural photography",
            "8k resolution", "highly detailed", "natural lighting",
            "photorealistic", "ultra realistic", "interior visualization",
            
            # Technical quality
            "sharp focus", "high definition", "perfect exposure",
            "color graded", "professionally retouched",
            
            # Lighting
            "soft shadows", "ambient lighting", "volumetric lighting",
            "global illumination", "indirect lighting",
            
            # Interior specific
            "detailed materials", "realistic textures",
            "perfect composition", "balanced layout",
            "professional staging", "designer furniture",
            "high-end finishes", "magazine quality"
        ]
        
        # Add room-specific enhancers if available
        specific_enhancers = self.ROOM_SPECIFIC_ENHANCERS.get(room_type, [])
        all_enhancers = general_enhancers + specific_enhancers
        
        return ", ".join(all_enhancers)

    def _construct_prompt(
        self,
        original_room: Room,
        modified_metadata: Optional[dict]
    ) -> str:
        """Construct prompt based on original or modified metadata."""
        if not modified_metadata:
            base_prompt = f"Generate a similar {original_room.room_type} interior, maintaining the {original_room.style} style"
        else:
            style = modified_metadata.get('style', original_room.style)
            room_type = modified_metadata.get('room_type', original_room.room_type)
            base_prompt = f"Transform this {original_room.room_type} into a {style} style {room_type}"

        # Add enhancers, including room-specific ones
        room_type = modified_metadata.get('room_type', original_room.room_type) if modified_metadata else original_room.room_type
        enhancers = self._get_enhancers(room_type)
        
        return f"{base_prompt}, {enhancers}"

    def _create_similar_room(
        self,
        original_room: Room,
        image_url: str,
        modified_metadata: Optional[dict]
    ) -> Room:
        """Create a new room based on original room and modifications."""
        room_data = original_room.dict()
        room_data.update({
            'id': None,  # Will be set by Firebase
            'image_url': image_url,
            'timestamp': datetime.now(),
            'is_original': False,
            'parent_room_id': original_room.id
        })
        
        if modified_metadata:
            room_data.update(modified_metadata)
        
        return Room(**room_data)