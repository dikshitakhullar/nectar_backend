from firebase_admin import storage, firestore
from datetime import datetime
from models.room import Room
from models.room_relationship import (
    RoomRelationship, GenerationMetadata, RelationshipType, 
    RoomChange, ChangeType
)
from typing import List, Optional, Dict
import uuid
import logging
from pathlib import Path

class FirebaseManager:
    def __init__(self):
        self.bucket = storage.bucket()
        self.db = firestore.client()
        self.logger = logging.getLogger(__name__)

    def upload_image(self, image_path: str) -> str:
        """Upload an image to Firebase Storage."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        extension = Path(image_path).suffix
        random_filename = f"{uuid.uuid4()}{extension}"
        
        blob = self.bucket.blob(f"room_images/{random_filename}")
        blob.upload_from_filename(image_path)
        blob.make_public()
        return blob.public_url

    async def upload_room(self, image_path: str, metadata: dict) -> tuple[str, str]:
        """Upload a room's image and metadata to Firebase."""
        # Upload image first
        image_url = self.upload_image(image_path)
        metadata['image_url'] = image_url
        metadata['timestamp'] = datetime.now()

        # Upload metadata
        doc_ref = self.db.collection('room_metadata').document()
        doc_ref.set(metadata)

        return doc_ref.id, image_url

    async def get_room(self, room_id: str) -> Room:
        """Fetch a room by its ID."""
        doc = self.db.collection('room_metadata').document(room_id).get()
        
        if not doc.exists:
            raise ValueError(f"Room with ID {room_id} not found")
        
        room_data = doc.to_dict()
        room_data['id'] = doc.id
        return Room(**room_data)

    def _determine_relationship_type(self, changes: List[RoomChange]) -> RelationshipType:
        """Determine relationship type from changes."""
        if not changes:
            return RelationshipType.AI_GENERATED
        
        type_mapping = {
            ChangeType.PALETTE.value: RelationshipType.PALETTE_VARIATION,
            ChangeType.STYLE.value: RelationshipType.STYLE_VARIATION,
            ChangeType.MATERIAL.value: RelationshipType.MATERIAL_VARIATION,
            ChangeType.LIGHTING.value: RelationshipType.LIGHTING_VARIATION,
            ChangeType.FURNITURE.value: RelationshipType.FURNITURE_VARIATION
        }
        
        try:
            primary_change = changes[0].change_type
            return type_mapping.get(primary_change, RelationshipType.AI_GENERATED)
        except (AttributeError, IndexError):
            return RelationshipType.AI_GENERATED

    async def create_room_relationship( 
        self,
        parent_id: str,
        similar_id: str,
        changes: Optional[List[RoomChange]] = None,
        prompt: str = "",
        model_version: str = ""
    ):
        """Create a relationship between original and generated room."""
        # Default empty changes if None
        changes = changes or []
        
        # Create a simple changes dictionary
        changes_dict = {}
        for c in changes:
            if isinstance(c, str):
                changes_dict[c] = None  # Handle string case
            else:
                # Handle RoomChange object case
                try:
                    changes_dict[c.change_type] = c.to_value
                except AttributeError:
                    self.logger.warning(f"Invalid change object received: {c}")
                    continue

        relationship = RoomRelationship(
            parent_room_id=parent_id,
            similar_room_id=similar_id,
            type=self._determine_relationship_type(changes),
            timestamp=datetime.now(),
            generation_metadata=GenerationMetadata(
                prompt=prompt,
                model_version=model_version,
                changes=changes_dict
            )
        )

        doc_ref = self.db.collection('room_relationships').document()
        doc_ref.set(relationship.model_dump(exclude_none=True))

    async def get_similar_rooms(
        self,
        room_id: str,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[Room]:
        """Fetch similar rooms for a given room ID."""
        query = self.db.collection('room_relationships')\
                     .where('parent_room_id', '==', room_id)
        
        if relationship_type:
            query = query.where('type', '==', relationship_type.value)
            
        relationships = query.stream()
        similar_room_ids = [doc.get('similar_room_id') for doc in relationships]
        
        if not similar_room_ids:
            return []
        
        rooms = []
        # Process in batches of 10 (Firestore limitation)
        for i in range(0, len(similar_room_ids), 10):
            batch = similar_room_ids[i:i + 10]
            batch_query = self.db.collection('room_metadata')\
                            .where(firestore.FieldPath.document_id(), 'in', batch)
            batch_rooms = [
                Room(**{**doc.to_dict(), 'id': doc.id}) 
                for doc in batch_query.stream()
            ]
            rooms.extend(batch_rooms)
            
        return rooms