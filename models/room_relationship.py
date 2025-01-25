from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel

class ColorPalette(str, Enum):
    WARM_NEUTRALS = "warm_neutrals"
    COOL_NEUTRALS = "cool_neutrals"
    EARTH_TONES = "earth_tones"
    JEWEL_TONES = "jewel_tones"
    MONOCHROMATIC = "monochromatic"
    COASTAL = "coastal"
    PASTELS = "pastels"

class ChangeType(str, Enum):
    PALETTE = "palette"
    STYLE = "style"
    MATERIAL = "material"
    LIGHTING = "lighting"
    FURNITURE = "furniture"

class MaterialType(str, Enum):
    FLOOR = "floor"
    WALLS = "walls"
    CEILING = "ceiling"
    COUNTERTOPS = "countertops"
    CABINETS = "cabinets"

class RelationshipType(str, Enum):
    AI_GENERATED = "aiGenerated"
    PALETTE_VARIATION = "paletteVariation"
    STYLE_VARIATION = "styleVariation"
    MATERIAL_VARIATION = "materialVariation"
    LIGHTING_VARIATION = "lightingVariation"
    FURNITURE_VARIATION = "furnitureVariation"

@dataclass
class RoomChange:
    change_type: str
    to_value: str
    from_value: Optional[str] = None
    material_type: Optional[MaterialType] = None
    furniture_type: Optional[str] = None

    @classmethod
    def palette(cls, from_value: ColorPalette, to_value: ColorPalette) -> 'RoomChange':
        return cls(
            change_type=ChangeType.PALETTE.value,
            to_value=to_value.value,
            from_value=from_value.value
        )

    @classmethod
    def style(cls, to_value: str) -> 'RoomChange':
        return cls(change_type=ChangeType.STYLE.value, to_value=to_value)

    @classmethod
    def material(cls, material_type: MaterialType, to_value: str) -> 'RoomChange':
        return cls(
            change_type=ChangeType.MATERIAL.value,
            to_value=to_value,
            material_type=material_type
        )

    @classmethod
    def lighting(cls, to_value: str) -> 'RoomChange':
        return cls(change_type=ChangeType.LIGHTING.value, to_value=to_value)

    @classmethod
    def furniture(cls, furniture_type: str, to_value: str) -> 'RoomChange':
        return cls(
            change_type=ChangeType.FURNITURE.value,
            to_value=to_value,
            furniture_type=furniture_type
        )

class GenerationMetadata(BaseModel):
    prompt: str
    model_version: str
    changes: Optional[Dict[str, str]] = None
    original_style: Optional[str] = None
    original_palette: Optional[ColorPalette] = None
    original_materials: Optional[Dict[str, str]] = None

class RoomRelationship(BaseModel):
    id: Optional[str] = None
    parent_room_id: str
    similar_room_id: str
    type: RelationshipType
    timestamp: datetime
    generation_metadata: GenerationMetadata