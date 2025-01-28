from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime

class RoomType(str, Enum):
   KITCHEN = "Kitchen"
   LIVING_ROOM = "Living Room"
   DINING_ROOM = "Dining Room"
   BATHROOM = "Bathroom"
   BEDROOM = "Bedroom"
   HOME_OFFICE = "Home Office"
   COFFEE_SHOP = "Coffee Shop"
   OFFICE = "Office"
   HALLWAY = "Hallway"
   GARDEN = "Garden"
   OUTDOOR_PATIO = "Outdoor Patio"
   STUDY_ROOM = "Study Room"
   NURSERY = "Nursery"

class ColorPalette(str, Enum):
   WARM_NEUTRALS = "warm_neutrals"
   COOL_NEUTRALS = "cool_neutrals"
   EARTH_TONES = "earth_tones"
   JEWEL_TONES = "jewel_tones"
   MONOCHROMATIC = "monochromatic"
   COASTAL = "coastal"
   PASTELS = "pastels"

class RoomStyle(str, Enum):
   MODERN = "modern"
   CONTEMPORARY = "contemporary"
   TRADITIONAL = "traditional"
   MINIMALIST = "minimalist"
   INDUSTRIAL = "industrial"
   SCANDINAVIAN = "scandinavian"
   MID_CENTURY_MODERN = "mid_century_modern"
   BOHEMIAN = "bohemian"
   RUSTIC = "rustic"
   COASTAL = "coastal"
   NEOCLASSICAL = "neoclassical"
   ART_DECO = "art_deco"
   ECLECTIC = "eclectic"
   FARMHOUSE = "farmhouse"
   MEDITERRANEAN = "mediterranean"
   ASIAN = "asian"
   TRANSITIONAL = "transitional"
   IMPERIAL = "imperial"
   VINTAGE = "vintage"
   LUXURY = "luxury"

class LightingType(str, Enum):
   NATURAL = "natural"
   AMBIENT = "ambient"
   TASK = "task"
   ACCENT = "accent"
   MIXED = "mixed"

class WindowType(str, Enum):
   BAY = "bay"
   BOW = "bow"
   CASEMENT = "casement"
   DOUBLE_HUNG = "double_hung"
   PICTURE = "picture"
   SLIDING = "sliding"
   FRENCH = "french"
   SKYLIGHT = "skylight"

class WindowTreatment(str, Enum):
   CURTAINS = "curtains"
   DRAPES = "drapes"
   BLINDS = "blinds"
   SHADES = "shades"
   SHUTTERS = "shutters"
   VALANCES = "valances"
   SHEERS = "sheers"

class FurnitureType(str, Enum):
   # Storage
   CABINET = "cabinet"
   SHELVING = "shelving"
   DRESSER = "dresser"
   WARDROBE = "wardrobe"
   BOOKCASE = "bookcase"
   SIDEBOARD = "sideboard"
   MEDIA_CONSOLE = "media_console"
   
   # Tables
   DINING_TABLE = "dining_table"
   COFFEE_TABLE = "coffee_table"
   SIDE_TABLE = "side_table"
   CONSOLE_TABLE = "console_table"
   DESK = "desk"
   
   # Beds
   QUEEN_BED = "queen_bed"
   KING_BED = "king_bed"
   TWIN_BED = "twin_bed"
   SOFA_BED = "sofa_bed"
   DAYBED = "daybed"
   
   # Seating
   DINING_CHAIR = "dining_chair"
   ARM_CHAIR = "arm_chair"
   SIDE_CHAIR = "side_chair"
   BAR_STOOL = "bar_stool"
   BENCH = "bench"
   SOFA = "sofa"
   LOVESEAT = "loveseat"
   OTTOMAN = "ottoman"
   WINDOW_SEAT = "window_seat"
   BANQUETTE = "banquette"
   ACCENT_CHAIR = "accent_chair"
   RECLINER = "recliner"
   CHAISE_LOUNGE = "chaise_lounge"

class MaterialFinish(BaseModel):
    material: str
    color: str
    texture: Optional[str] = None
    pattern: Optional[str] = None
    finish: Optional[str] = None

class Lighting(BaseModel):
    type: Optional[LightingType] = LightingType.MIXED
    intensity: Optional[str] = "medium"
    natural_light: Optional[str] = None
    fixtures: Optional[List[str]] = None
    features: Optional[List[str]] = None

class Window(BaseModel):
    type: Optional[WindowType] = WindowType.PICTURE
    count: int = 1
    treatment: Optional[WindowTreatment] = None
    treatment_material: Optional[str] = None
    treatment_color: Optional[str] = None

class FurnitureMaterial(BaseModel):
    material_type: str
    color: str
    finish: Optional[str] = None
    texture: Optional[str] = None
    pattern: Optional[str] = None

class FixtureMaterials(BaseModel):
    ceiling_lights: Optional[str] = None
    wall_lights: Optional[str] = None
    floor_lamps: Optional[str] = None
    faucets: Optional[str] = None
    sink: Optional[str] = None
    door_handles: Optional[str] = None
    cabinet_hardware: Optional[str] = None
    chandeliers: Optional[str] = None
    pendant_lights: Optional[str] = None
    track_lighting: Optional[str] = None
    sconces: Optional[str] = None

class FurniturePiece(BaseModel):
    furniture_type: FurnitureType
    material: Optional[FurnitureMaterial] = None  # Made optional
    quantity: int = 1
    notes: Optional[str] = None

class RoomFurniture(BaseModel):
    pieces: Optional[List[FurniturePiece]] = None  # Made optional

class CarpetMaterial(BaseModel):
    material_type: str  # e.g., wool, nylon, polyester
    color: str

class Painting(BaseModel):
    type: str  # e.g., abstract, landscape, portrait
    style: str  # e.g., modern, classical, contemporary
    size: str  # e.g., large, medium, small

class RoomMetadata(BaseModel):
    tags: Optional[List[str]] = None
    ceiling_height: Optional[float] = None
    color_palette: Optional[List[ColorPalette]] = None
    lighting: Optional[Lighting] = None
    floor: Optional[MaterialFinish] = None
    carpet: Optional[CarpetMaterial] = None
    paintings: Optional[List[Painting]] = None
    walls: Optional[MaterialFinish] = None
    ceiling: Optional[MaterialFinish] = None
    furniture: Optional[RoomFurniture] = None
    fixtures: Optional[FixtureMaterials] = None
    windows: Optional[List[Window]] = None

class Room(BaseModel):
    id: Optional[str] = None
    room_type: RoomType
    style: RoomStyle
    metadata: Optional[RoomMetadata] = None  # Made optional
    title: str
    description: str
    image_url: str
    generation_prompt: Optional[str] = None
    is_original: bool = True
    timestamp: Optional[datetime] = None
    views: int = 0
    saves: int = 0