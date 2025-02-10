# backend/api/models/requests.py
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Optional, Dict, Any
from models.room import Room, RoomType, RoomStyle, RoomMetadata
from enum import Enum
import numpy as np

class GenerateVariationType(str, Enum):
    SLIGHT = "slight"
    SIGNIFICANT = "significant"

class AnalyzeRequest(BaseModel):
    pinterest_url: str

class AnalyzeResponse(BaseModel):
    room: Room
    public_url: str
    detected_objects: List[Dict]
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist()
        }
    )

class GenerateRequest(BaseModel):
    variation_type: GenerateVariationType
    variation_parameters: Optional[Dict] = None
    room: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('room')
    @classmethod
    def validate_room(cls, value):
        if isinstance(value, dict):
            # Handle room_type
            room_type = value.get('room_type')
            if isinstance(room_type, dict) and room_type.get('__type') == 'RoomType':
                room_type = room_type['value']
            if isinstance(room_type, str):
                try:
                    room_type_key = room_type.replace(" ", "_").upper()
                    value['room_type'] = RoomType[room_type_key]
                except (KeyError, ValueError):
                    value['room_type'] = RoomType.LIVING_ROOM

            # Handle style
            style = value.get('style')
            if isinstance(style, dict) and style.get('__type') == 'RoomStyle':
                style = style['value']
            if isinstance(style, str):
                try:
                    value['style'] = RoomStyle(style.lower())
                except ValueError:
                    value['style'] = RoomStyle.MODERN
                    
            return Room(**value)
        return value

class GenerateResponse(BaseModel):
    generated_room: Room
    metadata: Optional[Dict] = None

class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None

class VariationParameters(BaseModel):
    modify_style: bool = False
    modify_colors: bool = False
    modify_lighting: bool = False
    modify_furniture: bool = False
    modify_layout: bool = False
    target_style: Optional[RoomStyle] = None
    target_lighting: Optional[str] = None
    additional_prompts: Optional[List[str]] = None