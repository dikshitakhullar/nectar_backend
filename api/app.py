# api/app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
from api.models.requests import (
    AnalyzeRequest,
    AnalyzeResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateVariationType,
    ErrorResponse
)
import logging
import traceback
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class APIError(Exception):
    def __init__(self, message: str, status_code: int = 500, details: Dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

def setup_routes(app: FastAPI):
    """Setup routes for the FastAPI application."""

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
                "status_code": exc.status_code
            }
        )

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_pinterest_image(request: AnalyzeRequest, req: Request):
        """Analyze a Pinterest image and return room details."""
        try:
            services = req.app.state.services
            if not services:
                raise HTTPException(status_code=500, detail="Services not initialized")

            # 1. Download image
            image_path = "downloaded_image.jpg"
            logger.info(f"Downloading image from {request.pinterest_url}")
            
            download_result = services['download_pinterest_image'](
                request.pinterest_url,
                image_path
            )
            if not download_result:
                raise HTTPException(status_code=400, detail="Failed to download Pinterest image")

            # 2. Analyze room
            room_style, room_type = services['get_clip_embeddings'](image_path)
            detected_objects = services['analyze_room_objects'](image_path)
            
            # 3. Clean detected objects (convert numpy types to Python types)
            cleaned_objects = []
            for obj in detected_objects:
                cleaned_obj = {
                    "class_name": obj["class_name"],
                    "confidence": float(obj["confidence"]),
                    "coordinates": [float(x) for x in obj["bounding_box"]],
                    "attributes": obj["attributes"]
                }
                cleaned_objects.append(cleaned_obj)

            # 4. Create room object
            room = services['create_room_from_analysis'](
                cleaned_objects,
                room_style,
                room_type
            )

            # 5. Upload to Firebase
            doc_id, public_url = await services['firebase_manager'].upload_room(
                image_path=image_path,
                metadata=room.dict()
            )

            # 6. Update room with ID and URL
            room.id = doc_id
            room.image_url = public_url

            # 7. Cleanup downloaded image
            if os.path.exists(image_path):
                os.remove(image_path)

            return AnalyzeResponse(
                room=room,
                public_url=public_url,
                detected_objects=cleaned_objects
            )

        except Exception as e:
            logger.error(f"Error in analyze_pinterest_image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate_variation(request: GenerateRequest, req: Request):
        """Generate a variation of the analyzed room."""
        logger.info(f"in generate_variation: {request}")
        try:
            services = req.app.state.services
            if not services:
                raise APIError("Services not initialized", status_code=500)

            try:
                if request.variation_type == GenerateVariationType.SLIGHT:
                    similar_rooms = await services['similar_service'].generate_img2img_from_pinterest(
                        room=request.room,
                        pinterest_url=request.room.image_url  # Use stored Firebase URL
                    )
                else:  # significant variation
                    similar_rooms = await services['similar_service'].generate_text2img_from_pinterest(
                        room=request.room
                    )
            except Exception as e:
                logger.error(f"Failed to generate variation: {str(e)}")
                raise APIError(
                    "Failed to generate room variation",
                    status_code=500,
                    details={
                        "error": str(e),
                        "variation_type": request.variation_type
                    }
                )

            if not similar_rooms:
                raise APIError(
                    "No rooms generated",
                    status_code=500,
                    details={"variation_type": request.variation_type}
                )

            return GenerateResponse(
                generated_room=similar_rooms[0],
                metadata={
                    "total_generated": len(similar_rooms),
                    "variation_type": request.variation_type
                }
            )

        except APIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_variation: {str(e)}")
            logger.error(traceback.format_exc())
            raise APIError(
                "Internal server error",
                status_code=500,
                details={"error": str(e)}
            )

    @app.get("/health")
    async def health_check(req: Request):
        """Health check endpoint."""
        services = req.app.state.services
        return {
            "status": "healthy" if services else "services not initialized",
            "version": "1.0.0",
            "services_status": {
                name: "initialized" for name in (services or {}).keys()
            }
        }
