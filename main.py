# main.py
import logging
from contextlib import asynccontextmanager
from firebase_admin import credentials, initialize_app, get_app
from config import config
from firebase_operations.firebase_manager import FirebaseManager
from stable_diffusion.img2img_service import StableDiffusionImg2Img
from stable_diffusion.text2img_service import StableDiffusionText2Img
from services.similar_images_service import SimilarImagesService
from pinterest_utils import download_pinterest_image
from clip import get_clip_embeddings
from room_object_analysis import analyze_room_objects, create_room_from_analysis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_firebase():
    """Initialize Firebase if not already initialized."""
    try:
        return get_app()
    except ValueError:
        cred = credentials.Certificate(config.FIREBASE_CREDENTIALS)
        return initialize_app(cred, {'storageBucket': config.FIREBASE_STORAGE_BUCKET})

def get_services():
    """Initialize all required services."""
    try:
        # Initialize Firebase safely
        firebase_app = initialize_firebase()
        logger.info("Firebase initialized successfully")
        
        # Create service instances
        firebase_manager = FirebaseManager()
        sd_service = StableDiffusionImg2Img(api_key=config.STABLE_DIFFUSION_API_KEY)
        text2img_service = StableDiffusionText2Img(api_key=config.STABLE_DIFFUSION_API_KEY)
        
        # Create SimilarImagesService
        similar_service = SimilarImagesService(
            firebase_manager=firebase_manager,
            sd_service=sd_service,
            text2img_service=text2img_service
        )

        return {
            'firebase_manager': firebase_manager,
            'similar_service': similar_service,
            'download_pinterest_image': download_pinterest_image,
            'get_clip_embeddings': get_clip_embeddings,
            'analyze_room_objects': analyze_room_objects,
            'create_room_from_analysis': create_room_from_analysis
        }
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application")
    # Initialize services
    app.state.services = get_services()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Nectar Room Analysis API",
    description="API for analyzing and generating room images",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes after services are initialized
from api.app import setup_routes
setup_routes(app)

# Add default route
@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return JSONResponse({
        "name": "Nectar Room Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "generate": "/api/generate",
            "health": "/health",
            "docs": "/docs"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)