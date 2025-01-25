from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # Firebase
    FIREBASE_CREDENTIALS = os.getenv('FIREBASE_CREDENTIALS')
    FIREBASE_STORAGE_BUCKET = os.getenv('FIREBASE_STORAGE_BUCKET')

    # Stable Diffusion
    STABLE_DIFFUSION_API_KEY = os.getenv('STABLE_DIFFUSION_API_KEY')
    STABLE_DIFFUSION_BASE_URL = os.getenv('STABLE_DIFFUSION_BASE_URL', 'https://modelslab.com/api/v6')


config = Config()