# Nectar Room Analysis API

## Overview
Nectar Room Analysis API is a FastAPI-based application designed to analyze and generate room variations from Pinterest images. It leverages Stable Diffusion models, Firebase services, and CLIP embeddings to deliver powerful room analysis and image generation features.

## Features
- Analyze room objects from images
- Generate room images using Stable Diffusion
- Find similar images based on room characteristics
- Seamless integration with a Chrome extension for easy image analysis from Pinterest

## Prerequisites
- Python 3.8+
- Firebase credentials (service account JSON)
- Stable Diffusion API key
- Chrome browser for extension integration

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Update the `config.py` file with your Firebase credentials and Stable Diffusion API key:
```python
FIREBASE_CREDENTIALS = "path/to/your/firebase_credentials.json"
FIREBASE_STORAGE_BUCKET = "your-firebase-storage-bucket"
STABLE_DIFFUSION_API_KEY = "your-stable-diffusion-api-key"
```

### 4. Run the Server
Execute the following command to start the FastAPI server:
```bash
python main.py
```
The server will start at `http://localhost:8000`.

### 5. API Endpoints
- **Analyze Room:** `POST /api/analyze`
- **Generate Room Image:** `POST /api/generate`
- **Health Check:** `GET /health`
- **API Documentation:** `GET /docs`

## Chrome Extension Setup

### 2. Load Extension in Chrome
1. Open Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in the top right corner).
3. Click **Load unpacked** and select the directory containing your `manifest.json` and extension files.
4. The "Nectar Room Analyzer" extension should now appear in your extensions list.

