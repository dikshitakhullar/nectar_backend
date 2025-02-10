from typing import List, Optional
import requests
import logging
import asyncio
from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class Text2ImgConfig:
    """Configuration for text-to-image generation."""
    prompt: str
    model_id: str = "realistic-vision-51"
    negative_prompt: Optional[str] = None
    width: str = "512"
    height: str = "512"
    samples: str = "1"
    num_inference_steps: str = "30"
    safety_checker: str = "yes"
    enhance_prompt: str = "yes"
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    panorama: str = "no"
    self_attention: str = "no"
    upscale: str = "no"
    embeddings_model: Optional[str] = None
    lora_model: Optional[str] = None
    tomesd: str = "yes"
    use_karras_sigmas: str = "yes"
    vae: Optional[str] = None
    lora_strength: Optional[float] = None
    scheduler: str = "UniPCMultistepScheduler"
    webhook: Optional[str] = None
    track_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, removing None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

class StableDiffusionText2Img:
    """Service for generating images from text using Stable Diffusion API."""
    
    def __init__(self, api_key: str, base_url: str = "https://modelslab.com/api/v6"):
        self.api_key = api_key
        self.base_url = base_url
        self.text2img_endpoint = f"{base_url}/images/text2img"
        self.logger = logging.getLogger(__name__)

    @classmethod
    def default_negative_prompt(cls) -> str:
        """Get default negative prompt for interior design."""
        return (
            "blur, blurry, distortion, distorted, low quality, "
            "text, watermark, signature, deformed, bad proportions, "
            "duplicate, out of frame, unclear, cropped, low resolution, "
            "bad shadows, inconsistent lighting, broken architecture, "
            "floating furniture, impossible architecture, poorly drawn, "
            "bad perspective, unrealistic scale, poor composition, "
            "missing details, unnatural colors, artificial textures"
        )

    async def generate_images(self, config: Text2ImgConfig) -> List[str]:
        """Generate images using text2img endpoint."""
        payload = {
            "key": self.api_key,
            **config.to_dict()
        }

        if not payload.get("negative_prompt"):
            payload["negative_prompt"] = self.default_negative_prompt()

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            self.logger.info("Making request to generate images...")
            response = requests.post(
                self.text2img_endpoint,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            
            self.logger.debug(f"Initial response: {result}")

            if result["status"] == "success":
                self.logger.info("Generation successful!")
                return result["output"]
            elif result["status"] == "processing":
                task_id = result["id"]
                self.logger.info(f"Images processing, task ID: {task_id}")
                return await self._poll_for_results(task_id)
            else:
                error_message = result.get("message", "Unknown API error")
                raise Exception(f"API Error: {error_message}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise

    async def _poll_for_results(self, task_id: str, max_attempts: int = 20) -> List[str]:
        """Poll the fetch endpoint until results are ready."""
        fetch_url = f"{self.base_url}/images/fetch/{task_id}"
        
        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Polling attempt {attempt + 1}/{max_attempts}")
                response = requests.post(
                    fetch_url,
                    json={"key": self.api_key},
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                result = response.json()
                
                self.logger.debug(f"Poll response: {result}")
                
                if result["status"] == "success":
                    if "output" in result and result["output"]:
                        return result["output"]
                    elif "future_links" in result and result["future_links"]:
                        return result["future_links"]
                    elif "meta" in result and "output" in result["meta"]:
                        return result["meta"]["output"]
                    else:
                        raise Exception("No image URLs found in successful response")
                elif result["status"] == "error":
                    raise Exception(f"API Error: {result.get('message', 'Unknown error')}")
                
                await asyncio.sleep(2)
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Polling request failed: {str(e)}")
                raise
                
        raise Exception("Max polling attempts reached without getting results")

# Example usage:
async def main():
    service = StableDiffusionText2Img(api_key="your_api_key")
    config = Text2ImgConfig(
        prompt="A modern minimalist living room with large windows",
        width="768",
        height="768",
        samples="2"
    )
    
    try:
        image_urls = await service.generate_images(config)
        print(f"Generated images: {image_urls}")
    except Exception as e:
        print(f"Error generating images: {e}")

if __name__ == "__main__":
    asyncio.run(main())