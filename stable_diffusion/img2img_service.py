from typing import List, Optional
import requests
import logging
import asyncio
import json
import random
from requests.exceptions import RequestException, Timeout

class Img2ImgConfig:
    def __init__(
        self,
        init_image: str,
        prompt: str,
        model_id: str = "realistic-vision-51",
        negative_prompt: Optional[str] = None,
        samples: str = "1",
        num_inference_steps: str = "31",
        safety_checker: str = "yes",
        enhance_prompt: str = "yes",
        guidance_scale: float = 7.5,
        strength: float = 0.7,
        scheduler: str = "UniPCMultistepScheduler",
        tomesd: str = "yes",
        use_karras_sigmas: str = "yes"
    ):
        self.init_image = init_image
        self.prompt = prompt
        self.model_id = model_id
        self.negative_prompt = negative_prompt
        self.samples = samples
        self.num_inference_steps = num_inference_steps
        self.safety_checker = safety_checker
        self.enhance_prompt = enhance_prompt
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.scheduler = scheduler
        self.tomesd = tomesd
        self.use_karras_sigmas = use_karras_sigmas
    
    def to_payload(self) -> dict:
        """Convert config to API payload format"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class StableDiffusionImg2Img:
    def __init__(self, api_key: str, base_url: str = "https://modelslab.com/api/v6"):
        self.api_key = api_key
        self.base_url = base_url
        self.img2img_endpoint = f"{base_url}/images/img2img"
        self.logger = logging.getLogger(__name__)
        
        # Configuration for retries
        self.max_retries = 5  # Maximum number of retries
        self.base_delay = 20  # Base delay in seconds
        self.max_delay = 180  # Maximum delay of 3 minutes
        self.request_timeout = 120  # Request timeout in seconds
        self.max_polling_attempts = 20  # Maximum number of polling attempts

    def get_default_negative_prompt(self) -> str:
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

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(self.base_delay * (1.5 ** (attempt - 1)), self.max_delay)
        return delay + random.uniform(1, 5)

    async def generate_similar_images(self, config: Img2ImgConfig) -> List[str]:
        """Generate similar images with bounded retries."""
        payload = {
            "key": self.api_key,
            **config.to_payload()
        }
        
        # Log the payload with redacted API key
        safe_payload = {**payload, "key": "REDACTED"}
        self.logger.info(f"Request payload: {json.dumps(safe_payload, indent=2)}")

        headers = {'Content-Type': 'application/json'}

        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(f"Generation attempt {attempt}/{self.max_retries}")
                
                response = requests.post(
                    self.img2img_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.request_timeout
                )

                self.logger.debug(f"Response status: {response.status_code}")
                
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON response: {response.text}")
                    raise Exception("Invalid JSON response from API")

                # Handle rate limits and retryable responses
                if response.status_code == 429 or (
                    isinstance(result, dict) and 
                    "try again" in str(result.get("message", "")).lower()
                ):
                    if attempt == self.max_retries:
                        raise Exception("Max retries reached - rate limit persists")
                    
                    delay = self._calculate_delay(attempt)
                    self.logger.info(f"Rate limit hit, waiting {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()

                if isinstance(result, dict):
                    if result.get("status") == "success" and "output" in result:
                        return result["output"]
                    elif result.get("status") == "processing":
                        self.logger.info("Request is processing, waiting for result...")
                        return await self._poll_for_results(result["id"])
                    else:
                        error_msg = result.get("message", "Unknown API error")
                        raise Exception(f"API Error: {error_msg}")

                raise Exception("Invalid response format from API")

            except (Timeout, RequestException) as e:
                if attempt == self.max_retries:
                    raise Exception(f"Failed after {self.max_retries} attempts: {str(e)}")
                
                delay = self._calculate_delay(attempt)
                self.logger.error(f"Request failed: {str(e)}, retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)

    async def _poll_for_results(self, task_id: str) -> List[str]:
        """Poll for results with bounded attempts."""
        fetch_url = f"{self.base_url}/images/fetch/{task_id}"
        poll_interval = 15
        
        for attempt in range(1, self.max_polling_attempts + 1):
            try:
                self.logger.info(f"Polling for results, attempt {attempt}/{self.max_polling_attempts}")
                response = requests.post(
                    fetch_url,
                    json={"key": self.api_key},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON in poll response: {response.text}")
                    raise Exception("Invalid JSON in poll response")
                
                if result.get("status") == "success" and "output" in result:
                    return result["output"]
                elif result.get("status") == "processing":
                    if attempt == self.max_polling_attempts:
                        raise Exception(f"Polling timed out after {self.max_polling_attempts} attempts")
                    
                    self.logger.info(f"Still processing, waiting {poll_interval} seconds...")
                    await asyncio.sleep(poll_interval)
                    continue
                else:
                    error_msg = result.get("message", "Unknown error")
                    raise Exception(f"API Error: {error_msg}")
                    
            except Exception as e:
                if attempt == self.max_polling_attempts:
                    raise Exception(f"Polling failed after {self.max_polling_attempts} attempts: {str(e)}")
                
                self.logger.error(f"Poll attempt failed: {str(e)}")
                await asyncio.sleep(poll_interval)