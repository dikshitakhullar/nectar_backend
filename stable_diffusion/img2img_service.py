from typing import List, Optional
import requests
import logging
import asyncio

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

class StableDiffusionImg2Img:
    def __init__(self, api_key: str, base_url: str = "https://modelslab.com/api/v6"):
        self.api_key = api_key
        self.base_url = base_url
        self.img2img_endpoint = f"{base_url}/images/img2img"
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

    async def generate_similar_images(self, config: Img2ImgConfig) -> List[str]:
        """Generate similar images using img2img endpoint"""
        payload = {
            "key": self.api_key,
            "model_id": config.model_id,
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt or self.default_negative_prompt(),
            "init_image": config.init_image,
            "samples": config.samples,
            "num_inference_steps": config.num_inference_steps,
            "safety_checker": config.safety_checker,
            "enhance_prompt": config.enhance_prompt,
            "guidance_scale": config.guidance_scale,
            "strength": config.strength,
            "scheduler": config.scheduler,
            "tomesd": config.tomesd,
            "use_karras_sigmas": config.use_karras_sigmas
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            self.logger.info("Making request to generate images...")
            response = requests.post(
                self.img2img_endpoint,
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
                fetch_url = f"{self.base_url}/images/fetch/{task_id}"
                self.logger.info(f"Images processing, fetching from: {fetch_url}")
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
                    # First try the output field
                    if "output" in result and result["output"]:
                        return result["output"]
                    # Then try future_links
                    elif "future_links" in result and result["future_links"]:
                        return result["future_links"]
                    # Then try meta.output
                    elif "meta" in result and "output" in result["meta"]:
                        return result["meta"]["output"]
                    else:
                        raise Exception("No image URLs found in successful response")
                elif result["status"] == "error":
                    raise Exception(f"API Error: {result.get('message', 'Unknown error')}")
                
                # Wait before next attempt
                await asyncio.sleep(2)
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Polling request failed: {str(e)}")
                raise
                
        raise Exception("Max polling attempts reached without getting results")