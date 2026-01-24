"""Image generation tool using local Stable Diffusion APIs."""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """Result from image generation."""

    success: bool
    path: str | None = None
    error: str | None = None
    seed: int | None = None
    generation_time: float | None = None


class ImageGenerationTool:
    """
    Image generation using local Stable Diffusion.

    Supports:
    - ComfyUI API (default, port 8188)
    - Automatic1111 API (port 7860)
    """

    def __init__(self) -> None:
        """Initialize the image generation tool."""
        self.comfyui_url = getattr(settings, "comfyui_url", "http://localhost:8188")
        self.a1111_url = getattr(settings, "automatic1111_url", "http://localhost:7860")
        self.output_dir = Path(getattr(settings, "image_output_dir", "data/images"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._backend: str | None = None

    def _detect_backend(self) -> str | None:
        """Detect which SD backend is available."""
        if self._backend:
            return self._backend

        # Try ComfyUI first
        try:
            response = httpx.get(f"{self.comfyui_url}/system_stats", timeout=2.0)
            if response.status_code == 200:
                self._backend = "comfyui"
                logger.info("Detected ComfyUI backend")
                return "comfyui"
        except Exception:
            pass

        # Try Automatic1111
        try:
            response = httpx.get(f"{self.a1111_url}/sdapi/v1/sd-models", timeout=2.0)
            if response.status_code == 200:
                self._backend = "a1111"
                logger.info("Detected Automatic1111 backend")
                return "a1111"
        except Exception:
            pass

        logger.warning("No Stable Diffusion backend detected")
        return None

    def check_availability(self) -> bool:
        """Check if any SD backend is available."""
        return self._detect_backend() is not None

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
    ) -> dict[str, Any]:
        """
        Generate an image from a prompt.

        Args:
            prompt: The positive prompt for generation
            negative_prompt: Negative prompt (things to avoid)
            width: Image width
            height: Image height
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (-1 for random)

        Returns:
            Dict with 'success', 'path', 'error', 'seed', 'generation_time'
        """
        backend = self._detect_backend()

        if not backend:
            return {
                "success": False,
                "error": (
                    "No Stable Diffusion backend available. "
                    f"Please start ComfyUI ({self.comfyui_url}) or "
                    f"Automatic1111 ({self.a1111_url})"
                ),
            }

        start_time = time.time()

        try:
            if backend == "a1111":
                result = self._generate_a1111(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                )
            else:
                result = self._generate_comfyui(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                )

            result["generation_time"] = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generation_time": time.time() - start_time,
            }

    def _generate_a1111(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
    ) -> dict[str, Any]:
        """Generate image using Automatic1111 API."""
        import base64

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "low quality, blurry, bad anatomy",
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "sampler_name": "Euler a",
        }

        response = httpx.post(
            f"{self.a1111_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        if "images" not in data or not data["images"]:
            return {"success": False, "error": "No images in response"}

        # Decode and save the image
        image_data = base64.b64decode(data["images"][0])
        timestamp = int(time.time())
        filename = f"sd_{timestamp}_{seed if seed != -1 else 'random'}.png"
        filepath = self.output_dir / filename

        with open(filepath, "wb") as f:
            f.write(image_data)

        # Get the actual seed used
        info = data.get("info", "{}")
        if isinstance(info, str):
            import json
            try:
                info = json.loads(info)
            except json.JSONDecodeError:
                info = {}

        actual_seed = info.get("seed", seed)

        logger.info(f"Generated image: {filepath}")

        return {
            "success": True,
            "path": str(filepath),
            "seed": actual_seed,
        }

    def _get_comfyui_checkpoint(self) -> str | None:
        """Get the first available checkpoint from ComfyUI."""
        try:
            response = httpx.get(
                f"{self.comfyui_url}/object_info/CheckpointLoaderSimple",
                timeout=5.0,
            )
            if response.status_code == 200:
                data = response.json()
                ckpts = (
                    data.get("CheckpointLoaderSimple", {})
                    .get("input", {})
                    .get("required", {})
                    .get("ckpt_name", [[]])[0]
                )
                if ckpts and isinstance(ckpts, (list, tuple)) and len(ckpts) > 0:
                    return ckpts[0]
        except Exception as e:
            logger.warning(f"Failed to get ComfyUI checkpoints: {e}")
        return None

    def _generate_comfyui(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
    ) -> dict[str, Any]:
        """Generate image using ComfyUI API."""
        import random
        import uuid

        # Get available checkpoint
        checkpoint = self._get_comfyui_checkpoint()
        if not checkpoint:
            return {
                "success": False,
                "error": "No checkpoints found in ComfyUI. Please add a model to your models/checkpoints folder.",
            }

        logger.info(f"Using ComfyUI checkpoint: {checkpoint}")

        # ComfyUI requires a workflow. Use a simple txt2img workflow.
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Basic txt2img workflow for ComfyUI
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg_scale,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "seed": seed,
                    "steps": steps,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": checkpoint,
                },
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": height,
                    "width": width,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": prompt,
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": negative_prompt or "low quality, blurry",
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "llm_agent",
                    "images": ["8", 0],
                },
            },
        }

        client_id = str(uuid.uuid4())

        # Queue the prompt
        response = httpx.post(
            f"{self.comfyui_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        prompt_id = data.get("prompt_id")

        if not prompt_id:
            return {"success": False, "error": "Failed to queue prompt"}

        # Wait for completion using queue status
        max_wait = 300  # 5 minutes for first-time model loading
        wait_time = 0
        last_status = ""

        while wait_time < max_wait:
            # Check queue status first
            try:
                queue_response = httpx.get(
                    f"{self.comfyui_url}/queue",
                    timeout=5.0,
                )
                if queue_response.status_code == 200:
                    queue_data = queue_response.json()
                    running = queue_data.get("queue_running", [])
                    pending = queue_data.get("queue_pending", [])

                    # Check if our prompt is still in queue
                    our_running = any(item[1] == prompt_id for item in running)
                    our_pending = any(item[1] == prompt_id for item in pending)

                    status = "running" if our_running else ("pending" if our_pending else "done")
                    if status != last_status:
                        logger.info(f"ComfyUI status: {status}")
                        last_status = status

                    # If not in queue anymore, check history for results
                    if not our_running and not our_pending:
                        history_response = httpx.get(
                            f"{self.comfyui_url}/history/{prompt_id}",
                            timeout=5.0,
                        )

                        if history_response.status_code == 200:
                            history = history_response.json()
                            if prompt_id in history:
                                # Check for errors
                                status_info = history[prompt_id].get("status", {})
                                if status_info.get("status_str") == "error":
                                    messages = status_info.get("messages", [])
                                    error_msg = "; ".join(str(m) for m in messages) if messages else "Unknown error"
                                    return {"success": False, "error": f"ComfyUI error: {error_msg}"}

                                outputs = history[prompt_id].get("outputs", {})
                                for node_id, output in outputs.items():
                                    if "images" in output:
                                        # Get the first image
                                        image_info = output["images"][0]
                                        filename = image_info.get("filename")
                                        subfolder = image_info.get("subfolder", "")

                                        # Download the image
                                        params = {"filename": filename, "subfolder": subfolder, "type": "output"}
                                        img_response = httpx.get(
                                            f"{self.comfyui_url}/view",
                                            params=params,
                                            timeout=30.0,
                                        )

                                        if img_response.status_code == 200:
                                            # Save locally
                                            timestamp = int(time.time())
                                            local_filename = f"sd_{timestamp}_{seed}.png"
                                            filepath = self.output_dir / local_filename

                                            with open(filepath, "wb") as f:
                                                f.write(img_response.content)

                                            logger.info(f"Generated image: {filepath}")

                                            return {
                                                "success": True,
                                                "path": str(filepath),
                                                "seed": seed,
                                            }

                                # No images found in completed job
                                return {"success": False, "error": "No images in output"}

            except Exception as e:
                logger.warning(f"Error checking queue: {e}")

            time.sleep(2)
            wait_time += 2

        return {"success": False, "error": f"Generation timed out after {max_wait}s"}

    def list_models(self) -> list[str]:
        """List available models."""
        backend = self._detect_backend()

        if backend == "a1111":
            try:
                response = httpx.get(
                    f"{self.a1111_url}/sdapi/v1/sd-models",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    models = response.json()
                    return [m.get("model_name", m.get("title", "unknown")) for m in models]
            except Exception as e:
                logger.error(f"Failed to list A1111 models: {e}")

        elif backend == "comfyui":
            try:
                response = httpx.get(
                    f"{self.comfyui_url}/object_info/CheckpointLoaderSimple",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    ckpts = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
                    return list(ckpts) if isinstance(ckpts, (list, tuple)) else []
            except Exception as e:
                logger.error(f"Failed to list ComfyUI models: {e}")

        return []
