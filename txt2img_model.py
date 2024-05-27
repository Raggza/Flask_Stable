import torch
from diffusers import StableDiffusionPipeline

rand_seed = torch.manual_seed(42)
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 0.75
HEIGHT = 512
WITDH = 512


def create_pipeline(model_name):
    if torch.cuda.is_available():
        print("Using GPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
        ).to("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
        ).to("mps")
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
        )
    return pipeline


def txt2img(prompt, pipeline):
    images = pipeline(
        prompt,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=rand_seed,
        num_images_per_request=1,
        height=HEIGHT,
        witdh=WITDH,
    ).images

    return images[0]
