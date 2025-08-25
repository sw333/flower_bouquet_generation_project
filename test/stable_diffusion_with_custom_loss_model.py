from diffusers import StableDiffusionPipeline
import torch
import os

# Define the path to your fine-tuned LoRA model and output folder
model_path = "stable_diffusion_with_custom_loss_model"
output_folder = "stable_diffusion_with_custom_loss_model_100_image"

# Load the Stable Diffusion model with LoRA weights
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# List of 100 prompts
prompts = [
    "A flower bouquet consisting of 9 roses, 11 orchids.",
    "A flower bouquet consisting of 14 lilies, 2 grtz.",
    "A flower bouquet consisting of 8 lilies, 11 dandelion.",
    "A flower bouquet consisting of 16 peonies, 1 tulips.",
    "A flower bouquet consisting of 8 peonies.",
    "A flower bouquet consisting of 18 roses.",
    "A flower bouquet consisting of 2 roses.",
    "A flower bouquet consisting of 1 lilies.",
    "A flower bouquet consisting of 3 dandelion.",
    "A flower bouquet consisting of 10 peonies.",
    "A flower bouquet consisting of 15 roses.",
    "A flower bouquet consisting of 3 orchids.",
    "A flower bouquet consisting of 14 dandelion.",
    "A flower bouquet consisting of 10 lisianthuses, 4 irises.",
    "A flower bouquet consisting of 10 orchids.",
    "A flower bouquet consisting of 7 dandelion.",
    "A flower bouquet consisting of 16 lilies, 4 grtz.",
    "A flower bouquet consisting of 8 dandelion.",
    "A flower bouquet consisting of 8 peonies.",
    "A flower bouquet consisting of 11 lisianthuses, 7 grtz.",
    "A flower bouquet consisting of 7 lisianthuses, 13 tulips.",
    "A flower bouquet consisting of 4 tulips, 3 dandelion.",
    "A flower bouquet consisting of 12 grtz, 5 dandelion.",
    "A flower bouquet consisting of 16 orchids.",
    "A flower bouquet consisting of 20 irises.",
    "A flower bouquet consisting of 7 grtz.",
    "A flower bouquet consisting of 11 tulips, 4 dandelion.",
    "A flower bouquet consisting of 1 irises.",
    "A flower bouquet consisting of 3 tulips.",
    "A flower bouquet consisting of 12 tulips.",
    "A flower bouquet consisting of 18 dandelion, 1 daisy.",
    "A flower bouquet consisting of 11 dandelion, 9 irises.",
    "A flower bouquet consisting of 17 irises.",
    "A flower bouquet consisting of 17 irises, 3 lisianthuses.",
    "A flower bouquet consisting of 9 daisy, 5 grtz.",
    "A flower bouquet consisting of 15 daisy, 4 lisianthuses.",
    "A flower bouquet consisting of 4 lilies, 9 irises.",
    "A flower bouquet consisting of 5 irises.",
    "A flower bouquet consisting of 2 irises.",
    "A flower bouquet consisting of 3 grtz, 5 roses.",
    "A flower bouquet consisting of 14 irises, 3 tulips.",
    "A flower bouquet consisting of 6 roses.",
    "A flower bouquet consisting of 19 roses, 1 lilies.",
    "A flower bouquet consisting of 9 irises, 1 daisy.",
    "A flower bouquet consisting of 19 irises, 1 lilies.",
    "A flower bouquet consisting of 10 lisianthuses.",
    "A flower bouquet consisting of 6 lilies, 7 peonies.",
    "A flower bouquet consisting of 15 orchids.",
    "A flower bouquet consisting of 5 tulips.",
    "A flower bouquet consisting of 11 roses, 7 grtz.",
    "A flower bouquet consisting of 9 peonies, 7 lisianthuses.",
    "A flower bouquet consisting of 17 irises, 1 lisianthuses.",
    "A flower bouquet consisting of 7 roses, 3 tulips.",
    "A flower bouquet consisting of 9 grtz.",
    "A flower bouquet consisting of 2 daisy.",
    "A flower bouquet consisting of 12 daisy, 4 grtz.",
    "A flower bouquet consisting of 17 lilies, 2 lisianthuses.",
    "A flower bouquet consisting of 7 lilies.",
    "A flower bouquet consisting of 20 roses.",
    "A flower bouquet consisting of 2 lisianthuses.",
    "A flower bouquet consisting of 1 tulips.",
    "A flower bouquet consisting of 18 roses.",
    "A flower bouquet consisting of 18 peonies.",
    "A flower bouquet consisting of 19 lisianthuses.",
    "A flower bouquet consisting of 9 lilies, 3 grtz.",
    "A flower bouquet consisting of 5 peonies, 9 dandelion.",
    "A flower bouquet consisting of 8 roses, 1 grtz.",
    "A flower bouquet consisting of 8 orchids.",
    "A flower bouquet consisting of 15 daisy.",
    "A flower bouquet consisting of 5 irises.",
    "A flower bouquet consisting of 20 irises.",
    "A flower bouquet consisting of 8 lilies, 1 tulips.",
    "A flower bouquet consisting of 15 tulips.",
    "A flower bouquet consisting of 6 orchids, 11 lisianthuses.",
    "A flower bouquet consisting of 4 roses, 11 lisianthuses.",
    "A flower bouquet consisting of 7 irises.",
    "A flower bouquet consisting of 17 irises.",
    "A flower bouquet consisting of 1 peonies.",
    "A flower bouquet consisting of 3 dandelion, 9 peonies.",
    "A flower bouquet consisting of 4 grtz.",
    "A flower bouquet consisting of 8 lilies, 7 grtz.",
    "A flower bouquet consisting of 12 grtz.",
    "A flower bouquet consisting of 9 peonies, 8 orchids.",
    "A flower bouquet consisting of 20 orchids.",
    "A flower bouquet consisting of 2 tulips.",
    "A flower bouquet consisting of 13 lilies.",
    "A flower bouquet consisting of 5 daisy, 12 irises.",
    "A flower bouquet consisting of 16 tulips, 3 lilies.",
    "A flower bouquet consisting of 17 lilies.",
    "A flower bouquet consisting of 2 dandelion.",
    "A flower bouquet consisting of 9 grtz, 9 orchids.",
    "A flower bouquet consisting of 13 lilies, 1 daisy.",
    "A flower bouquet consisting of 1 grtz.",
    "A flower bouquet consisting of 7 lisianthuses, 6 lilies.",
    "A flower bouquet consisting of 5 irises, 1 dandelion.",
    "A flower bouquet consisting of 7 roses.",
    "A flower bouquet consisting of 1 tulips.",
    "A flower bouquet consisting of 13 lisianthuses.",
    "A flower bouquet consisting of 3 grtz, 10 lisianthuses.",
    "A flower bouquet consisting of 15 dandelion, 2 roses."
]

# Generate and save an image for each prompt

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(f"{output_folder}/flower_bouquet_{i+1}.png")  # Save image with numbered file name
    print(f"Saved image {i+1} for prompt: {prompt}")

print("Image generation completed and saved in", output_folder)
