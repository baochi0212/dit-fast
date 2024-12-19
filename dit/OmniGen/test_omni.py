from OmniGen import OmniGenPipeline
import torch
pipe = OmniGenPipeline.from_pretrained("./OmniGen-v1")
# Note: Your local model path is also acceptable, such as 'pipe = OmniGenPipeline.from_pretrained(your_local_model_path)', where all files in your_local_model_path should be organized as https://huggingface.co/Shitao/OmniGen-v1/tree/main

#
# # Text to Image
# images = pipe(
#     prompt="A curly-haired man, Chinese in a blue shirt with VinAI words and Brain-shaped logo.",
#     height=1024,
#     width=1024,
#     guidance_scale=2.5,
#     seed=0,
# )
# images[0].save("./test_img/example_t2i.png")  # save output PIL Image
#
# Multi-modal to Image
# In the prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
while True:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # query = input("Query: ")
        # query = "A boy taking picture with a shiba dog. A boy is in <img><| image_1 | > < /img > . A shiba dog is in < img > < | image_2| > </img > ."
        images = pipe(
            # prompt="A full body image of a man who  is enjoying with curly-haired style and a blue shirt with 'FPT' words in the middle, realistic style and high resolution, crowd in the background. A man is in <img><|image_1|></img>.",
            prompt="An asian doctor with exact face of the man in <img><|image_1|></img>, doctor outfit, seriously posing with crowd in background. The background is vivid and realistic.",
            # prompt=query,
            input_images=["./test_img/chitb.jpg"],
            height=1024,
            width=1024,
            guidance_scale=2.5,
            img_guidance_scale=1.6,
            num_inference_steps=50,
            seed=0
        )
        # save output PIL image
        images[0].save("./test_img/example_face_swap.png")
        break
