import numpy as np
import torch
import os
import json
import io
import cv2
import time
import albumentations as A

from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from PIL import Image, ImageFilter

model_id_inpainting = "stabilityai/stable-diffusion-2-inpainting"
model_id_t2i = "stabilityai/stable-diffusion-2-1"
sam_model_name = "facebook/sam-vit-base"

cuda = "cuda"
cpu = "cpu"

# Initialize the processor with the pre-trained model
sam_processor = SamProcessor.from_pretrained(sam_model_name)
sam_model = SamModel.from_pretrained(sam_model_name)
sam_model = sam_model.to(cpu)

print("SAM loaded")

stable_inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id_inpainting,revision="fp16",torch_dtype=torch.float16,)
stable_inpainting_pipe = stable_inpainting_pipe.to(cpu)

stable_text_to_image_pipe = StableDiffusionPipeline.from_pretrained(model_id_t2i, revision='fp16', torch_dtype=torch.float16)
stable_text_to_image_pipe.to(cpu)

print("Stable diffusion loaded")

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def expand2square(pil_img, background_color=(255,255,255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def get_mask(image):
    global sam_model
    result = []
    input_points = [[[np.array(image.size)/2]]]

    input_data = sam_processor(image, input_points=input_points, return_tensors="pt")
    input_data = input_data.to(cuda)
    sam_model = sam_model.to(cuda)

    with torch.no_grad():
        prediction_output = sam_model(**input_data)
        
        prediction_output = sam_processor.image_processor.post_process_masks(
            prediction_output.pred_masks.cpu(), 
            input_data["original_sizes"].cpu(), 
            input_data["reshaped_input_sizes"].cpu()
        )
        
        if torch.cuda.is_available():
            # Empty the GPU cache and collect garbage
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    masks = np.transpose(prediction_output[0][0, :, :, :].numpy(), [1, 2, 0]).astype(np.uint8) * 255

    sam_model = sam_model.to(cpu)
    input_data = input_data.to(cpu)

    return Image.fromarray(masks)

def model_fn(model_dir):
    """
    Load the pre-trained model from the specified directory.

    Args:
        model_dir (str): Directory containing the pre-trained model files.

    Returns:
        model: Loaded pre-trained model.
    """
    print("Executing model_fn from inference.py ...")
    # env = os.environ
    # sam_model = SamModel.from_pretrained(sam_model_name)
    # sam_model = sam_model.to(cpu)

    # print("SAM loaded")

    return None

def input_fn(request_body, request_content_type):
    """
    Preprocess the input data.

    Args:
        request_body: Input data from the request.
        request_content_type (str): Content type of the request.

    Returns:
        inputs: Preprocessed input data.
    """
    print("Executing input_fn from inference.py ...")
    inputs = {}
    if request_content_type:
        # Load image array from request body
        # img_array = np.load(io.BytesIO(request_body), allow_pickle=True)
        # img = Image.fromarray(img_array)

        img_array = np.uint8(np.array(request_body["image"]))
        img = Image.fromarray(img_array).convert('RGB').resize((512, 512))
        img = expand2square(img)
        init_image = img.resize((512, 512))
        inputs["image"] = init_image

        # Preprocess the image using the processor
        mask_image = get_mask(init_image)
        # Convert the PIL.Image object to a numpy array
        mask_array = np.array(mask_image)

        # Find the indices where the colors are not white
        non_black_indices = np.where(np.any(mask_array != 0, axis=-1))

        # Set the non-white spots to white and everything else to black
        mask_array[non_black_indices] = [255, 255, 255]

        converted_mask_image = Image.fromarray(mask_array)
        anti_mask = Image.fromarray(255 - np.array(converted_mask_image))#.filter(ImageFilter.GaussianBlur(8))
        inputs["mask"] = anti_mask

        inputs["prompt"] = request_body["prompt"]

        # inputs["prompt_fr"] = request_body["prompt_fr"]
        # inputs["prompt_bg"] = request_body["prompt_bg"]
        # inputs["negative_prompt"] = request_body["negative_prompt"]

    else:
        raise Exception("Unsupported content type: " + request_content_type)
    return inputs
    
def predict_fn(input_data, model):
    """
    Perform the prediction using the input data and the loaded model.

    Args:
        input_data: Preprocessed input data.
        model: Loaded pre-trained model.

    Returns:
        result: Prediction output.
    """
    print("Executing predict_fn from inference.py ...")
    result = []
    with torch.no_grad():
        # Perform the prediction using the model
        result = sam_model(**input_data)

        # Post-process the predicted masks
        result = sam_processor.image_processor.post_process_masks(result.pred_masks.cpu(), input_data["original_sizes"].cpu(), input_data["reshaped_input_sizes"].cpu())

        if torch.cuda.is_available():
            # Empty the GPU cache and collect garbage
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    return result
        
def output_fn(prediction_output, content_type):
    """
    Process the prediction output and prepare the response.

    Args:
        prediction_output: Prediction output.
        content_type (str): Desired content type for the response.

    Returns:
        str: Response in the specified content type.
    """
    print("Executing output_fn from inference.py ...")
    masks = np.transpose(prediction_output[0][0, :, :, :].numpy(), [1, 2, 0]).astype(np.uint8) * 255
    mask_list = masks.tolist()
    return json.dumps(mask_list)