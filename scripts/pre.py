
import os, torch
import gradio as gr
import numpy as np
import pdb

from PIL import Image

MAX_COLORS = 12

def clear_prompts():
    return ""
def default_setting():
    # print(creg)
    return [50, 1.0, 0.6, 1]

def send_text_to_prompt(new_text, old_text):
    if old_text == "":  # if text on the textbox text2img or img2img is empty, return new text
        return new_text
    return old_text + ", " + new_text  # else join them together and send it to the textbox

def enable(button):
    if button == "Disable DenseNet":
        return [gr.update(value="Enable DenseNet"), gr.update(visible=False)]
    else:
        return [gr.update(value="Disable DenseNet"), gr.update(visible=True)]

def process_prompts(binary_matrixes, *seg_prompts):
    return [gr.update(visible=True), gr.update(value=', '.join(seg_prompts[:len(binary_matrixes)]))]

def create_binary_matrix(img_arr, target_color):
    mask = np.all(img_arr == target_color, axis=-1)
    binary_matrix = mask.astype(int)
    return binary_matrix

def process_sketch(masks, image=None):
    # masks is a image with white background and colored masks

    if image is not None:
        if masks.size[0] != image.size[0]:
            minsize_w = min(masks.size[0], image.size[0])
            minsize_h = min(masks.size[1], image.size[1])
            masks = masks.resize((minsize_w, minsize_h))
            image = image.resize((minsize_w, minsize_h))
        masks = np.array(masks)
        masks = masks[:,:,:3]
        image = np.array(image)
        image = image[:,:,:3]
        bgmask = np.equal(image, masks) *255
        masks = np.clip(masks + bgmask, a_min=0, a_max=255)
        

    colors = np.unique(masks.reshape(-1, 3), axis=0, return_counts=True)
    colors_rgb = []
    for i in range(colors[1].shape[0]):
        if colors[1][i]>666:
            if 'rgb'+str(tuple(colors[0][i,:])) != 'rgb(255, 255, 255)':
                colors_rgb.append('rgb'+str(tuple(colors[0][i,:])).strip())
        else:
            continue
    # canvas_data = {'image': masks, 'colors': colors_rgb}
    binary_matrixes = []
    im2arr = masks
    
    colors = [tuple(map(int, rgb[4:-1].split(','))) for rgb in colors_rgb]
    colors_fixed = []

    r, g, b = 255, 255, 255
    binary_matrix = create_binary_matrix(im2arr, (r,g,b))

    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    for color in colors:
        r, g, b = color
        if any(c != 255 for c in (r, g, b)):
            # binary_matrix = Image.fromarray(create_binary_matrix(im2arr, (r,g,b)).astype(np.int8))
            binary_matrix = create_binary_matrix(im2arr, (r,g,b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
    
    visibilities = []
    colors = []
    
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]

    # pdb.set_trace()    
    return [gr.update(visible=True), binary_matrixes, *visibilities, *colors]

def preprocess_mask(mask_, h, w, device):
    mask = np.array(mask_)
    mask = mask.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask