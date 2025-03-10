
import matplotlib.pyplot as plt
import pandas as pd
import openslide
from openslide.deepzoom import DeepZoomGenerator
import torch
import os
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
import cv2
import json
import concurrent.futures
import sys
import torch.nn as nn
sys.path.append('./')
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch
import os
from PIL import Image
from pathlib import Path
tokenizer = get_tokenizer()

classes = ['background',   
            'Low-Grade Urothelial Carcinoma', 
           'High-Grade Urothelial Carcinoma',
           'Urothelial Carcinoma']
prompts = [ 'an H&E image of background',
           'an H&E image of Low-Grade Urothelial Carcinoma', 
           'an H&E image of High-Grade Urothelial Carcinoma',
           'an H&E image of Urothelial Carcinoma'] 


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, image_embed, text_embed):
       
       
        if len(image_embed.shape) == 2:
            image_embed = image_embed.unsqueeze(0)  # [1, batch, dim]
        if len(text_embed.shape) == 2:
            text_embed = text_embed.unsqueeze(0)  # [1, batch, dim]
            
     
        image_embed = image_embed.permute(1, 0, 2)  # [batch, 1, dim] -> [1, batch, dim]
        text_embed = text_embed.permute(1, 0, 2)    # [batch, n, dim] -> [n, batch, dim]
        
        out, _ = self.attn(image_embed, text_embed, text_embed)
        return out.squeeze(0) 


# show all jupyter output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="./conch/pytorch_model.bin")
_ = model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
    


def save_normalized_patches(npy_file, he_file, output_dir):
   
    npy_name = os.path.splitext(os.path.basename(npy_file))[0]
    file_output_dir = output_dir 
    save_path = os.path.join(file_output_dir, f'{npy_name}.npy')
    if os.path.exists(save_path):   
        print(f"Skipping {npy_file} as it already exists.")
        return  
    try:
        results_map = np.load(npy_file)
        class_indices = np.argmax(results_map, axis=2)
        zero_indices = np.where(class_indices == 0)
        y_indices = zero_indices[0]
        x_indices = zero_indices[1]
        zero_probs = results_map[zero_indices[0], zero_indices[1], 0]
        sorted_idx = np.argsort(-zero_probs)
        y_indices = zero_indices[0][sorted_idx]
        x_indices = zero_indices[1][sorted_idx]
        
        final_coords =np.column_stack((y_indices, x_indices))
        print(len(final_coords))
        final_coords = final_coords.tolist()    
        slide = openslide.OpenSlide(he_file)
        tile = DeepZoomGenerator(slide, tile_size=128, overlap=160, limit_bounds=False)
        
        score_array = np.zeros((results_map.shape[0], results_map.shape[1], len(prompts)), dtype=np.float32)
        score_array[:,:,0] = 1

        cross_attn = CrossAttention(embed_dim=512).to(device) 

        for y, x in final_coords:
            try:
                img = tile.get_tile(tile.level_count - 2, (x, y))
                # img = np.array(img)          
                image_tensor = preprocess(img).unsqueeze(0).to(device)   
                with torch.inference_mode():
                        image_embedings = model.encode_image(image_tensor)
                        text_embedings = model.encode_text(tokenized_prompts)
                        img_emb_reshaped = image_embedings.unsqueeze(0)
                        txt_emb_reshaped = text_embedings.unsqueeze(0)
                        transformed_img_emb = cross_attn(img_emb_reshaped, txt_emb_reshaped).squeeze(0)
                        prob_scores = (transformed_img_emb @ text_embedings.T * model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()
                    
        
                        score_array[y, x, :] = prob_scores
            except Exception as e:
                if "Invalid address" in str(e):
                    print(f"Invalid address at coordinates (x={x}, y={y}). Skipping...")
                    continue
                else:
                    raise e
        print(f"Patches saved for {npy_file}")
     
        np.save(save_path, score_array)
        print(f"Saved scores array shape: {score_array.shape} to {save_path}")       
    except Exception as e:
        print(f"An error occurred while processing {npy_file}: {e}")

def find_corresponding_he_file(npy_file, he_dir):
    base_name = os.path.splitext(os.path.basename(npy_file))[0]
    for file in os.listdir(he_dir):       
        if base_name in file:
            return os.path.join(he_dir, file)
    return None
  
npy_dir = 'PATH/TO/org'
he_dir = 'PATH/TO/svs'
output_dir = 'PATH/TO/output'
os.makedirs(output_dir, exist_ok=True)



max_threads = 4 
with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = []
    for npy_file in os.listdir(npy_dir):
        if npy_file.endswith('.npy') :
            npy_file_path = os.path.join(npy_dir, npy_file)
            he_file_path = find_corresponding_he_file(npy_file_path, he_dir)
            if he_file_path:
              
                futures.append(executor.submit(save_normalized_patches, npy_file_path, he_file_path, output_dir))
    
   
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()  
        except Exception as e:
            print(f"Thread resulted in an exception: {e}")
