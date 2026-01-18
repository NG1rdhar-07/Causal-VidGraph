import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import av
import numpy as np
import json
import os
import argparse

# --- CONFIGURATION (Cluster Settings) ---
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf" 
DEVICE = "cuda" # This requires the Tesla V100

def read_video_pyav(container, indices):
    """
    Decodes specific frames from the video file.
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def run_inference(video_path, output_path):
    print(f"Loading Model: {MODEL_ID}...")
    
    # 1. Load Model (Heavy Step - GPU Only)
    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(DEVICE)
    
    # 2. Load Video
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    # Sample 8 frames uniformly
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video_clip = read_video_pyav(container, indices)
    
    # 3. Formulate Prompt
    # We ask for the JSON structure directly
    prompt = """
    [INST] <video>
    Analyze this video and output a JSON description of the events. 
    Include 'objects' (list of items) and 'actions' (list of movements) for each segment.
    Format: [{"frame_id": 0, "description": "...", "objects": [], "actions": []}]
    [/INST]
    """
    
    # 4. Generate
    inputs = processor(text=prompt, videos=video_clip, return_tensors="pt").to(DEVICE)
    
    print("Generating description...")
    out = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(out, skip_special_tokens=True)[0]
    
    # 5. Save Output
    # Basic cleanup to extract just the JSON part would happen here
    result = response.split("[/INST]")[-1].strip()
    
    with open(output_path, 'w') as f:
        f.write(result)
    print(f"Saved to {output_path}")



# yeh code ko change krna pdega !!!...there must be some error !!!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    args = parser.parse_args()
    
    try:
        run_inference(args.video_path, args.output_json)
    except Exception as e:
        print(f"Error: {e}")
        # On laptop, this will crash. On Cluster, it will work.