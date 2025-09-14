import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
import logging
import av
import numpy as np
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_file', help='Path to the TSV file containing video metadata and text.', required=True)
    parser.add_argument('--video_dir', help='Base directory containing video files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    return parser.parse_args()

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def find_video_path(base_dir, season, episode, clip, formats):
    for fmt in formats:
        video_path = os.path.join(base_dir, season, episode, f"{clip}{fmt}")
        if os.path.exists(video_path):
            return video_path
    return None

def read_video_pyav(container, indices):
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

def run_evaluation(args):
    logging.info("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf").to(device)
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    logging.info("Model initialized successfully.")

    logging.info(f"Loading TSV file from {args.tsv_file}...")
    data = pd.read_csv(args.tsv_file, sep='\t')
    logging.info("TSV file loaded successfully.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []
    video_formats = ['.mp4']

    logging.info("Starting evaluation...")
    for index, row in tqdm(data.iterrows(), total=len(data)):
        season = row['season']
        episode = row['episode']
        clip = row['clip']
        orig_text = row['text']
        text = f"Based on the above video wherein a speaker says the words: '{orig_text}' Your task is to classify these words based on possible emotion/intent from the list of 20 labels that best describes the speakers intent. List of Labels: \n"
        class_names = [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask for help' 
        ]
        for class_name in sorted(class_names):
            text += f"{class_name}\n"
        text += "Which of the labels do you feel best describes the intent of the speaker. generate a reasoning for the same too."
        
        prompt = f"USER: <video>{text} ASSISTANT:"

        sample_set = {'index': index, 'season': season, 'episode': episode, 'clip': clip, 'text': orig_text}

        video_path = find_video_path(args.video_dir, season, episode, clip, video_formats)

        if video_path:
            try:
                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                clip = read_video_pyav(container, indices)

                inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(device)
                generate_ids = model.generate(**inputs, max_length=2500)
                output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                output = output.split("ASSISTANT:")[1].strip()
                
                sample_set['pred'] = output
                output_list.append(sample_set)
                logging.info(f"Processed video {video_path} successfully.")
            except Exception as e:
                logging.error(f"Error processing video file '{season}/{episode}/{clip}.mp4': {e}")
        else:
            logging.warning(f"Video file '{season}/{episode}/{clip}.mp4' not found.")

    output_file_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    with open(output_file_path, 'w') as file:
        json.dump(output_list, file, indent=4)
    logging.info(f"Results saved to {output_file_path}")

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
