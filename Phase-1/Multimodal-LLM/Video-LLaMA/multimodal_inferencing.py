import pandas as pd
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation
import decord
import warnings
import json

warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.ERROR)

# Initialize Decord
decord.bridge.set_bridge('torch')

# Function to parse arguments
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

# Function to set up seeds for reproducibility
def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# Function to initialize the model
def initialize_model():
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')
    return chat


def process_row(chat, row):
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
        text += f"{class_name}\n"  # Include class names with their indices
    text += "Which of the labels do you feel best describes the intent of the speaker. generate a reasoning for the same too."
    

    video_path = f'MIntRec-1/video/{season}/{episode}/{clip}.mp4'
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load the video
    video = decord.VideoReader(video_path)

    # Set up chat state
    chat_state = default_conversation.copy()
    chat_state.system = "You are able to understand the visual content that the user provides. The video has a speaker making a statement. The statement either indicate an emotion or intent expressed by the speaker. Your task is to classify the statment based on the text provided by the user along with the video and generate a detailed explanation for the same. In case you don't understand the task, generate a detailed description of the video. Directly start generate the answer"
    
    # Process the video and text
    img_list = []
    chat.upload_video(video_path, chat_state, img_list)

    # Ask the model
    chat.ask(text, chat_state)
    response = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
    
    # print(text, response)
    
    return response

# Main function to read the TSV and process each row
def main():
    # Initialize the model
    chat = initialize_model()

    # Read the TSV file
    df = pd.read_csv('MIntRec-1/test.tsv', sep='\t')

    # List to store the results
    results = []

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        try:
            response = process_row(chat, row)
            response = process_row(chat, row)
            result = {
                "index": index,
                "season": row['season'],
                "episode": row['episode'],
                "clip": row['clip'],
                "text": row['text'],
                "label": row['label'],
                "pred": response
            }
            results.append(result)
            print("="*50)
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Write the results to a JSON file
    with open('train_res.json', 'w') as f:
        json.dump(results, f, indent=4)

# Run the main function
if __name__ == "__main__":
    main()
