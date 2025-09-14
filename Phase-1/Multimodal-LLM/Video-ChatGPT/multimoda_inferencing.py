import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--tsv_file', help='Path to the TSV file containing video metadata and text.', required=True)
    parser.add_argument('--video_dir', help='Base directory containing video files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)

    return parser.parse_args()

def find_video_path(base_dir, season, episode, clip, formats):
    """
    Find the path to the video file with the given metadata.

    Args:
        base_dir (str): Base directory containing video files.
        season (str): Season of the video.
        episode (str): Episode of the video.
        clip (str): Clip number of the video.
        formats (list): List of possible video formats.

    Returns:
        str: Path to the video file if found, None otherwise.
    """
    for fmt in formats:
        video_path = os.path.join(base_dir, season, episode, f"{clip}{fmt}")
        print(video_path)
        if os.path.exists(video_path):
            return video_path
    return None

def run_evaluation(args):
    """
    Run evaluation on the dataset using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # Load the TSV file
    data = pd.read_csv(args.tsv_file, sep='\t')

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4']

    # Iterate over each row in the TSV file
    for index, row in tqdm(data.iterrows(), total=len(data)):
        season = row['season']
        episode = row['episode']
        clip = row['clip']
        text = row['text']
        orig_text = text
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
        label = row['label']

        sample_set = {'index': index, 'season': season, 'episode': episode, 'clip': clip, 'text': orig_text, 'label': label}

        # Find the video file path
        video_path = find_video_path(args.video_dir, season, episode, clip, video_formats)

        if video_path:
            try:
                # Load the video file
                video_frames = load_video(video_path)
                # Run inference on the video and add the output to the list
                output = video_chatgpt_infer(video_frames, text, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
                sample_set['pred'] = output
                output_list.append(sample_set)
            except Exception as e:
                print(f"Error processing video file '{season}/{episode}/{clip}.mp4': {e}")
        else:
            print(f"Video file '{season}/{episode}/{clip}.mp4' not found.")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file, indent=4)

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
