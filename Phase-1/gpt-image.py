from openai import OpenAI
import os
import base64
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import time


# Read the TSV file
tsv_file = "MIntRec-debias/MIntRec-1/test.tsv"
df = pd.read_csv(tsv_file, delimiter='\t')


client = OpenAI(api_key="")
MODEL="gpt-4o"

# Function to extract frames from a video
def extract_frames(video_path, num_frames=6):
    #check if video file exists
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return []
    vidcap = cv2.VideoCapture(video_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, length-1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, image = vidcap.read()
        if success:
            frames.append(image)
        else:
            print(f"Failed to read frame at index {idx} from {video_path}")

    vidcap.release()
    return frames

# Function to downsize and convert image to base64
def process_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((1024, 1024))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Process each row in the TSV file
for index, row in df.iterrows():
    season = row['season']
    episode = row['episode']
    clip = row['clip']
    text = row['text']
    label = row['label']

    video_path = f"raw_data/{season}/{episode}/{clip}.mp4"
    # print(text, video_path)

    # Extract and process frames
    frames = extract_frames(video_path)
    base64_frames = [process_frame(frame) for frame in frames]
    # print(len(base64_frames))
    
    
    system_prompt = "Each of the input sequences either indicate an emotion expressed by the speaker or an intent expressed by speaker. From the list of given 20 labels, identify which label best describes the emotion or intent of the speaker in the input sequence. Only output the label from the list below. List of labels:" 
    class_names = [ 'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask_help']
    for class_name in sorted(class_names):
        system_prompt += f"{class_name}\n" 
    # print(system_prompt)

    messages_content = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": f"Input: {text}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            'url': f"data:image/jpeg;base64,{base64_frames[0]}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            'url': f"data:image/jpeg;base64,{base64_frames[1]}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            'url': f"data:image/jpeg;base64,{base64_frames[2]}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            'url': f"data:image/jpeg;base64,{base64_frames[3]}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            'url': f"data:image/jpeg;base64,{base64_frames[4]}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            'url': f"data:image/jpeg;base64,{base64_frames[5]}"
                        }
                    }
                    
                ]
            }
        ]

    try:
        response = client.chat.completions.create(
        model=MODEL,
        messages=messages_content,
        max_tokens=10,
        temperature=0.0,
        )
        print(index, "\t", response.choices[0].message.content)
        print("\n-------------------\n")

    except Exception as e:
        time.sleep(60)