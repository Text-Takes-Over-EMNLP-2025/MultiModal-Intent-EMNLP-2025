import os
import base64
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import anthropic

# Read the TSV file
tsv_file = "MIntRec-1/test.tsv"
df = pd.read_csv(tsv_file, delimiter='\t')

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="",
)

# Function to extract frames from a video
def extract_frames(video_path, num_frames=4):
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
    image = image.resize((512, 512))
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

    video_path = f"MIntRec-1/raw_data/{season}/{episode}/{clip}.mp4"
    
    # Extract and process frames
    frames = extract_frames(video_path)
    base64_frames = [process_frame(frame) for frame in frames]

    messages_content = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Frame {1}:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_frames[0],
                        },
                    },
                    {"type": "text", "text": f"Frame {2}:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_frames[1],
                        },
                    },
                    {"type": "text", "text": f"Frame {3}:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_frames[2],
                        },
                    },
                    {"type": "text", "text": f"Frame {4}:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_frames[3],
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Input: {text}"
                    }
                ]
            }
        ]

    # Send message to Claude
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=80,
        system="""Each of the input sequences either indicate an emotion expressed by the speaker or an intent expressed by speaker. From the list of given 20 labels, identify which label best describes the emotion or intent of the speaker in the input sequence. Only generate the label. List of labels: \nadvise\nagree\napologise\narrange\nask_help\ncare\ncomfort\ncomplain\ncriticize\nflaunt\ngreet\ninform\nintroduce\njoke\nleave\noppose\npraise\nprevent\ntaunt\nthank""",        
        messages=messages_content,
    )
    print(index, "\t", message.content[0].text)
    print("\n-------------------------------------------------------\n")
#     print(f"Message sent for Season {season}, Episode {episode}, Clip {clip}")

# print("All messages sent.")
