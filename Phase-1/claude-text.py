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

# Process each row in the TSV file
for index, row in df.iterrows():
    season = row['season']
    episode = row['episode']
    clip = row['clip']
    text = row['text']
    label = row['label']

    
    messages_content = [
            {
                "role": "user",
                "content": [
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
        max_tokens=50,
        system="""Each of the input sequences either indicate an emotion expressed by the speaker or an intent expressed by speaker. From the list of given 20 labels, identify which label best describes the emotion or intent of the speaker in the input sequence. List of labels: \nadvise\nagree\napologise\narrange\nask_help\ncare\ncomfort\ncomplain\ncriticize\nflaunt\ngreet\ninform\nintroduce\njoke\nleave\noppose\npraise\nprevent\ntaunt\nthank""",        
        messages=messages_content,
    )
    print(index, "\t", message.content[0].text)
    print("\n-------------------------------------------------------\n")
#     print(f"Message sent for Season {season}, Episode {episode}, Clip {clip}")

# print("All messages sent.")
