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

# Process each row in the TSV file
for index, row in df.iterrows():
    text = row['text']
    label = row['label']

    
    system_prompt = "Each of the input sequences either indicate an emotion expressed by the speaker or an intent expressed by speaker. From the list of given 20 labels, identify which label best describes the emotion or intent of the speaker in the input sequence. Only output the label from the list below. List of labels:" 
    class_names = [ 'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask_help']
    for class_name in sorted(class_names):
        system_prompt += f"{class_name}\n" 
    
    sample_prompt = """Some Sample Input-Output pairs include:\n\nInput: i actually just bought some stock in a drone company.\nResponse: Inform\n\nInput: there's, like, eight of us, that's more of a walk.	\nResponse: Complain\n\nInput: sandra, code cottontail.\nResponse: Inform\n\nInput: never forget how fun you are, jonah.\nResponse: Taunt\n\nInput: you can donate to our toy drive.\nResponse: Advise\n\n"""
    # print(system_prompt)
    # print(sample_prompt)

    messages_content = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": f"Input: {sample_prompt}"
                    },
                    {
                        "type": "text", "text": f"Input: {text}\nResponse: "
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