import os
from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)

import bitsandbytes as bnb
from peft import LoraConfig
import pandas as pd
import random
number_epoch = 20


def create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

    return bnb_config

def load_model(model_name, bnb_config):

    # Get number of GPU device and set maximum memory
    n_gpus = torch.cuda.device_count()
    max_memory = f'{81920}MB'

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto", # dispatch the model efficiently on the available resources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )

    # Load model tokenizer with the user authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)

    # Set padding token as EOS token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_prompt_formats(sample):
    # Read class names from a file

   
    # Initialize static strings for the prompt template
    INTRO_BLURB = "Each of the input sequences either indicate an emotion or intent expressed by the speaker. From the list of given 20 labels, identify which label best describes the emotion of the speaker in the input sequence. List of labels:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    blurb = f"{INTRO_BLURB}\n"
    
    class_names = ['Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask_help']
    
    for class_name in sorted(class_names):
        #convert class_name to lower case
        class_name = class_name.lower()
        blurb += f"{class_name}\n"  # Include class names with their indices
    
    #truncate sample text to 150 if length is greater than 150
    if len(sample['text']) > 150:
        sample['text'] = sample['text'][:150]
    
    #convert label to lowercase
    sample['label'] = sample['label'].lower()
    if sample['label'] == 'ask for help':
        sample['label'] = 'ask_help'
    
    

    input_context = f"{INPUT_KEY}\n{sample['text']}" if sample["text"] else None
    response = f"{RESPONSE_KEY}\n{sample['label']}"  # Use label index instead of label name
    end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts = [part for part in [blurb, input_context, response, end] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)

    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt

    return sample


def get_max_length(model):

    # Pull model configuration
    conf = model.config
    # Initialize a "max_length" variable to store maximum sequence length as null
    max_length = None
    # Find maximum sequence length in the model configuration and save it in "max_length" if found
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    # Set "max_length" to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = ["text", "label"],
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed = seed)

    return dataset

seed = 42


def create_peft_config(r, lora_alpha, target_modules, lora_dropout, bias, task_type):

    config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
    )

    return config

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit = False):

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )



# Define the model and tokenizer
model_name = <model name>

load_in_4bit = True
bnb_4bit_use_double_quant = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.bfloat16


bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
model, tokenizer = load_model(model_name, bnb_config)



def evaluate_on_dataset(input_data, output_data, model, tokenizer):
    predictions = []
    ground_truth = []
    sequences = []

    for i in range(len(input_data)):
        sequence = input_data[i]
        ground_label = output_data[i]



        blurb = ""
        # Create the formatted prompt template
        # INTRO_BLURB = "From the given set of labels, identify the underlying emotion or intent of the speaker in the Input sequence: "
        INTRO_BLURB = "Each of the input sequences either indicate an emotion or intent expressed by the speaker. From the list of given 30 labels, identify which label best describes the emotion of the speaker in the input sequence. List of labels:"
        INPUT_KEY = "Input: "
        END_KEY = "### End"

        blurb = f"{INTRO_BLURB}\n"
        class_names = [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange', 
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain', 
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain', 
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite', 
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise', 
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
        ]
        for class_name in sorted(class_names):
            #convert class_name to lower case
            class_name = class_name.lower()
            if class_name == 'asking for opinions':
                class_name = 'ask_opinions'
            if class_name == 'ask for help':
                class_name = 'ask_help'
            
            blurb += f"{class_name}\n"  # Include class names with their indices
        
        #truncate sample text to 150 if length is greater than 150
        if len(sequence) > 150:
            sequence = sequence[:150]
        
        #convert label to lowercase
        ground_label = ground_label.lower()
        if ground_label == 'asking for opinions':
            ground_label = 'ask_opinions'
        
        if ground_label == 'ask for help':
            ground_label = 'ask_help'
        
        
        input_context = f"{INPUT_KEY} {sequence}"
        end = f"{END_KEY}"

        parts = [part for part in [blurb, input_context, end] if part]
        formatted_prompt = "\n\n".join(parts)
        #print(formatted_prompt)

        # Generate a response
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
        input_ids = input_ids.to('cuda')
        # print(formatted_prompt)
        len_pr = len(formatted_prompt)
        output = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id,max_length=len_pr+10, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)


        class_names = [class_name.lower() for class_name in class_names]
        response = None
        response_part = generated_text.split(input_context)
        if len(response_part) > 1:
            for element in response_part[1].split():
                if element in class_names:
                    response = element
                    break
                
        if response is None or response not in class_names:
            response_part = generated_text.split("### Response:")
            if len(response_part) > 1:
                response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
                response = response.split('\n')[0].strip()
        
        if response is None or response not in class_names:
            response_part = generated_text.split("response:")
            if len(response_part) > 1:
                response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
                response = response.split('\n')[0].strip()
        
        
        if response is None or response not in class_names:
            response_part = generated_text.split("Response:")
            if len(response_part) > 1:
                response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
                response = response.split('\n')[0].strip()
        
        if response is None or response not in class_names:
            response_part = generated_text.split("### Ending label:")
            if len(response_part) > 1:
                response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
                response = response.split('\n')[0].strip()
                
            #iterate through all words in the response and find the first word that is in class_names
            if response is not None and response not in class_name:
                for element in response.split():
                    if element in class_names:
                        response = element
                        break
                
                
        if response is None or response not in class_names:
            response_part = generated_text.split("Ending response:")
            if len(response_part) > 1:
                response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
                response = response.split('\n')[0].strip()
        
        if response is None or response not in class_names:
          response_part = generated_text.split("End:")
          if len(response_part) > 1:
            response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
            response = response.split('\n')[0].strip()
        
        if response is None:
          response_part = generated_text.split("Ending:")
          if len(response_part) > 1:
            response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
            response = response.split('\n')[0].strip()
        
        if response is None or response not in class_names:
          response_part = generated_text.split("End")
          if len(response_part) > 1:
            response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
            response = response.split('\n')[0].strip()
        
        if response is None or response not in class_names:
          response_part = generated_text.split("end")
          if len(response_part) > 1:
            response = response_part[1].strip().replace(" ", "").split('#')[0].strip()
            response = response.split('\n')[0].strip()
        
        
        if response is None or response not in class_names:
            str1 = input_context
            response_part = response.split(str1)
            if len(response_part) > 1:
                response = response_part[1].strip()
                #iterate through all words in the response and find the first word that is in class_names
                for element in response.split():
                    if element in class_names:
                        response = element
                        break
                    
        if response is None or response not in class_names:
            label = None
            for element in response_part:
                if "End" in element:
                    index = response_part.index(element)
                    if index < len(response_part) - 1:
                        label = response_part[index + 1].strip()
                        #remove whatever comes after space
                        label = label.split(' ')[0]
                        label = label.split('\n')[0]
                        break

            if label is not None:
                response = label
            
        
        response = response.lower()
        predictions.append(response)
        ground_truth.append(ground_label)
        sequences.append(sequence)
        print(i+1,sequence, response, ground_label)

    return sequences,predictions, ground_truth


def read_tsv_file(file_path):
    df = pd.read_csv(file_path, sep='\t')

    input_data = df['Text'].tolist()
    output_data = df['Label'].tolist()

    return input_data, output_data


file_path = 'test.tsv'  # Replace 'your_file_path_here.tsv' with the path to your .tsv file
input_data, output_data = read_tsv_file(file_path)

sequence,predictions, ground_truth = evaluate_on_dataset(input_data, output_data, model, tokenizer)
c_results = pd.DataFrame({'Input_Sequence': sequence, 'Actual_Label': ground_truth, 'Predicted_Label': predictions})
c_results.to_csv('./test/mintrec2-'+'.csv', index=False)

print("Results saved to CSV file.")