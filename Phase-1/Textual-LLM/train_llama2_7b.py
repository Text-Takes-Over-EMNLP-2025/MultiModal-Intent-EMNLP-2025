import os
from random import randrange
from functools import partial
import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          )

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from datasets import Dataset

number_epochs = 20

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
    print("Number of GPUs: ",n_gpus)
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
    INTRO_BLURB = "Each of the input sequences either indicate an emotion or intent expressed by the speaker. From the list of given 30 labels, identify which label best describes the emotion of the speaker in the input sequence. List of labels:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
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
    
    
    if len(sample['text']) > 150:
        sample['text'] = sample['text'][:150] 
    
    #convert sample['label] to lower case
    sample['label'] = sample['label'].lower()

    if sample['label'] == 'asking for opinions':
        sample['label'] = 'ask_opinions'
        
    elif sample['label'] == 'ask for help':
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
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {75 * trainable_params / all_param}"
    )


def fine_tune(model,
          tokenizer,
          dataset,
          lora_r,
          lora_alpha,
          lora_dropout,
          bias,
          task_type,
          per_device_train_batch_size,
          gradient_accumulation_steps,
          warmup_steps,
          max_steps,
          learning_rate,
          fp16,
          logging_steps,
          output_dir,
          optim):


    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)

    # Get LoRA module names
    target_modules = find_all_linear_names(model)

    # Create PEFT configuration for these modules and wrap the model to PEFT
    peft_config = create_peft_config(lora_r, lora_alpha, target_modules, lora_dropout, bias, task_type)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model = model,
        train_dataset = dataset,
        args = TrainingArguments(
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = warmup_steps,
            max_steps = max_steps,
            learning_rate = learning_rate,
            fp16 = fp16,
            logging_steps = logging_steps,
            output_dir = output_dir,
            optim = optim,
        ),
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False),
        
    )

    model.config.use_cache = False

    do_train = True

    # Launch training and log metrics
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    # Save model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok = True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()
    return metrics

    # The pre-trained model from the Hugging Face Hub to load and fine-tune
model_name = "meta-llama/Llama-2-7b-hf"

# Activate 4-bit precision base model loading
load_in_4bit = True

# Activate nested quantization for 4-bit base models (double quantization)
bnb_4bit_use_double_quant = True

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Compute data type for 4-bit base models
bnb_4bit_compute_dtype = torch.bfloat16

"""Finally, we will call the above functions to get `model` and `tokenizer` objects."""

bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
model, tokenizer = load_model(model_name, bnb_config)

import pandas as pd

def read_tsv_file(file_path):
    df = pd.read_csv(file_path, sep='\t')

    input_data = df['Text'].tolist()
    output_data = df['Label'].tolist()

    return input_data, output_data

# Example usage:
file_path  = 'train.tsv'  # Replace 'your_file_path_here.tsv' with the path to your .tsv file
input_data, output_data = read_tsv_file(file_path)
print(input_data[:5])  # Print the first 5 input sequences
print(output_data[:5])  # Print the first 5 ground labels



# Create a DataFrame from the merged data
df = pd.DataFrame({"text": input_data, "label": output_data})

# Create a dataset using the DataFrame
dataset = Dataset.from_pandas(df)


print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')
print(dataset[randrange(len(dataset))])

seed = 42

max_length = get_max_length(model)
preprocessed_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)


print(preprocessed_dataset)
print(preprocessed_dataset[0])
print(create_prompt_formats(dataset[randrange(len(dataset))]))


# LoRA attention dimension
lora_r = 16

# Alpha parameter for LoRA scaling
lora_alpha = 64

# Dropout probability for LoRA layers
lora_dropout = 0.1

# Bias
bias = "none"

# Task type
task_type = "CAUSAL_LM"

# Output directory where the model predictions and checkpoints will be stored
output_dir = "MIntRec2_data"

# Batch size per GPU for training
per_device_train_batch_size = 16

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Optimizer to use
optim = "paged_adamw_32bit"

# Number of training steps (overrides num_train_epochs)
max_steps = number_epochs

# Linear warmup steps from 0 to learning_rate
warmup_steps = 2

# Enable fp16/bf16 training (set bf16 to True with an A75)
fp16 = True

# Log every X updates steps
logging_steps = 1

metric = fine_tune(model,
    tokenizer,
    preprocessed_dataset,
    lora_r,
    lora_alpha,
    lora_dropout,
    bias,
    task_type,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    warmup_steps,
    max_steps,
    learning_rate,
    fp16,
    logging_steps,
    output_dir,
    optim)


print(model)
print(tokenizer)


model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map = "auto", torch_dtype = torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = "results/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok = True)
model.save_pretrained(output_merged_dir, safe_serialization = True)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)

new_model = <new_model_name>

model.push_to_hub(new_model, use_auth_token = True)
tokenizer.push_to_hub(new_model, use_auth_token = True)


