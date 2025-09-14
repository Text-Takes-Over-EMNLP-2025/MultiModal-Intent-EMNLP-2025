# README
This folder contains scripts and files for training various models including Llama2-7b, Llama2-13b, Llama3-8b, and Mistral-7b. The process includes adjusting the `class_names` list according to the dataset you are using and optionally adding the `model_name` if you want to push your model to the Huggingface Hub.

## Files and Directories

1. `train_llama_7b.py`
2. `train_llama_13b.py`
3. `train_llama_8b.py`
4. `train_mistral_7b.py`
5. `test.py`

## Instructions

### 1. Adjust the `class_names` List

Before running the training scripts, ensure that the `class_names` list in each script matches the classes in your dataset. This list defines the output labels for the model.

### 2. Adding `model_name`

If you intend to push your model to the Huggingface Hub, you need to specify the `model_name`. This can be added in the script where the model is defined or in the `test.py` script.

### Training Scripts

Each script is designed to train a specific model:

- `train_llama2_7b.py`: Script to train the Llama2-7b model.
- `train_llama2_13b.py`: Script to train the Llama2-13b model.
- `train_llama3_8b.py`: Script to train the Llama3-8b model.
- `train_mistral_7b.py`: Script to train the Mistral-7b model.

### Testing and Generating Output

The `test.py` script will load the trained model and create a CSV file with the output labels. This is useful for evaluating the model's performance on your test dataset.

#### Example Usage

1. **Train a Model:**
   ```bash
   python train_llama2_7b.py
   ```

2. **Test the Model and Generate CSV:**
   ```bash
   python test.py
   ```

### Pushing to Huggingface Hub

If you want to push your model to the Huggingface Hub, ensure you have set the `model_name` in the scripts and follow these steps:

1. Install the Huggingface CLI tool:
   ```bash
   pip install huggingface_hub
   ```

2. Log in to your Huggingface account:
   ```bash
   huggingface-cli login
   ```

3. Push your model to the hub

Ensure your `model_name` is unique and descriptive to avoid conflicts with existing models.

## Conclusion

By following the steps above, you can train, test, and optionally share your models with the community on Huggingface Hub. Adjust the `class_names` list according to your dataset and specify the `model_name` if needed. Use the `test.py` script to evaluate your models and generate results in a CSV format.

For any issues or contributions, please feel free to open an issue or a pull request on the repository. Happy training!