To get started, we'll need to follow these steps for setting up and running the inferencing scripts for the three multimodal LLM projects: Video-ChatGPT, Video-LLaVA, and Video-LLaMA. Here's a detailed guide:

1. **Clone the Original GitHub Repo**:
    - First, we need to clone the repository containing the scripts and setup instructions.
  
2. **Install Required Dependencies**:
    - Each project might have specific dependencies. We will install them as per the instructions in their respective `README` files or equivalent documentation.

3. **Place Dataset and Raw Videos**:
    - Ensure that the dataset and raw videos are placed in the appropriate directories as expected by each script.

4. **Run the Inferencing Scripts**:
    - Execute the given commands for each project with the specified configurations and arguments.

Here is a step-by-step process:

### Step 1: Clone the Repository

```bash
git clone <original_repo_url>
cd <cloned_repo_directory>
```

Replace `<original_repo_url>` with the actual URL of the GitHub repository.

### Step 2: Install Required Dependencies

Navigate to each of the project folders (Video-ChatGPT, Video-LLaVA, Video-LLaMA) and follow the installation steps mentioned in their respective documentation. Usually, this involves:

```bash
cd Video-ChatGPT
pip install -r requirements.txt

cd ../Video-LLaVA
pip install -r requirements.txt

cd ../Video-LLaMA
pip install -r requirements.txt
```

If there are additional setup steps (e.g., installing specific libraries or setting up environment variables), ensure you follow those as well.

### Step 3: Place Dataset and Raw Videos

Assuming your dataset and videos are in a folder called `MIntRec-1`, the structure should look like this:

```
<cloned_repo_directory>/
├── MIntRec-1/
│   ├── train.tsv
│   ├── test.tsv
│   └── video/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
├── Video-ChatGPT/
│   ├── multimodal_inferencing.py
│   ├── ...
├── Video-LLaVA/
│   ├── multimodal_inferencing.py
│   ├── ...
└── Video-LLaMA/
    ├── multimodal_inferencing.py
    ├── ...
```

Ensure the videos and tsv files are correctly placed in the `MIntRec-1` directory.

### Step 4: Run the Inferencing Scripts

Navigate to each project folder and run the respective commands:

#### For Video-LLaMA:
```bash
cd Video-LLaMA
python multimodal_inferencing.py --cfg-path eval_configs/video_llama_eval_withaudio.yaml --model_type llama_v2 --gpu-id 0
```

#### For Video-LLaVA:
```bash
cd ../Video-LLaVA
python -W "ignore" multimodal_inferencing.py --tsv_file /MIntRec-1/train.tsv --video_dir /MIntRec-1/video/ --output_dir Video-LLaVA/ --output_name train_res
```

#### For Video-ChatGPT:
```bash
cd ../Video-ChatGPT
python -W "ignore" multimodal_inferencing.py --tsv_file MIntRec-1/test.tsv --video_dir MIntRec-1/video/ --output_dir Video-ChatGPT/ --output_name result --model-name Video-ChatGPT/LLaVA-7B-Lightening-v1-1 --projection_path Video-ChatGPT/video_chatgpt-7B.bin
```

### Additional Notes

- Make sure your environment has the necessary computational resources (e.g., GPU) as required by these scripts.
- Verify the paths in the commands to ensure they are correct relative to your directory structure.
- If any errors occur, check the log output for clues and refer to the project documentation or issue tracker for solutions.To get started, we'll need to follow these steps for setting up and running the inferencing scripts for the three multimodal LLM projects: Video-ChatGPT, Video-LLaVA, and Video-LLaMA. Here's a detailed guide:

1. **Clone the Original GitHub Repo**:
    - First, we need to clone the repository containing the scripts and setup instructions.
  
2. **Install Required Dependencies**:
    - Each project might have specific dependencies. We will install them as per the instructions in their respective `README` files or equivalent documentation.

3. **Place Dataset and Raw Videos**:
    - Ensure that the dataset and raw videos are placed in the appropriate directories as expected by each script.

4. **Run the Inferencing Scripts**:
    - Execute the given commands for each project with the specified configurations and arguments.

Here is a step-by-step process:

### Step 1: Clone the Repository

```bash
git clone <original_repo_url>
cd <cloned_repo_directory>
```

Replace `<original_repo_url>` with the actual URL of the GitHub repository.

### Step 2: Install Required Dependencies

Navigate to each of the project folders (Video-ChatGPT, Video-LLaVA, Video-LLaMA) and follow the installation steps mentioned in their respective documentation. Usually, this involves:

```bash
cd Video-ChatGPT
pip install -r requirements.txt

cd ../Video-LLaVA
pip install -r requirements.txt

cd ../Video-LLaMA
pip install -r requirements.txt
```

If there are additional setup steps (e.g., installing specific libraries or setting up environment variables), ensure you follow those as well.

### Step 3: Place Dataset and Raw Videos

Assuming your dataset and videos are in a folder called `MIntRec-1`, the structure should look like this:

```
<cloned_repo_directory>/
├── MIntRec-1/
│   ├── train.tsv
│   ├── test.tsv
│   └── video/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
├── Video-ChatGPT/
│   ├── multimodal_inferencing.py
│   ├── ...
├── Video-LLaVA/
│   ├── multimodal_inferencing.py
│   ├── ...
└── Video-LLaMA/
    ├── multimodal_inferencing.py
    ├── ...
```

Ensure the videos and tsv files are correctly placed in the `MIntRec-1` directory.

### Step 4: Run the Inferencing Scripts

Navigate to each project folder and run the respective commands:

#### For Video-LLaMA:
```bash
cd Video-LLaMA
python multimodal_inferencing.py --cfg-path eval_configs/video_llama_eval_withaudio.yaml --model_type llama_v2 --gpu-id 0
```

#### For Video-LLaVA:
```bash
cd ../Video-LLaVA
python -W "ignore" multimodal_inferencing.py --tsv_file /MIntRec-1/train.tsv --video_dir /MIntRec-1/video/ --output_dir Video-LLaVA/ --output_name train_res
```

#### For Video-ChatGPT:
```bash
cd ../Video-ChatGPT
python -W "ignore" multimodal_inferencing.py --tsv_file MIntRec-1/test.tsv --video_dir MIntRec-1/video/ --output_dir Video-ChatGPT/ --output_name result --model-name Video-ChatGPT/LLaVA-7B-Lightening-v1-1 --projection_path Video-ChatGPT/video_chatgpt-7B.bin
```

### Additional Notes

- Make sure your environment has the necessary computational resources (e.g., GPU) as required by these scripts.
- Verify the paths in the commands to ensure they are correct relative to your directory structure.
- If any errors occur, check the log output for clues and refer to the project documentation or issue tracker for solutions.
