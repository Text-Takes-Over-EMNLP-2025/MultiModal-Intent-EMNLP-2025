## Evaluation of Small Multimodal Models

To evaluate small multimodal models using the MIntRec-1 and MIntRec-2 datasets, follow the steps below. This guide covers the necessary steps to clone the repositories, place the video and audio features correctly, and run the evaluation scripts for MAG-BERT and SDIF_DA models.

MIntRec-1: https://github.com/thuiar/MIntRec.git
MIntRec2.0: https://github.com/thuiar/MIntRec2.0.git

### Step 1: Clone the Repositories

1. Clone the MIntRec-1 repository:
    ```bash
    git clone <MIntRec-1_repo_url>
    cd MIntRec-1
    ```

2. Clone the MIntRec-2 repository:
    ```bash
    git clone <MIntRec-2_repo_url>
    cd MIntRec-2
    ```

### Step 2: Place Video and Audio Features

Place the `video_feats` and `audio_feats` directories in the appropriate locations within the cloned repositories. Ensure the directory structure is as follows:

#### For MIntRec-1:
```
MIntRec-1/
├── video_feats/
│   ├── video1_feat
│   ├── video2_feat
│   └── ...
├── audio_feats/
│   ├── audio1_feat
│   ├── audio2_feat
│   └── ...
├── examples/
│   ├── run_mag_bert.sh
│   └── ...
└── ...
```

#### For MIntRec-2:
```
MIntRec-2/
├── video_feats/
│   ├── video_feat.pkl
│  
├── audio_feats/
│   ├── audio_feat.pkl
│   
└── ...
```

### Step 3: Run MAG-BERT Evaluation

Navigate to the MIntRec-1 repository and execute the MAG-BERT evaluation script:

```bash
cd MIntRec-1
sh examples/run_mag_bert.sh
```

### Step 4: Run SDIF_DA Evaluation

1. Clone the SDIF_DA repository:
    ```bash
    git clone <SDIF_DA_repo_url>
    cd SDIF_DA
    ```

2. Place the necessary files in the correct locations as specified in the SDIF_DA repository's script.

3. Follow the detailed steps described in the original SDIF_DA repository to set up and run the evaluation.

### Additional Notes

- Ensure all dependencies and environment settings are correctly configured as per the requirements specified in each repository's `README` file.
- Verify the paths and filenames in the scripts to match your directory structure and file names.
- If any issues arise, refer to the troubleshooting sections or issue trackers in the respective repositories for assistance.

By following these steps, you should be able to successfully evaluate the performance of small multimodal models on the MIntRec-1 and MIntRec-2 datasets.