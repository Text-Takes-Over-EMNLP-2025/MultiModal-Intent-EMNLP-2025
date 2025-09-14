# Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection

## Overview

**MultiModal_Intent_EMNLP2025** is the official code and dataset repository for the EMNLP 2025 main conference paper:

**“Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection.”**  
📄 [Read the paper on arXiv](https://arxiv.org/abs/2508.16122v1)

This project investigates **intent recognition using multimodal data** (text, audio, and video) and presents a systematic study of **modality bias** in existing datasets and models. The repository contains datasets, experimental pipelines, and scripts required to **reproduce, analyze, and extend** the findings of the paper.

**Authors:**  
- **Ankan Mullick** — Department of Computer Science and Engineering, IIT Kharagpur, India  
- **Saransh Sharma** — Adobe Research, India  
- **Abhik Jana** — Department of Computer Science and Engineering, IIT Bhubaneswar, India  
- **Pawan Goyal** — Department of Computer Science and Engineering, IIT Kharagpur, India  

**Contact Emails:**  
- ankanm@kgpian.iitkgp.ac.in  
- sarsharma@adobe.com | saransh03sharma@gmail.com  
- abhikjana@iitbbs.ac.in  
- pawang@cse.iitkgp.ac.in  

For any issues or doubts, contact: **Saransh Sharma**  
📧 [sarsharma@adobe.com](mailto:sarsharma@adobe.com) | [saransh03sharma@gmail.com](mailto:saransh03sharma@gmail.com)

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Dataset](#dataset)
* [Directory Structure](#directory-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Citation](#citation)

---

## Features

* 💡 **Research Contribution**: Codebase for the EMNLP 2025 paper *“Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection.”*
* 📂 **Datasets Included**:

  * Original datasets from **MIntRec-1** and **MIntRec 2.0**.
  * **Debiased variants** created after Phase-3 of our study.
  * **Automated annotation variant** generated after Phase-2.
* ⚙️ **Experimental Pipelines**:

  * `Phase-1/` contains scripts for reproducing baseline experiments.
  * Instructions provided for evaluating **different categories of models** (LLMs, multimodal encoders, text-only baselines, etc.).
  * `Phase-2.md` and `Phase-3.md` contain instructions for downloading the codebases for Phase-2 and Phase-3 experiments and reproducing the results.

---

## Dataset

The `Dataset/` folder provides:

* **Original**: Original dataset files provided by the authors of MIntRec-1 and 2.0. (Kindly follow their GitHub repositories for downloading raw `.mp4` video files and audio feature `.pkl` files.)
* **Debiased**: Debiased variant of the original datasets. This is obtained by carefully filtering as part of Phase-3 of our study. We encourage all future researchers to also test their proposed solutions on debiased variants to ensure their models perform well and are not simply text-based models in disguise.
* **Phase-2-Automated\_Annotation**: We provide the raw CSVs of the 7 variants of masking used during Phase-2 of our study. Each CSV contains labels predicted by the corresponding masking variant (e.g., `sdif_bert_text.csv` means that apart from text, both video and audio are masked), along with the output probability corresponding to the ground label.
* **Human-Annotation**: We also provide human annotated dataset files for future exploration in training of routers.
---

## Directory Structure

```
MultiModal_Intent_EMNLP2025/
│
├── Dataset/             # All dataset variants (original, debiased, annotated)
├── Phase-1/             # Scripts and configs for Phase-1 experiments
├── requirements.txt     # Python dependencies
├── Phase-2.md           # Phase-2 reproduction instructions
├── Phase-3.md           # Phase-3 reproduction instructions
├── Wordclouds/          # Wordcloud analysis for a few more intent labels
├── README.md            # Documentation (this file)
└── ...
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Text-Takes-Over-EMNLP-2025/MultiModal-Intent-EMNLP-2025.git
   cd MultiModal-Intent-EMNLP-2025
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Instructions to replicate experiments for Phase-1, Phase-2, and Phase-3 are provided in their respective folders and markdown files.

> More detailed instructions can be found inside each phase directory.

We acknowledge the repositories of:

* [MIntRec](https://github.com/thuiar/MIntRec)
* [MIntRec 2.0](https://github.com/thuiar/MIntRec2.0)
* [SDIF-DA](https://github.com/JoeYing1019/SDIF-DA)
* [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
* [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
* [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)

---

## Citation

If you use this repository, please cite our EMNLP 2025 paper:

```bibtex
@misc{mullick2025texttakesoverstudy,
      title={Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection}, 
      author={Ankan Mullick and Saransh Sharma and Abhik Jana and Pawan Goyal},
      year={2025},
      eprint={2508.16122},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.16122}, 
}
```
