# Modality Masking Experiments with MIntRec and SDIF-DA for Phase-2

This repository extends the official codebases of **MIntRec-1**, **MIntRec-2.0**, and **SDIF-DA** for modality-masking analysis:

* [MIntRec-2.0](https://github.com/thuiar/MIntRec2.0)
* [MIntRec-1](https://github.com/thuiar/MIntRec)
* [SDIF-DA](https://github.com/JoeYing1019/SDIF-DA)

---

## 📂 Setup

1. Clone the repositories:

   ```bash
   git clone https://github.com/thuiar/MIntRec2.0
   git clone https://github.com/thuiar/MIntRec
   git clone https://github.com/JoeYing1019/SDIF-DA
   ```

2. Install the required dependencies for each repository (see their individual `requirements.txt` files).

---

## 🔧 Modality Masking

To analyze the effect of different modalities, modify the preprocessing scripts:

* `data/audio_pre.py`
* `data/video_pre.py`
* `data/text_pre.py`

In each script, **mask the feature vector of the corresponding modality with zero vectors**.
For example, to analyze the impact of text modality, replace the **text feature vectors** with zero vectors.

---

## ⚙️ Automated Annotation Protocol

Run **training + inference** under **7 modality-masking combinations**:

1. Mask **Text only**
2. Mask **Video only**
3. Mask **Audio only**
4. Mask **Video + Audio**
5. Mask **Video + Text**
6. Mask **Audio + Text**
7. Mask **No Modality**

For each case:

* Collect **predictions** and **probabilities**.
* Identify the **smallest subset of modalities** that either:

  * Correctly classifies the query, **or**
  * Assigns the maximum probability to the **ground-truth label**.

---

## 📊 Objective

This procedure allows systematic evaluation of the **contribution of each modality** and their combinations in intent recognition tasks.

