# Phase-3 Experiments: Debiasing and Evaluation

This repository contains the implementation details for **Phase-3 experiments**, where we debias the dataset and re-run evaluations to measure the robustness of intent detection models.  

---

## 📌 Overview
In Phase-3, we use the following codebases and models:
- **SDIF-DA** (for MIntRec-2.0 experiments) → [GitHub Repo](https://github.com/JoeYing1019/SDIF-DA)  
- **MIntRec-1 & 2.0** (MulT and BERT) → [GitHub Repo (MIntRec-1)](https://github.com/thuiar/MIntRec), [GitHub Repo (MIntRec-2.0)](https://github.com/thuiar/MIntRec2.0)  
- **LLaMA-2-7B** (from Phase-1 training)  

---

## ⚙️ Debiasing Procedure
1. **Inference:**  
   Run inference on the dataset using all three models:
   - LLaMA-2-7B (Phase-1 checkpoint)  
   - MulT / SDIF (from MIntRec authors)  
   - BERT (from MIntRec authors)  

2. **Agreement Filtering:**  
   - Identify queries correctly classified by **≥2 models**.  
   - Remove these queries from the dataset.  
   - The resulting subset is the **Debiased Dataset**.  

---

## 🔁 Phase-1 Re-evaluation
After obtaining the debiased dataset:
1. Re-run **Phase-1 evaluations and experiments** on the debiased dataset.  
2. Compare model performance on:  
   - **Original dataset** (Phase-1 results)  
   - **Debiased dataset** (Phase-3 results)  

This helps measure how dataset bias affects model performance.

---

## 🚀 How to Run
1. Clone required codebases:
   ```bash
   git clone https://github.com/thuiar/MIntRec
   git clone https://github.com/thuiar/MIntRec2.0
   git clone https://github.com/JoeYing1019/SDIF-DA
    ```

2. Run inference with each model on the dataset.
3. Use provided filtering scripts to generate the debiased dataset.
4. Re-run Phase-1 experiments on the debiased dataset.

---

## 📊 Expected Outcomes

* Debiased dataset reduces **textual bias** present in MIntRec-style datasets.
* Phase-3 results provide a fairer benchmark for comparing multimodal and textual models.

---

## 📜 References

* [MIntRec-1](https://github.com/thuiar/MIntRec)
* [MIntRec-2.0](https://github.com/thuiar/MIntRec2.0)
* [SDIF-DA](https://github.com/JoeYing1019/SDIF-DA)

