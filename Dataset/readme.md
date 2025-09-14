# Auto-Annotate & Debiased Dataset

This repository contains three main directories: `Phase-2-Automated-Annotation`, `Original` and `Debiased`. Each serves a specific purpose in data processing and annotation.

## Folder Structure

```
.
├── Phase-2-Automated-Annotation
│   ├── MIntRec-1-auto-annotate
│   │   ├── combined_intent_modality.csv
│   │   ├── meta_sdif_bert.csv
│   │   ├── sdif_bert_all.csv
│   │   ├── sdif_bert_audio.csv
│   │   ├── sdif_bert_audio_video.csv
│   │   ├── sdif_bert_text.csv
│   │   ├── sdif_bert_text_audio.csv
│   │   ├── sdif_bert_text_video.csv
│   │   ├── sdif_bert_video.csv
│   │   ├── test.tsv
│   ├── MIntRec-2-auto-annotate
│   │   ├── (same files as MIntRec-1-auto-annotate with sdif replaced with mult)
│
├── Debiased
│   ├── MIntRec-1
│   │   ├── debiased_train.tsv
│   │   ├── debiased_dev.tsv
│   │   ├── debiased_test.tsv
│   ├── MIntRec-2
│   │   ├── (same files as MIntRec-1)
├── Original
│   ├── MIntRec-1
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   ├── MIntRec-2
│   │   ├── (same files as MIntRec-1)

```

---

## Auto-Annotate

The `Auto-Annotate` folder contains automatically annotated datasets for two datasets: `MIntRec-1-auto-annotate` and `MIntRec-2-auto-annotate`. Each of these subfolders contains:

### CSV Files and Their Meaning:
- **combined_intent_modality.csv**: Contains combined intent and modality annotations.
- **meta_sdif/mult_bert.csv**: Metadata file associated with SDIF/mult-BERT processing.
- **sdif/mult_bert_all.csv**: All modalities were used in processing.
- **sdif/mult_bert_audio.csv**: Only the audio modality was used; text and video were masked.
- **sdif/mult_bert_audio_video.csv**: Both audio and video were used; text was masked.
- **sdif/mult_bert_text.csv**: Only the text modality was used; audio and video were masked.
- **sdif/mult_bert_text_audio.csv**: Both text and audio were used; video was masked.
- **sdif/mult_bert_text_video.csv**: Both text and video were used; audio was masked.
- **sdif/mult_bert_video.csv**: Only the video modality was used; text and audio were masked.
- **test.tsv**: The test dataset in TSV format.

---

## Debiased_Dataset

The `Debiased_Dataset` folder contains debiased versions of the datasets, following the Phase-3 FGIU (Fairness, Generalization, and Interpretability in Understanding) guidelines.

Each dataset (`MIntRec-1` and `MIntRec-2`) contains:
- **debiased_train.tsv**: Training dataset after debiasing.
- **debiased_dev.tsv**: Development dataset after debiasing.
- **debiased_test.tsv**: Test dataset after debiasing.

These datasets ensure that biases in multimodal intent recognition are mitigated according to the established Phase-3 guidelines.

---

## Usage
- The `Auto-Annotate` folder is useful for working with different modality-based feature masking strategies.
- The `Debiased_Dataset` folder is intended for fair evaluation using debiased train/dev/test splits.


