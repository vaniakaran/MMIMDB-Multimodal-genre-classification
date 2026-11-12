# MMIMDB-Multimodal-genre-classification

**Project Overview**
This project explores multi-label movie genre classification using movie posters and plot summaries from the MM-IMDb dataset. Four approaches are compared to evaluate whether combining image and text information improves genre prediction.

**Dataset**
Dataset: MM-IMDb

Movies: 25,959

Genres: 23

Modalities: Movie posters (images) and plot summaries (text)

Source: https://www.kaggle.com/datasets/johnarevalo/mmimdb

The dataset is downloaded separately and is not redistributed through this repository.

**Methodology**

The project compares four approaches:

Poster-only: ResNet-50 image features with a neural classifier.

Plot-only: BERT text features with a neural classifier.

Early Fusion: Concatenated image and text features passed to a neural classifier.

Late Fusion: Weighted averaging of Poster-only and Plot-only prediction probabilities.

The pretrained ResNet-50 and BERT encoders are frozen during feature extraction.

**Experimental Setup**

Training split: 70%

Validation split: 15%

Test split: 15%

Dataset split seed: 42

Loss function: BCEWithLogitsLoss

Optimiser: Adam

Learning rate: 0.001

Maximum training epochs: 20

Batch size: 64

Classification thresholds and the Late Fusion weight are selected using validation data.


## Results

The four models were evaluated on the MM-IMDb test set using validation-selected classification thresholds.

**Results**
| Model | Micro-F1 | Macro-F1 | Sample-F1 | Macro-AP | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Poster-only | 0.4723 | 0.3241 | 0.4728 | 0.3165 | 0.1641 |
| Plot-only | 0.6394 | 0.5765 | 0.6373 | 0.6102 | 0.0959 |
| **Early Fusion** | **0.6607** | **0.5930** | **0.6618** | **0.6187** | **0.0826** |
| Late Fusion | 0.6394 | 0.5765 | 0.6373 | 0.6102 | 0.0959 |

Early Fusion achieved the highest Macro-F1 in the main test experiment. However, repeated-seed validation results did not consistently demonstrate an advantage over Plot-only.

**How to Run**

Open MMIMDb_Classification.ipynb in Google Colab.

Enable a GPU runtime if available.

Install the dependencies listed in requirements.txt.

Run the notebook cells in order.

The notebook downloads the MM-IMDb dataset and pretrained encoders, extracts features, trains classifiers and evaluates the models.

Note: Internet access and sufficient disk space are required. The complete experiment may take considerable time, particularly during feature extraction. The notebook uses Colab-specific /content/ paths.

**Reproducibility**

The dataset split uses seed 42. Additional classifier training runs use seeds 42, 123 and 456 to examine the effect of training randomness.

Saved notebook outputs show the reported experimental results. Re-execution may produce small differences depending on the software environment and hardware.

**Limitations**

The dataset contains substantial genre imbalance. Class-weighted training was not applied. The experiment also uses frozen pretrained encoders and a fixed text input length, which may limit performance on rare genres and longer plot summaries.

**References**

Arevalo et al. (2017), Gated Multimodal Units for Information Fusion.

He et al. (2016), Deep Residual Learning for Image Recognition.

Devlin et al. (2019), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
