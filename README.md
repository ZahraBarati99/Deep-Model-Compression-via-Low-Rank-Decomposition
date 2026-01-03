# Deep-Model-Compression-via-Low-Rank-Decomposition

Model Compression of Deep Neural Networks Using Low-Rank Matrix Decomposition

---

## Overview

This project investigates **model compression in deep neural networks** using
**low-rank matrix decomposition** techniques based on *Singular Value Decomposition (SVD)*.
The main objective is to reduce model complexity while preserving predictive performance,
with a particular focus on **industrial deployment constraints** such as limited computational
resources and inference efficiency.

A pretrained **ResNet-18** model is used as the backbone architecture. The spectral properties
of selected layers are first analyzed to motivate low-rank approximations. Subsequently,
the fully connected classification layer is replaced by a low-rank factorized equivalent,
and the effects on accuracy, parameter count, and inference speed are systematically evaluated.

---

## Methodology

### Spectral Analysis of Model Weights

In the first stage, the weight matrices of a **middle convolutional layer** and the **final
fully connected layer** are extracted from the pretrained model.  
Using Singular Value Decomposition (SVD), the **singular value spectrum** and the
**cumulative explained variance** are analyzed.

This analysis reveals that a significant portion of the total variance is captured by a
subset of dominant singular values, indicating the presence of **redundancy** in the parameter
space and motivating low-rank compression.

---

### Low-Rank Model Compression

Based on the spectral analysis, the fully connected layer is replaced by a pair of consecutive
linear layers whose product approximates the original weight matrix using a truncated SVD.
Different **compression ratios** are examined by retaining only a fraction of the singular values.

To preserve performance, the backbone network is frozen and only the classifier head is
lightly fine-tuned using a small subset of training data. This strategy significantly reduces
training cost while partially recovering accuracy loss caused by compression.

---

## Experimental Setup

- **Model:** Pretrained ResNet-18  
- **Dataset:** CIFAR-10 (used for fine-tuning and evaluation)  
- **Input Resolution:** Reduced input size to improve evaluation speed  
- **Compression Ratios:** 0% (baseline), 50%, and 80%  
- **Evaluation Metrics:**
  - Top-1 classification accuracy
  - Number of model parameters
  - CPU inference time per batch

All experiments are conducted on CPU to reflect realistic deployment conditions.

---

## Results

The experimental results demonstrate a clear trade-off between **compression rate** and
**model accuracy**. Moderate compression achieves parameter reduction with acceptable accuracy
degradation, while aggressive compression leads to significant performance loss.

Fine-tuning the compressed models improves accuracy but cannot fully compensate for extreme
rank reduction. Inference time measurements indicate that compressing only the final layer
does not significantly accelerate CPU inference, highlighting the importance of targeting
computational bottlenecks in practical systems.

---

## Key Findings

- Singular value analysis reveals substantial redundancy in deep model weight matrices.
- Low-rank approximation is effective for reducing parameter count with limited performance loss.
- Fine-tuning the compressed classifier head improves stability and accuracy.
- Compressing only the final layer has limited impact on inference speed.
- Spectral properties can guide the selection of layers suitable for compression.

---

## Conclusion

This study demonstrates that **low-rank decomposition** provides a principled and interpretable
approach to deep model compression. Spectral analysis offers valuable insights into parameter
redundancy and enables informed compression decisions. While low-rank methods are effective for
reducing storage and improving numerical stability, achieving inference speedups requires
compression of computationally dominant layers.

The results highlight the importance of combining mathematical analysis with system-level
considerations for deploying deep learning models in industrial environments.
