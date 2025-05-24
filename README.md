# 🧠 EMNIST Letters Recognition with Custom Neural Network

This project implements a **handwritten character recognition system** for the **EMNIST Letters dataset** using a fully connected neural network built from scratch with TensorFlow/Keras. The model classifies images of handwritten English letters (A–Z) and investigates the effects of different neural network configurations on accuracy and generalization.

---

## 🗒️ Table of Contents

- [📌 Problem Statement](#-problem-statement)
- [🎯 Objectives](#-objectives)
- [🧪 Methodology](#-methodology)
- [📈 Results](#-results)
- [🧠 Key Insights](#-key-insights)
- [⚠️ Challenges](#️-challenges)
- [🚀 Improvements](#-improvements)
- [📎 Conclusion](#-conclusion)
- [🛠️ Setup Instructions](#️-setup-instructions)

---

## 📌 Problem Statement

Handwritten character recognition is a core challenge in machine learning with real-world applications like document digitization and assistive technology. While digit recognition via MNIST is well-researched, **letter classification (A–Z)** is harder due to **visual similarities and case variance**.

This project utilizes the **EMNIST Letters dataset**, consisting of 145,600 grayscale 28x28 images of handwritten characters, for the classification task.

---

## 🎯 Objectives

- Build a custom neural network architecture **from scratch**.
- Evaluate different configurations:
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Varying hidden layer depths
  - Different batch sizes
- Address overfitting and performance limitations.
- Analyze misclassifications and model features.

---

## 🧪 Methodology

### 📁 Data Preprocessing
- **Dataset**: EMNIST Letters
- **Train/Test Split**: 88,800 / 14,800 samples
- **Normalization**: Scaled pixels to [0, 1]
- **Label Adjustment**: Converted labels 1–26 → 0–25
- **Image Rotation Correction**: Applied `np.rot90()` and `np.fliplr()` to fix 90° rotation

### 🧱 Baseline Architecture

| Layer Type   | Details                        |
|--------------|--------------------------------|
| Input        | Flatten 28x28 → 784            |
| Hidden Layer | Dense(128), ReLU               |
| Hidden Layer | Dense(64), ReLU                |
| Output       | Dense(26), Softmax             |
| Optimizer    | Adam                           |
| Loss         | Sparse Categorical Crossentropy|
| Regularization | EarlyStopping + LR Scheduler |

---

## 📈 Results

| Variant            | Accuracy | Loss   | Notes                                           |
|--------------------|----------|--------|--------------------------------------------------|
| **Baseline (128-64)** | **88.16%** | 0.388  | Best overall performance                         |
| More Layers (256-128-64) | 87.70%   | 0.418  | Higher complexity without gain                   |
| Fewer Layers (64 only)   | 84.54%   | 0.493  | Underfitting observed                            |
| Sigmoid Activation       | 86.97%   | 0.409  | Struggled with vanishing gradients               |
| Tanh Activation          | 87.39%   | 0.398  | Outperformed Sigmoid, still behind ReLU          |
| Small Batch (32)         | 88.16%   | 0.388  | No significant gain                              |
| Large Batch (256)        | 88.16%   | 0.388  | Reduced generalization                           |

---

## 🧠 Key Insights

- **ReLU** consistently outperformed Sigmoid and Tanh in both accuracy and stability.
- **Two hidden layers (128, 64)** hit the sweet spot for capacity without overfitting.
- **Batch size** had negligible impact, thanks to the Adam optimizer.
- Common misclassifications:
  - **L ↔ I**: Similar shapes when slanted.
  - **G ↔ Q**: Round structure confusion.

---

## ⚠️ Challenges

- Initial dataset rotation required manual correction.
- Early cross-validation was invalid due to missing model reinitialization.
- Larger models caused significant strain on system memory and training time.

---

## 🚀 Improvements

Future enhancements could include:

- **Dropout Regularization** (e.g., `Dropout(0.3)`) to combat overfitting.
- **Data Augmentation**: Random rotations/distortions to simulate handwriting variance.
- **Case-Sensitive Datasets**: EMNIST "byclass" for uppercase/lowercase distinction.
- **Advanced Activations**: Swish, Leaky ReLU for deeper network support.

---

## 📎 Conclusion

The **baseline model** with **ReLU**, **128-64 hidden layers**, and a **batch size of 128** gave the best trade-off between performance and complexity:

✅ **Test Accuracy: 88.16%**  
✅ **Stable generalization across folds**  
✅ **Efficient learning with minimal overfitting**

This project proves that **a well-tuned shallow network** can effectively handle complex classification problems like handwritten letter recognition when paired with proper regularization and preprocessing.

---

## 🛠️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/emnist-letter-classifier.git
cd emnist-letter-classifier

# Run the .ipynb file
