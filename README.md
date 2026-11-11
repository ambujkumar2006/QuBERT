# Qu-BERT: A Hybrid Quantum–Classical Architecture for Text Sentiment Analysis

Qu-BERT is a hybrid quantum–classical model that combines **BERT** with a **Variational Quantum Circuit (VQC)** for three-class sentiment classification.

The project investigates how quantum data encoding, circuit architecture, and BERT fine-tuning influence sentiment-classification performance.

## 1. Overview

The proposed architecture integrates:

- **BERT:** Extracts contextual representations from text.
- **Learned attention pooling:** Aggregates token-level representations into a sentence-level vector.
- **Linear projection:** Reduces the representation to a dimension suitable for quantum encoding.
- **Two-qubit VQC:** Processes the encoded features using a parameterised quantum circuit.
- **Linear classifier:** Uses quantum measurement outputs to predict sentiment.

The model classifies text into three sentiment categories:

- Negative
- Neutral
- Positive

## 2. Model Architecture

The Qu-BERT pipeline follows these steps:

1. Input text is tokenised using the BERT WordPiece tokenizer.
2. BERT generates contextual token representations.
3. Learned masked attention pooling produces a sentence-level representation.
4. A linear projection maps the representation to a low-dimensional feature vector.
5. The features are encoded into a two-qubit quantum state.
6. A trainable variational quantum circuit processes the state.
7. Pauli-Z expectation values are measured and passed to a linear classifier.

## 3. Experimental Configurations

The experiments investigate two quantum data-encoding strategies and two variational circuit architectures.

| Component | Configurations |
|---|---|
| Data encoding | Amplitude encoding, Angle encoding |
| Circuit ansatz | Basic Entangling Layers, Strongly Entangling Layers |
| Quantum circuit | 2 qubits, 2 layers |
| BERT training | Fine-tuned and frozen |
| Classical baseline | Fine-tuned BERT |

### Quantum Encoding

- **Amplitude encoding:** Represents the projected feature vector as the amplitudes of a quantum state.
- **Angle encoding:** Maps projected features to rotation angles using RX gates.

### Variational Circuit Ansätze

- **Basic Entangling Layers:** Uses parameterised RX rotations followed by CNOT entangling gates.
- **Strongly Entangling Layers:** Uses parameterised general rotation gates followed by CNOT entangling gates.

The four hybrid configurations are:

1. Amplitude Encoding + Basic Entangling Layers
2. Amplitude Encoding + Strongly Entangling Layers
3. Angle Encoding + Basic Entangling Layers
4. Angle Encoding + Strongly Entangling Layers

## 4. Dataset

The experiments use the Twitter Sentiment Dataset.

- Total samples: 10,000 tweets
- Input text column: `clean_text`
- Label column: `labels`
- Classes: Negative (-1), Neutral (0), Positive (+1)

The dataset is split into training, validation, and test sets using a stratified 80:10:10 split.

- Random seed: 42
- Test samples: 1,000

## 5. Experimental Results

The fine-tuned hybrid configurations and classical BERT baseline were evaluated using accuracy, macro-precision, macro-recall, macro-F1, and ROC-AUC.

| Model | Accuracy | Macro-F1 | AUC |
|---|---:|---:|---:|
| Classical BERT | 0.898 | 0.891 | 0.970 |
| Amplitude + Basic | 0.887 | 0.880 | 0.947 |
| Amplitude + Strong | 0.901 | 0.893 | 0.959 |
| Angle + Basic | 0.443 | 0.205 | 0.641 |
| Angle + Strong | 0.902 | 0.894 | 0.951 |

### Key Observations

- Three of the four fine-tuned hybrid configurations achieved performance comparable to the classical BERT baseline.
- The Strong ansatz showed a nominal improvement over the Basic ansatz in the amplitude-encoding setting.
- The Angle + Basic configuration exhibited a classification collapse, predicting the positive class for every test sample.
- Freezing BERT reduced hybrid-model accuracy to 0.685–0.736.

The results are based on a single random seed and noiseless two-qubit simulation. The reported bootstrap confidence intervals quantify test-sample uncertainty, not variation across independent training runs.

## 6. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro-Precision
- Macro-Recall
- Macro-F1
- One-vs-rest ROC-AUC
- Confusion matrices
- 95% bootstrap confidence intervals for accuracy and macro-F1

## 7. Implementation

The project uses the following frameworks and libraries:

- PyTorch
- Hugging Face Transformers
- PennyLane
- Scikit-learn

The quantum circuits are simulated using PennyLane's `default.qubit` simulator. No physical quantum hardware is used.

For installation and execution, refer to the dependency files and experiment scripts included in this repository. Use the dataset in the format described above.

## 8. Limitations and Future Work

The current study has several limitations:

- Evaluation is restricted to noiseless two-qubit simulation.
- Experiments use a single random seed.
- A bottleneck-matched classical baseline is needed to isolate the contribution of quantum processing.
- Robustness and probability calibration have not been evaluated.

Future work will investigate larger quantum circuits, multiple random seeds, bottleneck-matched classical comparisons, finite-shot measurements, quantum noise, and execution on physical quantum hardware.

## 9. Research Paper

**Qu-BERT: Quantum-Driven Transformer Architecture for Robust Text Sentiment Analysis**

Authors: Anom
Affiliation: Motilal Nehru College, University of Delhi, India

## 10. Citation

If you use this implementation in your research, please cite the associated Qu-BERT paper.

```text
Anom,
"Qu-BERT: Quantum-Driven Transformer Architecture
for Robust Text Sentiment Analysis."
```

## 11. Acknowledgements

This project uses BERT from Hugging Face Transformers and PennyLane for quantum circuit simulation.
