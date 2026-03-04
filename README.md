# NLP-2026
## Repository Contents
* `dev.txt`: Final ensemble predictions for the official Dev Set.
* `test.txt`: Final ensemble predictions for the official unseen Test Set.
* `advanced_eda1_syntax.png`: Stage 2 Exploratory Data Analysis (EDA) violin plots comparing the syntactic density (adjectives and pronouns) of PCL versus non-PCL text.
* `advanced_eda2_tsne.png`: Stage 2 Exploratory Data Analysis (EDA) t-SNE scatter plot visualizing the semantic overlap and severe class imbalance between the positive and negative labels.
* `BestModel/`: Folder containing the final model code.
The ipynb notebook containing the code for my model can be found [here](BestModel/Solution.ipynb)

## Model Information
The system pipeline consists of three main stages: Data Augmentation, Independent Model Training, and Ensemble Voting.

### 1. Data Augmentation
To address the extreme class imbalance and provide the models with more positive signals, the minority class (PCL = 1) was expanded using a hybrid pipeline:
* **Back-Translation:** Translating English PCL samples to French and back to English to generate syntactically diverse paraphrases.
* **Easy Data Augmentation (EDA):** Applying synonym replacement to further diversify the vocabulary without losing the core semantic meaning.

### 2. The Dual-Model Architecture
Instead of relying on a single transformer, two distinct architectures were trained to cover each other's blind spots:

* **Model 1: Optimized RoBERTa-base**
  * **Progressive Layer Freezing:** The encoder was frozen for the first 40% of training, allowing the randomly initialized classification head to adapt without destroying the pre-trained weights.
  * **Multi-Sample Dropout:** Used to stabilize the decision boundary for highly subjective labels.
  * **Discriminative Learning Rates & Label Smoothing (0.1):** Applied to handle the inherent noise in the dataset's ground truth.

* **Model 2: DeBERTa-v3-base**
  * Utilizes **Disentangled Attention**, which separately encodes the content of words and their relative positions. This makes it highly effective at detecting structural condescension (like the "Savior Complex" trope) that standard attention mechanisms often miss.

### 3. Soft-Voting Ensemble
The system averages the out-of-fold probability distributions from both RoBERTa and DeBERTa. An F1-optimized threshold (0.3090) is then applied to the blended probabilities to make the final binary decision. This approach successfully acts as a mutual regularizer, where the models effectively veto each other's false-positive predictions.

## Model Weights (Google Drive)
Because transformer model weights exceed GitHub's file size limits, the best-performing model checkpoints have been securely uploaded to Google Drive. 

You can access and download the exact model folders used in this ensemble here:
[https://drive.google.com/drive/folders/1Tu3-q8A5tAo0kQVP22ZoqoWnvgOnZdIy]
