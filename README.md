# News Topic Classifier Using BERT

## ■ Objective of the Task
The primary goal of this project is to fine-tune a pre-trained transformer model to classify news headlines into four distinct categories: World, Sports, Business, and Sci/Tech. Using the **AG News Dataset**, the project demonstrates a full machine learning pipeline from data preprocessing and model training to interactive deployment.

## ■ Methodology / Approach
To achieve high classification accuracy, the following technical approach was implemented:

* **Dataset & Preprocessing**: The project utilizes the **AG News Dataset** from Hugging Face. Headlines were processed using the `BertTokenizer` with a `max_length` of **128**, incorporating truncation and padding to ensure uniform input sizes for the model.
* **Model Selection**: The **`bert-base-uncased`** model was selected as the foundation for transfer learning. 
* **Fine-Tuning Strategy**: The model was fine-tuned using the Hugging Face `Trainer` API. Key hyperparameters included:
    * **Learning Rate**: $2 \times 10^{-5}$.
    * **Batch Size**: 16 for both training and evaluation.
    * **Epochs**: 2.
    * **Weight Decay**: 0.01 to manage overfitting.
* **Evaluation Metrics**: Performance was measured using **Accuracy** and **Weighted F1-score** to ensure balanced classification across all topics.
* **Deployment**: An interactive "AI News Assistant" was developed using **Streamlit**. This interface allows users to input headlines in real-time and view classification results alongside confidence visualizations powered by **Altair**.

## ■ Key Results or Observations
The fine-tuning process yielded highly successful results over two training epochs:

| Metric | Epoch 1 | Epoch 2 (Final) |
| :--- | :--- | :--- |
| **Training Loss** | 0.1932 | 0.1213 |
| **Validation Loss**| 0.1784 | 0.1911 |
| **Accuracy** | 94.46% | **94.83%** |
| **F1-Score** | 0.9446 | **0.9483** |

**Observations:**
* The model achieved nearly **95% accuracy** in just two epochs, demonstrating the power of BERT for Natural Language Processing tasks.
* The validation loss remained low and stable relative to the training loss, suggesting that the chosen weight decay and learning rate effectively optimized the model without significant overfitting.
* The deployment interface successfully provides visual feedback and "soft warnings" when prediction confidence falls below **60%**, helping users interpret the AI's reliability.

## ■ How to Run
1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the app**:
    ```bash
    streamlit run app.py
    ```