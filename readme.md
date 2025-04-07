# 🧠 COVID-19 Fake News Detection with Transformers

This project evaluates and compares four transformer-based language models on a binary classification task: determining whether a COVID-19-related tweet is **real** or **fake**.

## 📦 Models Compared

1. **BERT-base-uncased**  
2. **SocBERT-base** ([sarkerlab/SocBERT-base](https://huggingface.co/sarkerlab/SocBERT-base))  
3. **TWHIN-BERT** ([twitter/twhin-bert-base](https://huggingface.co/twitter/twhin-bert-base))  
4. **COVID-Twitter-BERT** ([digitalepidemiologylab/covid-twitter-bert](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert))

---

## 🧪 Dataset

Each model is trained and evaluated on a dataset of tweets labeled as `real` or `fake`. The data is split into:
- **Training**
- **Validation**
- **Test**

Each tweet is under 128 tokens and tokenized accordingly.

---

## 🏋️ Training Configuration

- Optimizer: AdamW  
- Learning Rate: `2e-5`  
- Epochs: `3`  
- Mixed Precision (`fp16`): Enabled  
- Batch Size: `8`  
- Evaluation Strategy: Per Epoch  
- Framework: HuggingFace Transformers + PyTorch

---

## 🔢 Final Results

| Metric        | **BERT**      | **SocBERT**    | **TWHIN-BERT** | **COVID-Twitter-BERT** |
|---------------|---------------|----------------|----------------|-------------------------|
| **Accuracy**  | 0.962         | 0.956          | 0.967          | **0.975**               |
| **Precision** | 0.962         | 0.957          | 0.967          | **0.975**               |
| **Recall**    | 0.962         | 0.956          | 0.967          | **0.975**               |
| **F1 Score**  | 0.962         | 0.956          | 0.967          | **0.975**               |
| **Val Loss**  | 0.215         | 0.249          | 0.187          | **0.198**               |
| **Train Loss**| 0.016         | 0.038          | 0.082          | **0.010**               |

✅ **COVID-Twitter-BERT** achieves the best overall performance across all metrics.

---

## 📤 Prediction Output

Each model was used to predict labels on the test set. The predictions were saved into:

```
test_predictions.csv
```

It includes:
- True label  
- Predicted labels from each model (`predicted_label_bert`, `predicted_label_socbert`, `predicted_label_twhin`, `predicted_label_covid_bert`)  

---

## 🔍 Hidden Representation Extraction

You can extract hidden embeddings from each model using:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "MODEL_NAME"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer("Some tweet", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_embedding = last_hidden_state.mean(dim=1)
```

This embedding can be used for clustering, visualization (e.g., t-SNE), or downstream tasks.

---

## 🛠 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- scikit-learn
- pandas

Install with:

```bash
pip install transformers datasets scikit-learn pandas
```

---

## 📌 Notes

- Mixed precision (`fp16=True`) speeds up training significantly on supported GPUs.
- Each model logs its results separately (`./logs_*`, `./results-*`).
- Training time varies by model size and hardware (around 10–15 min per model on a good GPU).

