# 🚀 Distributed Sports News Classification System (Big Data)
## 📋 Overview

This project implements a distributed system for large-scale sports news classification using Naive Bayes on Apache Spark (PySpark).

The system is designed to handle massive datasets (up to 150 million records) efficiently while maintaining high accuracy and fast response time.

## 🎯 Key Features

⚡ Distributed data processing using Apache Spark

🧠 Machine Learning pipeline with Naive Bayes

📊 Efficient handling of large-scale datasets (150M+ records)

🚀 Fast inference with Streamlit web interface

📦 Optimized storage using Apache Parquet

## 🛠 Tech Stack
Python
PySpark (Apache Spark)
Naive Bayes
Pandas
Streamlit
Apache Parquet
## 🏗 System Architecture
### 🔹 Distributed Processing (Spark)
Master–Worker architecture
Parallel data processing across nodes
### 🔹 Machine Learning Pipeline
Text → Tokenization → HashingTF → IDF → Naive Bayes → Prediction
## 📊 Performance
📈 Dataset size: 150,000,000 records
⏱ Training time: ~53.5 minutes
🎯 Accuracy: ~95%
⚡ UI response time: < 1 second
```text
#📁 Project Structure
spark-naivebayes-text-classifier/
│
├── data/               # Sample data or generation scripts
├── models/             # Trained models (excluded large files)
├── notebooks/          # Experiment notebooks
├── src/
│   ├── train_spark.py
│   └── app_streamlit.py
├── config/
├── docs/               # Reports & architecture diagrams
├── requirements.txt
├── .gitignore
└── README.md
```
# ⚙️ Installation
## 1. Clone repository
git clone https://github.com/yourusername/spark-naivebayes-text-classifier.git
cd spark-naivebayes-text-classifier
## 2. Install dependencies
pip install -r requirements.txt
## 3. Run training
python src/train_spark.py
## 4. Run Streamlit app
streamlit run src/app_streamlit.py
# ⚠️ Notes
Large datasets (Parquet files) are not included in this repository
Please generate or download data separately
#📸 Demo (Recommended)
<img width="945" height="434" alt="image" src="https://github.com/user-attachments/assets/009dd09f-7d41-4781-9490-3f2c33f968dc" />

# 🤝 Contributing
Fork the repository
Create a feature branch
Commit changes
Open a Pull Request
# 📞 Contact
Author: Nguyen Thien An
GitHub: https://github.com/Thienan12703
