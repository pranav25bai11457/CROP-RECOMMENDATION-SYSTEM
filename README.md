# 🌾 Crop Recommendation System

A machine learning-based web application that recommends the most suitable crop to grow based on soil nutrients and climate conditions. Built using Python, Scikit-learn, and Streamlit.

---

## 📌 Problem Statement

Farmers in India often lack access to data-driven guidance when deciding which crop to grow. Poor crop selection based on incomplete knowledge of soil chemistry and weather leads to low yields, resource waste, and economic loss. This project addresses that gap by building a smart recommendation system accessible through a simple web interface.

---

## 🚀 Features

- Takes **7 input parameters**: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall
- Predicts the **best crop** using a trained Random Forest Classifier
- Displays **Top 3 crop recommendations** with confidence scores
- Shows **crop-specific growing tips**
- Clean, responsive **Streamlit web interface**

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Model | Random Forest (Scikit-learn) |
| Web App | Streamlit |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📂 Project Structure

```
crop_recommendation/
│
├── data/
│   └── Crop_recommendation.csv   # Dataset (download from Kaggle)
│
├── models/
│   ├── crop_model.pkl            # Trained model (generated after training)
│   └── label_encoder.pkl         # Label encoder (generated after training)
│
├── app/
│   └── app.py                    # Streamlit web application
│
├── plots/                        # EDA & evaluation plots (auto-generated)
│
├── train_model.py                # Model training script
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/crop-recommendation-system.git
cd crop-recommendation-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

- Go to: [Kaggle – Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Download `Crop_recommendation.csv`
- Place it inside the `data/` folder

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Perform EDA and save plots in `plots/`
- Train a Random Forest Classifier
- Print accuracy and classification report
- Save `crop_model.pkl` and `label_encoder.pkl` in `models/`

### 5. Launch the Web App

```bash
streamlit run app/app.py
```

Open your browser and go to `http://localhost:8501`

---

## 📊 Dataset

- **Source:** [Kaggle – Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Rows:** 2,200 samples
- **Classes:** 22 crops
- **Features:** N, P, K, temperature, humidity, ph, rainfall

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~99% |
| Algorithm | Random Forest |
| Train/Test Split | 80% / 20% |

---

## 🌱 Supported Crops

Rice, Wheat, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 📸 Screenshots

> After running the app, you'll see:
> - Input sliders for soil and climate parameters
> - Top crop recommendation with emoji and growing tip
> - Confidence scores for Top 3 crops
> - Input summary table

---

## 🤝 Acknowledgements

- Dataset by [Atharva Ingle on Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Built as part of the BYOP (Bring Your Own Project) capstone assignment

---

## 📄 License

This project is for educational purposes. Dataset is publicly available on Kaggle under its respective license.
