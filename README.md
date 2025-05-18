# 🔋 AI-Powered Intelligent Battery Management & Health Monitoring for EVs

Predict and analyze the remaining useful life (RUL) of electric vehicle batteries with AI-powered visualizations, statistical tools, and interactive dashboards.

---

## 📑 Table of Contents

- [🔍 Features](#-features)  
- [🛠 Technologies Used](#-technologies-used)  
- [🚀 Installation](#-installation)  
- [📦 File Structure](#-file-structure)  
- [📡 API Endpoints](#-api-endpoints)  
- [⚙️ Configuration](#️-configuration)  
- [📄 License](#-license)  
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🔍 Features

- ✅ **RUL Prediction**: Predict battery life using a pre-trained Keras model.
- 📊 **Data Visualization**: 15+ chart types including:
  - Histograms, Boxplots, Violin plots  
  - Scatter plots, Heatmaps, Contour plots  
  - Q-Q plots, Hexbin plots
- 📈 **Statistical Analysis**:
  - Descriptive stats: mean, median, mode  
  - Hypothesis testing: t-tests, ANOVA, chi-square  
  - Correlation: Pearson, Spearman  
  - 20+ built-in statistical tests
- 🌲 **Feature Importance**: Using Random Forest & Optuna.
- 🤖 **AI-Powered Insights**: Battery health interpretation using Gemini API.
- 🚘 **EV Suggestions**: Recommends vehicles based on insights.

---

## 🛠 Technologies Used

- **Backend**: Flask  
- **Machine Learning**: Keras, scikit-learn  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **Statistics**: SciPy, statsmodels  
- **Optimization**: Optuna  
- **AI Integration**: Google Gemini API  
- **NLP**: NLTK  

---

## 🚀 Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/yourusername/battery-rul-prediction.git
cd battery-rul-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 📦 File Structure

```bash
battery-rul-prediction/
├── app.py                 # Main Flask application
├── static/
│   └── images/            # Generated charts and visuals
├── templates/
│   ├── index.html         # Home page
│   ├── inputs.html        # Form for RUL prediction
│   ├── visualization.html # Data visualization page
│   └── casual.html        # Stats & insights interface
├── rlu.h5                 # Pre-trained Keras model
├── Battery_RUL.csv        # Battery dataset
└── requirements.txt       # Python dependencies
```

---

## 📡 API Endpoints

| Endpoint           | Method | Description                  |
|--------------------|--------|------------------------------|
| `/`                | GET    | Home page                    |
| `/inputs`          | GET    | Input form for predictions   |
| `/submit_data`     | POST   | Submit data for RUL analysis |
| `/visualizations`  | GET    | Dashboard for visualizations |
| `/visualize_test`  | POST   | Generate specific charts     |
| `/stat_test`       | POST   | Run statistical tests        |
| `/optuna`          | GET    | Optuna analysis interface    |
| `/c_analysis`      | GET    | Run feature importance check |

---

## ⚙️ Configuration

In `app.py`, set your **Gemini API Key**:

```python
import google.generativeai as genai

genai.configure(api_key='YOUR_API_KEY_HERE')
```

- ✅ Model: `rlu.h5` (pre-trained)
- ✅ Images generated are saved in: `static/images/`

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Battery RUL Dataset (Public)
- Flask, Keras, and Plotly
- Google Gemini API
- NLTK, Optuna, Seaborn

---

### ✅ How to Use This README

To include this in your project:

1. Copy the entire content above
2. Save it as `README.md` in your root directory
3. Push to GitHub – it will auto-render beautifully!

---
