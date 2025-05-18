# 🔋 Ai-powered Intelligent Battery Management & Health Monitoring for Ev’s


Predict and analyze the remaining useful life of batteries with AI-powered visualizations, statistical tools, and interactive UI.

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

- **RUL Prediction**: Predict battery life using a pre-trained Keras model.
- **Data Visualization**: 15+ charts including:
  - Histograms, Boxplots, Violin plots
  - Scatter plots, Heatmaps, Contour plots
  - Q-Q plots, Hexbin plots
- **Statistical Analysis**:
  - 20+ statistical tests
  - Descriptive stats: mean, median, mode
  - Hypothesis testing: t-tests, ANOVA, chi-square
  - Correlation analysis: Pearson, Spearman
- **Feature Importance**: With Optuna and Random Forest.
- **AI-Powered Insights**: Google Gemini API for health interpretation.
- **Vehicle Recommendations**: AI-generated EV suggestions.

---

## 🛠 Technologies Used

- **Backend**: Flask  
- **Machine Learning**: Keras, scikit-learn  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **Statistical Analysis**: SciPy, statsmodels  
- **Optimization**: Optuna  
- **AI Integration**: Google Gemini API  
- **NLP**: NLTK  

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/battery-rul-prediction.git
cd battery-rul-prediction

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"


battery-rul-prediction/
├── app.py                # Main application
├── static/
│   └── images/           # Generated visuals
├── templates/
│   ├── index.html        # Home page
│   ├── inputs.html       # Prediction form
│   ├── visualization.html# Visual interface
│   └── casual.html       # Analysis interface
├── rlu.h5                # Trained Keras model
├── Battery_RUL.csv       # Dataset
└── requirements.txt      # Python dependencies


