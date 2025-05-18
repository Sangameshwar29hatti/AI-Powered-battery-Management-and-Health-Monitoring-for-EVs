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


## 📦 File Structure

```bash




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


## 📡 API Endpoints

| Endpoint          | Method | Description             |
| ----------------- | ------ | ----------------------- |
| `/`               | GET    | Home page               |
| `/inputs`         | GET    | Input form              |
| `/submit_data`    | POST   | Process prediction      |
| `/visualizations` | GET    | Visualization UI        |
| `/visualize_test` | POST   | Generate visualizations |
| `/stat_test`      | POST   | Run statistical tests   |
| `/optuna`         | GET    | Show analysis interface |
| `/c_analysis`     | GET    | Run Optuna analysis     |

## ⚙️ Configuration

Set Gemini API key in app.py:
python
genai.configure(api_key='YOUR_API_KEY_HERE')
Model: Uses pre-trained rlu.h5
Visuals save to static/images/

## 📄 License

MIT License - See LICENSE
Acknowledgments
Battery RUL dataset
Flask, Keras, Plotly libraries
Google Gemini API

## 🙏 Acknowledgments

To download this file:
1. Copy the entire content above
2. Save it as `README.md` in your project root
3. The markdown will render properly on GitHub/GitLab
The file includes:
- Badges for key technologies
- Table of Contents for easy navigation
- Concise feature descriptions
- Installation commands in code blocks
- Visual file structure tree
- API endpoint table
- Configuration instructions
- License and acknowledgments
- Download prompt at the bottom
All sections are properly linked and formatted for optimal readability on code hosting platforms
<style type="text/css">@media print {
 *, :after, :before {background: 0 0 !important;color: #000 !important;box-shadow: none !important;text-shadow: none !im
 a, a:visited {text-decoration: underline}
 a[href]:after {content: " (" attr(href) ")"}
 abbr[title]:after {content: " (" attr(title) ")"}
 a[href^="#"]:after, a[href^="javascript:"]:after {content: ""}
 blockquote, pre {border: 1px solid #999;page-break-inside: avoid}
 thead {display: table-header-group}
 img, tr {page-break-inside: avoid}
 img {max-width: 100% !important}
 h2, h3, p {orphans: 3;widows: 3}
 h2, h3 {page-break-after: avoid}
}
html {font-size: 12px}
@media screen and (min-width: 32rem) and (max-width: 48rem) {
 html {font-size: 15px}
}
@media screen and (min-width: 48rem) {
 html {font-size: 16px}
}
body {line-height: 1.85}
.air-p, p {font-size: 1rem;margin-bottom: 1.3rem}
.air-h1, .air-h2, .air-h3, .air-h4, h1, h2, h3, h4 {margin: 1.414rem 0 .5rem;font-weight: inherit;line-height: 1.42}
.air-h1, h1 {margin-top: 0;font-size: 3.998rem}
.air-h2, h2 {font-size: 2.827rem}
.air-h3, h3 {font-size: 1.999rem}
.air-h4, h4 {font-size: 1.414rem}
.air-h5, h5 {font-size: 1.121rem}
.air-h6, h6 {font-size: .88rem}
.air-small, small {font-size: .707em}
canvas, iframe, img, select, svg, textarea, video {max-width: 100%}
body {color: #444;font-family: 'Open Sans', Helvetica, sans-serif;font-weight: 300;margin: 0;text-align: center}
img {border-radius: 50%;height: 200px;margin: 0 auto;width: 200px}
a, a:visited {color: #3498db}
a:active, a:focus, a:hover {color: #2980b9}
pre {background-color: #fafafa;padding: 1rem;text-align: left}
blockquote {margin: 0;border-left: 5px solid #7a7a7a;font-style: italic;padding: 1.33em;text-align: left}
li, ol, ul {text-align: left}
p {color: #777}</style>



