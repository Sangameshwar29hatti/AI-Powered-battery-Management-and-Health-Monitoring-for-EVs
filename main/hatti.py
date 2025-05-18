import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from lazypredict.Supervised import LazyRegressor

# Load dataset
data = pd.read_csv('Battery_RUL.csv')
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())

# Detect columns with fewer than 10 unique values
for i in data.columns.values:
    if len(data[i].value_counts().values) < 10:
        print(data[i].value_counts())

# Outlier removal using z-scores
out = []
for i in data.columns.values:
    data['z_scores'] = (data[i] - data[i].mean()) / data[i].std()
    outlier = np.abs(data['z_scores'] > 3).sum()
    if outlier > 3:
        out.append(i)

print(len(data))
thresh = 3
for i in out:
    upper = data[i].mean() + thresh * data[i].std()
    lower = data[i].mean() - thresh * data[i].std()
    data = data[(data[i] > lower) & (data[i] < upper)]

print(len(data))

# Correlation with RUL
corr = data.corr()['RUL']
corr = corr.drop(['RUL', 'z_scores'])
x_cols = [i for i in corr.index if corr[i] > 0]
x = data[x_cols]
y = data['RUL']

print(x.columns.values)

'''# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Initialize models
ext = ExtraTreesRegressor()
rf = RandomForestRegressor()
xgb = XGBRegressor()

# Train models
ext.fit(x_train, y_train)
rf.fit(x_train, y_train)
xgb.fit(x_train, y_train)

# Predictions
print("Extra Trees prediction:", ext.predict(x_test[:5]))
print("Random Forest prediction:", rf.predict(x_test[:5]))
print("XGBoost prediction:", xgb.predict(x_test[:5]))

Evaluation metrics for XGBoost
mse = mean_squared_error(y_test, xgb.predict(x_test))
print("MSE for XGBoost:", mse)
print("RMSE for XGBoost:", np.sqrt(mse))
print("R2 score for XGBoost:", r2_score(y_test, xgb.predict(x_test)))

LazyRegressor for benchmarking
lazy = LazyRegressor()
models, predictions = lazy.fit(x_train, x_test, y_train, y_test)
print(models)

#LIME explanations
explainer = LimeTabularExplainer(
    training_data=np.array(x_train),
    feature_names=x_cols,
    class_names=['RUL'],
    mode='regression'
)

def save_lime_html(model, model_name, x_test_sample):
    # Create explanation for the first sample in the test set
    exp = explainer.explain_instance(
        data_row=x_test_sample,
        predict_fn=model.predict
    )
    # Save to HTML
    exp.save_to_file(f'{model_name}_lime_explanation.html')
    print(f"LIME explanation saved for {model_name}!")

Generate LIME explanations for each model
x_test_sample = np.array(x_test.iloc[0])
save_lime_html(ext, "ExtraTrees", x_test_sample)
save_lime_html(rf, "RandomForest", x_test_sample)
save_lime_html(xgb, "XGBoost", x_test_sample)