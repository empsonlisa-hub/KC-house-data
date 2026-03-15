import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset (Replace 'kc_house_data.csv' with your local filename)
df = pd.read_csv('kc_house_data.csv')

# Step 2: Select relevant columns and split x/y
# Dropping ignored columns: id, date, view, yr_renovated, lat, long
cols_to_drop = ['id', 'date', 'view', 'yr_renovated', 'lat', 'long']
df_cleaned = df.drop(columns=cols_to_drop)

X = df_cleaned.drop('price', axis=1)
y = df_cleaned['price']

# Encode categorical data (Zipcode) and scale numerical features
categorical_features = ['zipcode']
numerical_features = X.columns.drop('zipcode').tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Step 2: Split data (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

models = {
    "Multiple Linear": LinearRegression(),
    "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
    "Linear SVR": SVR(kernel='linear'),
    "Non-Linear SVR (RBF)": SVR(kernel='rbf'),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Training and Scoring
results = {}
for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    results[name] = r2_score(y_test, y_pred)

# Polynomial Regression (Separate due to feature transformation)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_processed)
X_poly_test = poly.transform(X_test_processed)
poly_model = LinearRegression().fit(X_poly_train, y_train)
results["Polynomial (Deg 2)"] = r2_score(y_test, poly_model.predict(X_poly_test))

# Compare Results
for model_name, score in results.items():
    print(f"{model_name} R2 Score: {score:.4f}")
# Create new house data (Ensure all features match the original DataFrame order)
# Features: bedrooms, bathrooms, sqft_living, sqft_lot, waterfront, floors, condition, grade, sqft_above, sqft_basement, yr_built, zipcode, sqft_living15
new_house = pd.DataFrame([[3, 2, 1800, 5000, 0, 1, 3, 7, 1800, 0, 1990, 98028, 1800]],
                         columns=X.columns)

# Process and Predict
new_house_processed = preprocessor.transform(new_house)
estimated_price = models["Random Forest"].predict(new_house_processed)

print(f"Estimated Price for Kenmore Home: ${estimated_price[0]:,.2f}")

