import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# download dataset automatically
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
data = pd.read_csv(url)

# fill missing values
data["total_bedrooms"] = data["total_bedrooms"].fillna(
    data["total_bedrooms"].median()
)

# convert categorical column
data = pd.get_dummies(data, columns=["ocean_proximity"])

# features and target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestRegressor(n_estimators=200)

model.fit(X_train, y_train)

# save model
joblib.dump(model, "house_model.pkl")

print("Model trained and saved")