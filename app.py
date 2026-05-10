import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib

from google.colab import files
import pandas as pd # Ensure pandas is imported here for robustness
print("Please upload the 'housing.csv' file.")
uploaded = files.upload()
if uploaded:
    data_file_name = next(iter(uploaded))
    data = pd.read_csv(data_file_name)
    print(f"Successfully loaded {data_file_name}")
else:
    print("No file uploaded. Please upload the 'housing.csv' file.")
    # Create an empty DataFrame to avoid NameError if no file is uploaded, though subsequent steps will fail.
    data = pd.DataFrame()
data.head()

data.info()
data.describe()

plt.figure(figsize=(6,4))
sns.histplot(data["median_house_value"], bins=50)
plt.title("House Price Distribution")
plt.show()

sns.scatterplot(
x="median_income",
y="median_house_value",
data=data
)

plt.figure(figsize=(12,8))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
plt.show()

data.plot(
kind="scatter",
x="longitude",
y="latitude",
alpha=0.4,
s=data["population"]/100,
label="population"
)

data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]

data["total_bedrooms"] = data["total_bedrooms"].fillna(
    data["total_bedrooms"].median()
)

data = pd.get_dummies(
data,
columns=["ocean_proximity"]
)

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


data["total_bedrooms"] = data["total_bedrooms"].fillna(
    data["total_bedrooms"].median()
)


data["total_bedrooms"] = data["total_bedrooms"].fillna(
    data["total_bedrooms"].median()
)


data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())
data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]
display(data.head())

data.info()

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data scaled successfully.")

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)
print("Linear Regression model trained and predictions made successfully.")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

print("Linear Regression R2:", r2_score(y_test,y_pred))
tree = DecisionTreeRegressor()
tree.fit(X_train,y_train)
tree_pred = tree.predict(X_test)
print("Decision Tree R2:", r2_score(y_test,tree_pred))

forest = RandomForestRegressor()
forest.fit(X_train,y_train)
forest_pred = forest.predict(X_test)
print("Random Forest R2:", r2_score(y_test,forest_pred))

scores = cross_val_score(
lin_model,
X_train,
y_train,
cv=5,
scoring="r2"
)
print(scores)
print("Average Score:", scores.mean())

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators":[100,200],
    "max_depth":[10,20]
}
grid = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    n_jobs=-1
)
grid.fit(X_train,y_train)

if not hasattr(grid, 'best_estimator_'):
    print("GridSearchCV's fit() method has not been executed. Running fit now...")
    grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

y_pred = best_model.predict(X_test)
print("Final R2:", r2_score(y_test,y_pred))
print("MSE:", mean_squared_error(y_test,y_pred))

joblib.dump(best_model,"house_price_model.pkl")

sample = X_test[:1]
prediction = best_model.predict(sample)
print("Predicted Price:", prediction)
