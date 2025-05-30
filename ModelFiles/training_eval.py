from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = read_csv("cleaned_data.csv")

X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training
print("Training Model Now..")
model = LinearRegression()
model.fit(X_train, y_train)

# evaluation
print("Evaluating Model Now...")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Model R² Score on Test Data: {r2:.4f}")

# saving trained model
dump(model, "HousingModel.pkl")          