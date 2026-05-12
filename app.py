import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Title
st.title("Iris Dataset SVM Classifier")

# Read Dataset
data = pd.read_csv("IRIS.csv")

# Display Dataset
st.subheader("Dataset")

st.write(data.head())

# Features and Target
X = data.drop("species", axis=1)

y = data["species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Kernel Selection
kernel = st.selectbox(
    "Select Kernel",
    ["linear", "rbf", "poly", "sigmoid"]
)

# Train Model
model = SVC(kernel=kernel)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Accuracy")

st.write(f"Accuracy: {accuracy:.2f}")

# Classification Report
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# Prediction Section
st.subheader("Predict Flower Species")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"Predicted Flower: {prediction[0]}")