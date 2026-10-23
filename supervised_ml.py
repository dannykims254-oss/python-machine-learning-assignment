#student Exam score prediction
#using Linear Regtression

from sklearn.linear_model import LinearRegression

# 1. Prepare the input data (hours studied)
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

# 2. Prepare the target data (exam scores)
y = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

# 3. Create the Linear Regression model
model = LinearRegression()

# 4. Train the model
model.fit(X, y)

# 5. Ask the user for study hours
hours = float(input("Enter hours studied: "))

# 6. Predict the examination score
prediction = model.predict([[hours]])


# 7. Display the prediction to two decimal places
print(f"Predicted exam score: {prediction[0]:.2f}")


