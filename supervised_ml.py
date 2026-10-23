# Student Exam Score Prediction
# Supervised Machine Learning using Linear Regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Step 1: Prepare the dataset
x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]
# Step 2: Create the model
model = LinearRegression()
# Step 3: Train the model
model.fit(x, y)
# Step 4: study hours
hours = float(input("Enter hours studied: "))
# Step 5: Predict the exam score
prediction = model.predict([[hours]])
print(f"Predicted exam score: {prediction[0]:.2f}")
# Step 6: Draw the graph 
plt.scatter(x, y, color='blue', label='actual scores')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Student Exam Score Prediction")
plt.legend()
plt.grid(True)
plt.show()
