import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Data
data = {'Hours_study': [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], 
        'Exam_score': [5, 20, 26, 32, 51, 60, 72, 78, 83, 85, 97]}

data_frame = pd.DataFrame(data) 
# print(data_frame)

X = data_frame[['Hours_study']]
Y = data_frame[['Exam_score']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = LinearRegression()

model.fit(X_train, Y_train)

user_input = float(input("Enter Your Study Hour: "))

predicted_score = model.predict([[user_input]])

print(predicted_score)