import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv("/home/yxlin/github/LHY_ML2025/MLHW2/ML2025Spring-hw2-public/train.csv")

# Load the test data
test_df = pd.read_csv("/home/yxlin/github/LHY_ML2025/MLHW2/ML2025Spring-hw2-public/test.csv")

# Drop the 'id' column since it's not needed for training or testing
train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)

# Check if there are any missing values in the training data
print(train_df.isnull().sum())

# If there are missing values, you might want to handle them. For simplicity, let's drop rows with missing values.
train_df.dropna(inplace=True)

# Similarly, check for missing values in the test data and handle them if necessary
print(test_df.isnull().sum())
test_df.dropna(inplace=True)

X_train = train_df.drop(columns=['tested_positive_day3'])
y_train = train_df['tested_positive_day3']

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
test_predictions = model.predict(test_df)