
# Import pandas library to work with datasets
import pandas as pd


# STEP 1: DATASET LOADING

# Read the CSV file and store it in a DataFrame
data = pd.read_csv('loan_approval_dataset.csv')

# Display all rows and columns of the dataset
print(data.to_string())


# STEP 2: DATASET EXPLORATION

# Display information about columns, data types and non-null values
print(data.info())

# Display statistical information such as mean, min, max and standard deviation
print(data.describe())

# Check the number of missing values in each column
print(data.isnull().sum())


# STEP 3: DATA CLEANING

# Remove rows that contain missing values
data.dropna(inplace=True)

# Verify that missing values have been removed
print(data.isnull().sum())

# Check the number of duplicate records
print('To find the duplicates')
print(data.duplicated().sum())

# Remove duplicate rows from the dataset
data.drop_duplicates(inplace=True)

# Verify that duplicate records have been removed
print("After drop duplicates records")
print(data.duplicated().sum())


# STEP 4: DATA PREPROCESSING

# Import LabelEncoder to convert categorical labels into numbers
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Convert loan status values (Approved/Rejected) into numerical values
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])

# Display the numerical mapping of loan status labels
print("Loan Status Mapping:")

# Show which number represents each loan status
for number, label in enumerate(le.classes_):
    print(number, "=", label)


# STEP 5: DISPLAY PROCESSED DATASET

# Display updated dataset information
print(data.info())

# Display the first five records
print(data.head())

# Display the last five records
print(data.tail())

# Display all column names
print(data.columns)


# STEP 6: FEATURE SELECTION

# Select independent variables used to predict loan approval
input_labels = data[['Annual_Income', 'CIBIL_Score',
                     'Account_Age', 'Loan_Amount']]

# Select the dependent variable (target/output)
out_label = data['Loan_Status']


# STEP 7: SPLITTING THE DATASET

# Import train_test_split to divide the dataset
from sklearn.model_selection import train_test_split

# Split 80% of the data for training and 20% for testing
# random_state=42 makes the split reproducible
input_label_train, input_label_test, out_label_train, out_label_test = train_test_split(
    input_labels,
    out_label,
    test_size=0.2,
    random_state=42
)

# Display training input features
print(input_label_train)

# Display training output labels
print(out_label_train)


# STEP 8: FEATURE SCALING

# Import StandardScaler to standardize numerical features
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
ss = StandardScaler()

# Learn the mean and standard deviation from training data
# and transform the training features
input_label_train_scaled = ss.fit_transform(input_label_train)

# Transform testing features using the same training scaler
# This avoids data leakage from the test dataset
input_label_test_scaled = ss.transform(input_label_test)


# STEP 9: MODEL SELECTION

# Import Logistic Regression for binary classification
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model
model = LogisticRegression()


# STEP 10: MODEL TRAINING

# Train the model using scaled training inputs and actual output labels
model.fit(input_label_train_scaled, out_label_train)

# Display confirmation after model training
print("Model Trained Successfully......")


# STEP 11: MODEL TESTING

# Display the testing input features
print(input_label_test)

# Display the actual loan status values
print('Actual Outputs:', out_label_test)

# Predict loan status using scaled testing data
test_prediction = model.predict(input_label_test_scaled)

# Display the predicted loan status values
print('Model Predicted Outputs:', test_prediction)


# STEP 12: MODEL PERFORMANCE EVALUATION

# Import metrics to evaluate classification performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy measures the proportion of correct predictions
accuracy = accuracy_score(out_label_test, test_prediction)
print('Accuracy:', accuracy * 100, '%')

# Precision measures how many predicted positive cases are correct
precision = precision_score(out_label_test, test_prediction)
print('Precision:', precision * 100, '%')

# Recall measures how many actual positive cases are identified
recall = recall_score(out_label_test, test_prediction)
print('Recall:', recall * 100, '%')

# F1-score balances precision and recall
f1score = f1_score(out_label_test, test_prediction)
print('F1-score:', f1score * 100, '%')


# STEP 13: PREDICTION FOR A NEW LOAN APPLICANT

# Create a new applicant record with four input features
unseendata = pd.DataFrame({
    'Annual_Income': [700000],
    'CIBIL_Score': [720],
    'Account_Age': [5],
    'Loan_Amount': [350000]
})

# Apply the same scaling used during model training
unseen_data_scaled = ss.transform(unseendata)

# Predict loan approval status for the new applicant
prediction = model.predict(unseen_data_scaled)

# Display the numerical prediction
print('Results:', prediction)

# Convert the numerical prediction back to its original status label
predicted_status = le.inverse_transform(prediction)[0]

# Display the final loan approval prediction
print("Loan Status:", predicted_status)
