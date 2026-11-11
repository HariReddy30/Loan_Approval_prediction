import pandas as pd
#to load dataset
df = pd.read_csv('loan_approval_dataset.csv')
print(df.info())
print(df.describe())
#to find null values
print('Null values:',df.isnull().sum())
#df.dropna(inplace=True)
#df.fillna(100,inplace=True)
#to find duplicated records
print('repeated records:',df.duplicated().sum())
print(df)
#to convert into textual data into numerical data
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df['Loan_Status']= le.fit_transform(df['Loan_Status'])
print("Loan status mapping:")

for number, label in enumerate(le.classes_):
    print(number, "=", label)

print('after processing...')

# import matplotlib.pyplot as plt
# plt.plot(df['Annual_Income'],df['Loan_Status'])
# plt.show()
inputs_x = df[['Annual_Income','CIBIL_Score','Account_Age','Loan_Amount']]
output_Y = df['Loan_Status']

#data splitting training and testing
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test= train_test_split(inputs_x,output_Y,test_size=0.2,random_state=42)
print('Training dataset:\n',X_train)
print('training output:\n',Y_train)
print('testing input:\n',X_test)

#feature scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train_scaled=ss.fit_transform(X_train)
X_test_scaled=ss.transform(X_test)
#model selection

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train_scaled,Y_train)
print('Model trained Succussfully.......')
prediction_testing=model.predict(X_test_scaled)
print('testing prediction:',prediction_testing)
print('actual testing output:',Y_test)
#print(df)

#
from sklearn.metrics import accuracy_score,precision_score,recall_score
print('Accuracy score:',accuracy_score(Y_test,prediction_testing))
print('precision score:',precision_score(Y_test,prediction_testing))
print('recall score:',recall_score(Y_test,prediction_testing))
new=pd.DataFrame({'Annual_Income':[200000],'CIBIL_Score':[500],'Account_Age':[2],'Loan_Amount':[40000]})
scaled_new=ss.transform(new)
new_prediction=model.predict(scaled_new)
if new_prediction==1:
    print('Loan rejected')
else:
    print('Loan approved')
print(new_prediction)
#print(scaled_new)

# STEP 14: SAVING THE TRAINED ML MODEL

# Import joblib library
import joblib

# Save the trained Logistic Regression model
joblib.dump(model, "model.pkl")

# Save the StandardScaler
joblib.dump(ss, "scaler.pkl")

# Save the LabelEncoder
joblib.dump(le, "label_encoder.pkl")

print("Model, Scaler and Label Encoder saved successfully!")