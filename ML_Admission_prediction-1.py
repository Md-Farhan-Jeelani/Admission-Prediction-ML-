import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("Admission_Predict.csv",sep = ",")
df.head()
df.tail()
df.sample(5)
df.describe()
df.dtypes
df.shape
df.Research.value_counts()
df.drop(["Serial No."],axis=1,inplace = True)
df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
y = df["Chance of Admit"]

x = df.drop(["Chance of Admit"],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.20,random_state=42)
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
score=model.score(x_test, y_test)
print(score)
print(y_predict[0:5])
print(y_test[0:5])
gre=int(input("What is your GRE Score (between 290 to 340):"))
toefl=int(input("What is your TOEFL Score (between 90 to 120):"))
univ=int(input("What is your University Rating ( 1 to 5 ):"))
sop=int(input("Rate your Statement of Purpose ( 1 to 5):"))
lor=int(input("What is strength of  your Letter of Recommendation ( 1 to 5) :"))
cgpa=int(input("What is your CGPA ( 6 to 10):"))
research=int(input("Do You have Research Experience (Enter 0 for No and 1 for Yes:"))
newx=[[gre,toefl,univ,sop,lor,cgpa,research]]
newy=model.predict(newx)
print("Your Chance of Admission is: ",newy)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfr.fit(x_train,y_train)
y_predict_rfr = rfr.predict(x_test)
score_rfr=rfr.score(x_test, y_test)
print(score_rfr)
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(x_train,y_train)
y_predict_dtr = dtr.predict(x_test)
score_dtr=dtr.score(x_test, y_test)
print(score_dtr)
print('Coefficients: ', model.coef_)
print(model.intercept_)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
df
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
print('Mean Absolute Error:', mean_absolute_error(y_test, y_predict))
print('Mean Squared Error:', mean_squared_error(y_test, y_predict))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_predict)))

