import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Salary_data.csv")
X=df.iloc[:,0:1].values
y=df.iloc[:,[1]].values

#from sklearn.model_selection import train_test_split
#X_test,y_test,X_train,y_train=train_test_split(X, y,test_size=2,random_state=45)

from sklearn.linear_model import LinearRegression
li_model=LinearRegression()
li_model.fit(X,y)

plt.scatter(X,y,color="Red")
plt.plot(X,li_model.predict(X),color="Blue")
plt.title("Experience vs Salary")
plt.xlabel("year and Experince")
plt.ylabel(" Salary")
plt.show()           



from sklearn.preprocessing import PolynomialFeatures
poly_feach=PolynomialFeatures(degree=3)
X_polyno= poly_feach.fit_transform(X)
li_model_poly=LinearRegression()
li_model_poly.fit(X_polyno,y)
y_pred=li_model_poly.predict(X_polyno)


plt.scatter(X,y,color="Red")
plt.scatter(X,y_pred,color="Blue")
plt.title("Experience vs Salary")
plt.xlabel("year and Experince")
plt.ylabel(" Salary")
plt.show()    