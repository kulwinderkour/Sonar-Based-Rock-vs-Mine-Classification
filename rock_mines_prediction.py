import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df= pd.read_csv("Copy of sonar data.csv",header=None)
print(df.head(5))


#check the number of rows and columns 

print(df.shape)   #(207, 61)

#Measure the statistical data
print(df.describe())  


#check how many data for rocks and how many for mines
print(df.iloc[:,60].value_counts())    # using indexing
# Mines    111
# Rocks     96

# check the mean value using grouping   
grouped_data = df.groupby(60).mean()
print(grouped_data)



# seperating data and lables

x = df.drop(columns=60,axis=1)
y=  df[60]
print(x)
print(y)



#train and test data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y )




model = LogisticRegression()

#train the logistic regression model with the trained data

model.fit(x_train,y_train)

print(x_train)
print(y_train)





#model train prediction

x_train_prediction = model.predict(x_train)
# accuracy on trained data
training_data  = accuracy_score(x_train_prediction,y_train)

print(training_data)   # 80% trained data is accurate

#model tested prediction
x_test_prediction = model.predict(x_test)
# accuracy on test_data
test_accuracy = accuracy_score(x_test_prediction,y_test)
print(test_accuracy)    # 80% tested data is accurate




# MAKING A PREDICTIVE SYSTEM

# input_data  = (0.0131,0.0387,0.0329,0.0078,0.0721,0.1341,0.1626,0.1902,0.2610,0.3193,0.3468,0.3738,0.3055,0.1926,0.1385,0.2122,0.2758,0.4576,0.6487,0.7154,0.8010,0.7924,0.8793,1.0000,0.9865,0.9474,0.9474,0.9315,0.8326,0.6213,0.3772,0.2822,0.2042,0.2190,0.2223,0.1327,0.0521,0.0618,0.1416,0.1460,0.0846,0.1055,0.1639,0.1916,0.2085,0.2335,0.1964,0.1300,0.0633,0.0183,0.0137,0.0150,0.0076,0.0032,0.0037,0.0071,0.0040,0.0009,0.0015,0.0085)
# take the input from the user


user_input = input("Enter 60 sonar values seperated by commas:")
input_data = [float(x) for x in user_input.split(",")]



#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# reshape for prediction
reshape_input_data = input_data_as_numpy_array.reshape(1,-1)
model_predict =model.predict(reshape_input_data)
print(model_predict)


if(model_predict[0] == "R"):
    print("The object is rock")
else:
    print("the object is mine")




