import math
import numpy as np
import pandas as pd
from sklearn import linear_model

def gradient_descent(x,y):
    m_cur=b_cur=0
    itr=300000   #infinte loop
    n=len(x)
    learning_rate=0.00021
    curr_cost=0
    # for i in range(itr):kd
    i=0
    while i<100000:
        y_predicted=m_cur*x+b_cur    ## y=m*x+b
        cost=(1/n)*sum([val**2 for val in (y-y_predicted)])
        md=-(2/n)*sum(x*(y-y_predicted))                    ## m derivative 
        bd=-(2/n)*sum(y-y_predicted)
        m_cur=m_cur-learning_rate*md
        b_cur=b_cur-learning_rate*bd
        print(f"m {m_cur} , b {b_cur}, iterations {i} ,cost {cost}")
        if math.isclose(cost,curr_cost,rel_tol=1e-20):
            break
        curr_cost=cost
        i+=1

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
gradient_descent(x,y)
"""
if __name__=="__main__":
    df=pd.read_csv("test_scores.csv")
    x=df.math.values
    y=df.cs.values
    print(f" x - {x} ; y - {y}")"""
    
"""       
reg=linear_model.LinearRegression()
reg.fit(df[['math']],df.cs)
print(f"m {reg.coef_}, b {reg.intercept_}")
"""
# at itration of 390812
# m 1.0177384103254512 , b 1.9150653539521123, iterations 390812 ,cost 31.60451133489039

