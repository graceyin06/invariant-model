#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm, poisson
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statistics import mean
import math


# # Construct the model
# 
# $$X_1 \sim \mathcal{N}(0,\sigma^2)$$
# $$Y= \beta_1 X_1+\varepsilon_y$$
# $$X_2=\beta_2Y+\varepsilon_2$$
# $$\varepsilon_2 \sim \mathcal{N}(0,1)$$
# $$\varepsilon_y \sim \mathcal{N}(0,\sigma^2)$$
# 
# $$\tilde{P}^{\sigma}(dX_1,dX_2,dY)=P(dX_1)W_{X_1}(dX_2)Q_{X_1}^{\sigma}(dy)$$ 
# The linear regression model is equivariant.
# 
# The invariant loss function is $$MSE=\frac{1}{n}\sum_{i=1}^{n}(y-\hat{y})^2$$
# The risk function is 
# $$r(\hat{y})=\int_{\mathcal{X}\times \mathcal{Y}}L(x,y,\hat{y})dP(x  ,y)$$

# ## Compute the estimated coefficients with X1, X2 and (X1, X2)

# In[50]:


#Define coefficient function for y=beta_1 x1
def coef_1(n,sigma,g1,beta1,beta2):
    X1=np.random.normal(0, sigma, n)+g1 #intervention on X1 in environment 1
    Y=beta1*X1+np.random.normal(0,sigma, n)
    X2=beta2*Y+np.random.normal(0, 1, n)
    X=np.hstack([X1.reshape(-1,1)])
    y=np.hstack([Y.reshape(-1,1)])
    reg=LinearRegression().fit(X, y)  #fit the linear regression with only X1
    return np.append(reg.coef_,reg.intercept_)


# In[51]:


#Define coefficient function for y=beta_2 x2
def coef_2(n,sigma,g2,beta1,beta2):
    X1=np.random.normal(0, sigma, n)
    Y=beta1*X1+np.random.normal(0,sigma, n)
    X2=beta2*Y+np.random.normal(0, 1, n)+g2 #intervention on X2 in environment 1(for X2)
    X=np.hstack([X2.reshape(-1,1)]) #fit the linear regression with only X2
    y=np.hstack([Y.reshape(-1,1)])
    reg=LinearRegression().fit(X, y)  
    return np.append(reg.coef_,reg.intercept_)


# In[52]:


#Define coefficient function for y=beta_1 x1+beta_2 x2
def coef_12(n,sigma,g1,g2,beta1,beta2):
    X1=np.random.normal(0, sigma, n)+g1 #intervention on X1 in environment 1(for X1)
    Y=beta1*X1+np.random.normal(0,sigma, n)
    X2=beta2*Y+np.random.normal(0, 1, n)+g2 #intervention on X2 in environment 1(for X2)
    X=np.hstack([X1.reshape(-1,1),X2.reshape(-1,1)]) #fit the linear regression with X1 and X2
    y=np.hstack([Y.reshape(-1,1)])
    reg=LinearRegression().fit(X, y)  
    return np.append(reg.coef_,reg.intercept_)


# ## Shift X1

# In[53]:


# risk function of shifting on X2, estimating with X1
def risk_function1_same(n1,n2,sigma,x,iterate,g1,g2,beta1,beta2):
    list1=[]
    list2=[]
    list12=[]
 
    for i in range(1,iterate):
        X1=np.random.normal(0, sigma, n2)+x[0]
        Y=beta1*X1+np.random.normal(0,sigma, n2)
        X2=beta2*Y+np.random.normal(0, 1, n2)
        X=np.hstack([X1.reshape(-1,1)])
        #y=np.hstack([Y.reshape(-1,1)])
        pred_1=X1*coef_1(n1,sigma,g1,beta1,beta2)[0]+coef_1(n1,sigma,g1,beta1,beta2)[1]
        pred_2=X2*coef_2(n1,sigma,g2,beta1,beta2)[0]+coef_2(n1,sigma,g2,beta1,beta2)[1]
        pred_12=X1*coef_12(n1,sigma,g1,g2,beta1,beta2)[0]+X2*coef_12(n1,sigma,g1,g2,beta1,beta2)[1]+coef_12(n1,sigma,g1,g2,beta1,beta2)[2]
        #use the coefficients and intercept in envronment 1 for X1 to make prediction
        list1.append(0.5*sum((Y-pred_1)**2)) 
        list2.append(0.5*sum((Y-pred_2)**2)) 
        list12.append(0.5*sum((Y-pred_12)**2))
        l=np.append(sum(list1)/(iterate),sum(list2)/(iterate))
    return np.append(l,sum(list12)/(iterate))


# In[54]:


def shift_same_X1(beta1,beta2,n):
    a=np.linspace(0., 20., 30)
    b=np.linspace(0., 20., 30)
    m=np.hstack([a.reshape(-1,1),b.reshape(-1,1)])
    matrix=risk_function1_same(n,100,1,m[0],100,0,0,beta1,beta2)


   
    for i in range(1,30):
        matrix=np.column_stack((matrix,risk_function1_same(n,100,1,m[i],100,0,0,beta1,beta2)))
        
    list1=(matrix[0])
    
    list2=(matrix[1])
    list3=(matrix[2])

    

    plt.figure(figsize=(10,7))

    N = 30
    plt.rcParams['text.usetex'] = True

    colors = np.random.rand(N)
    area = (30 * 0.5)**2 
    plt.scatter(np.linspace(0., 20., 30), list1, s=area, alpha=0.5)
    plt.scatter(np.linspace(0., 20., 30), list2, s=area, alpha=0.5)
    plt.scatter(np.linspace(0., 20., 30), list3, s=area, alpha=0.5)
    classes = [ r'predict with $X_1$',r'predict with $X_2$',r'predict with $X_1$ and $X_2$']
    plt.legend(labels=classes)
    plt.xlabel(r'Shifting value ($g_1=g_2$)',fontsize=30)
    plt.ylabel("Risks",fontsize=30)

    #plt.legend(loc='best')
    message = f"Risk function of shifting $X_1$, $\\beta_1$={beta1}, $\\beta_2$={beta2}"
    plt.title(message,fontsize=30) 
    plt.show() 


# In[55]:


shift_same_X1(1,1,1000)


# In[56]:


shift_same_X1(1,10,100000)


# In[57]:


shift_same_X1(1,10,1000)


# In[58]:


shift_same_X1(10,1,1000)


# ### sample size=100000

# In[59]:


def shift_same_X1_subset(beta1,beta2,n1,n2):
    a=np.linspace(0., 20., 30)
    b=np.linspace(0., 20., 30)
    m=np.hstack([a.reshape(-1,1),b.reshape(-1,1)])
    matrix1=risk_function1_same(n1,100,1,m[0],100,0,0,beta1,beta2)
    matrix2=risk_function1_same(n2,100,1,m[0],100,0,0,beta1,beta2)


   
    for i in range(1,30):
        matrix1=np.column_stack((matrix1,risk_function1_same(n1,100,1,m[i],100,0,0,beta1,beta2)))
        matrix2=np.column_stack((matrix2,risk_function1_same(n2,100,1,m[i],100,0,0,beta1,beta2)))
        
    list1=(matrix1[0])
    list2=(matrix2[0])
    
    

    plt.figure(figsize=(10,7))

    N = 30

    colors = np.random.rand(N)
    area = (30 * 0.5)**2 
    plt.scatter(np.linspace(0., 20., 30), list1, s=area, alpha=0.5)
    plt.scatter(np.linspace(0., 20., 30), list2, s=area, alpha=0.5)
    #plt.scatter(np.linspace(0., 20., 30), list2, s=area, alpha=0.5)
    #plt.scatter(np.linspace(0., 20., 30), list3, s=area, alpha=0.5)
    classes = [ 'sample size=1000','sample size=100000']
    plt.legend(labels=classes)
    plt.xlabel(r'Shifting value ($g_1=g_2$)',fontsize=30)
    plt.ylabel("Risks",fontsize=30)
    #plt.ylim([45, 75])

    #plt.legend(loc='best')
    message = message = f"Risk function of predicting with $X_1$, shifting $X_1$, $\\beta_1$={beta1}, $\\beta_2$={beta2}"
    plt.title(message,fontsize=25) 
    plt.show() 


# In[62]:


shift_same_X1_subset(1,10,1000,100000)


# ## Use same y and shift X2

# In[63]:


# risk function of shifting on X2, estimating with X1
def risk_function2_same(n1,n2,sigma,x,iterate,g1,g2,beta1,beta2):
    list1=[]
    list2=[]
    list12=[]
    for i in range(1,iterate):
        X1=np.random.normal(0, sigma, n2) 
        Y=beta1*X1+np.random.normal(0,sigma, n2)
        X2=beta2*Y+np.random.normal(0, 1, n2)+x[1]
        #X=np.hstack([X1.reshape(-1,1)])
        #y=np.hstack([Y.reshape(-1,1)])
        pred_1=X1*coef_1(n1,sigma,g1,beta1,beta2)[0]+coef_1(n1,sigma,g1,beta1,beta2)[1]
        pred_2=X2*coef_2(n1,sigma,g2,beta1,beta2)[0]+coef_2(n1,sigma,g2,beta1,beta2)[1]
        pred_12=X1*coef_12(n1,sigma,g1,g2,beta1,beta2)[0]+X2*coef_12(n1,sigma,g1,g2,beta1,beta2)[1]+coef_12(n1,sigma,g1,g2,beta1,beta2)[2]
        #use the coefficients and intercept in envronment 1 for X1 to make prediction
        list1.append(0.5*sum((Y-pred_1)**2)) 
        list2.append(0.5*sum((Y-pred_2)**2)) 
        list12.append(0.5*sum((Y-pred_12)**2))
        l=np.append(sum(list1)/(iterate),sum(list2)/(iterate))
    return np.append(l,sum(list12)/(iterate))


# In[64]:


def shift_same_X2(beta1,beta2,n):
    a=np.linspace(0., 20., 30)
    b=np.linspace(0., 20., 30)
    list1=[]
    list2=[]
    list3=[]
    m=np.hstack([a.reshape(-1,1),b.reshape(-1,1)])
    matrix=risk_function2_same(n,100,1,m[0],100,0,0,beta1,beta2)


   
    for i in range(1,30):
        matrix=np.column_stack((matrix,risk_function2_same(n,100,1,m[i],100,0,0,beta1,beta2)))
        
    list1=(matrix[0])
    list2=(matrix[1])
    list3=(matrix[2])



    

    plt.figure(figsize=(10,7))

    N = 30

    colors = np.random.rand(N)
    area = (30 * 0.5)**2 
    plt.scatter(np.linspace(0., 20., 30), list1, s=area, alpha=0.5)
    plt.scatter(np.linspace(0., 20., 30), list2, s=area, alpha=0.5)
    plt.scatter(np.linspace(0., 20., 30), list3, s=area, alpha=0.5)
    classes = [r'predicted with $X_1$', r'predicted with $X_2$',r'predicted with $X_1$ and $X_2$']
    plt.legend(labels=classes)
    plt.xlabel(r"Shifting value ($g_1=g_2$)",fontsize=30)
    plt.ylabel("Risks",fontsize=30)

    message = f"Risk function of shifting $X_2$, $\\beta_1$={beta1}, $\\beta_2$={beta2}"
    plt.title(message,fontsize=30) 
    plt.show() 


# In[65]:


shift_same_X2(1,1,1000)


# In[66]:


shift_same_X2(1,10,1000)


# In[67]:


shift_same_X2(10,1,1000)


# ## Shift X1 and X2

# In[68]:


# risk function of shifting on X2, estimating with X1
def risk_function12_same(n1,n2,sigma,x,iterate,g1,g2,beta1,beta2):
    list1=[]
    list2=[]
    list12=[]
    for i in range(1,iterate):
        X1=np.random.normal(0, sigma, n2)+x[0]
        Y=beta1*X1+np.random.normal(0,sigma, n2)
        X2=beta2*Y+np.random.normal(0, 1, n2)+x[1]
        #X=np.hstack([X1.reshape(-1,1)])
        #y=np.hstack([Y.reshape(-1,1)])
        pred_1=X1*coef_1(n1,sigma,g1,beta1,beta2)[0]+coef_1(n1,sigma,g1,beta1,beta2)[1]
        pred_2=X2*coef_2(n1,sigma,g2,beta1,beta2)[0]+coef_2(n1,sigma,g2,beta1,beta2)[1]
        pred_12=X1*coef_12(n1,sigma,g1,g2,beta1,beta2)[0]+X2*coef_12(n1,sigma,g1,g2,beta1,beta2)[1]+coef_12(n1,sigma,g1,g2,beta1,beta2)[2]
        #use the coefficients and intercept in envronment 1 for X1 to make prediction
        list1.append(0.5*sum((Y-pred_1)**2)) 
        list2.append(0.5*sum((Y-pred_2)**2)) 
        list12.append(0.5*sum((Y-pred_12)**2))
        l=np.append(sum(list1)/(iterate),sum(list2)/(iterate))
    return np.append(l,sum(list12)/(iterate))


# In[69]:


def shift_same_X12(beta1,beta2,n):
    a=np.linspace(0., 20., 30)
    b=np.linspace(0., 20., 30)
    list1=[]
    list2=[]
    list3=[]
    m=np.hstack([a.reshape(-1,1),b.reshape(-1,1)])
    matrix=risk_function12_same(n,100,1,m[0],100,0,0,beta1,beta2)


   
    for i in range(1,30):
        matrix=np.column_stack((matrix,risk_function12_same(n,100,1,m[i],100,0,0,beta1,beta2)))
        
    list1=(matrix[0])
    list2=(matrix[1])
    list3=(matrix[2])

    

    plt.figure(figsize=(10,7))

    N = 30

    colors = np.random.rand(N)
    area = (30 * 0.5)**2 
    plt.scatter(np.linspace(0., 20., 30), list1, s=area, alpha=0.5)
    plt.scatter(np.linspace(0., 20., 30), list2, s=area, alpha=0.5)
    plt.scatter(np.linspace(0., 20., 30), list3, s=area, alpha=0.5)
    classes = [r'predicted with $X_1$', r'predicted with $X_2$',r'predicted with $X_1$ and $X_2$']
    plt.legend(labels=classes)
    plt.xlabel(r"Shifting value ($g_1=g_2$)",fontsize=30)
    plt.ylabel("Risks",fontsize=30)

    message = f"Risk function of shifting $X_1$ and $X_2$, $\\beta_1$={beta1}, $\\beta_2$={beta2}"
    plt.title(message,fontsize=30) 
    plt.show() 


# In[70]:


shift_same_X12(1,1,1000)


# In[71]:


shift_same_X12(1,10,1000)


# In[72]:


shift_same_X12(10,1,1000)


# In[ ]:




