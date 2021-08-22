# Lasso Regresion

#Requaired libraries
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Dataset
boston = load_boston()
x = boston.data
y = boston.target

#Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x,y)

# Coef
lasso_coef = lasso.coef_

#plot
plt.plot(range(13), lasso_coef)
plt.xticks(range(13), boston.feature_names)
plt.ylabel("coefficents")