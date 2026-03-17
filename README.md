# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as pandas, sklearn, matplotlib, and seaborn and Load the dataset HeightsWeights.csv.
2. Select the features Height (Inches) and Weight (Pounds).
3. Visualize the original data distribution using a scatter plot.
4. Apply StandardScaler to standardize the feature values and Apply PCA to reduce the dimensionality of the data.
5. Calculate the explained variance ratio of the principal components and Transform the dataset into principal components.
Visualize the PCA-transformed data using a scatter plot.

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: VIJAYAPRATHISHA J
RegisterNumber: 212225240184
*/
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('HeightsWeights.csv')
print(data.head())
X=data[['Height(Inches)','Weight(Pounds)']]
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)',y='Weight(Pounds)',data=data)
plt.title("Original Data Distribution")
plt.show()
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
print("Explained Variance Ratio:",pca.explained_variance_ratio_)
pca_df=pd.DataFrame(X_pca,columns=['PC1','PC2'])
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1',y='PC2',data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:
<img width="633" height="168" alt="image" src="https://github.com/user-attachments/assets/b30ce8bd-1608-49d5-a22c-323e69c63cf3" />

<img width="769" height="612" alt="image" src="https://github.com/user-attachments/assets/1a74232c-c7df-4c46-aeab-524c23987b09" />

<img width="744" height="604" alt="image" src="https://github.com/user-attachments/assets/d84b780f-4d77-4842-bd00-c01af5b9b17d" />

## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
