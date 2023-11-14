## 1. Various Machine Learning Algorithms
#### Introducing various algorithms in machine learning.

## 2. Machine Learning Algorithms Guided by Scikit-learn
#### We examine how Scikit-learn classifies and guides machine learning algorithms.

## 3. Hello Scikit-learn
#### We will install Scikit-learn and take a closer look at it.

## 4. Key Modules in Scikit-learn
#### Data Representation Methods
#### Regression Model Practice
#### Dataset Modules
#### Practice Classification Problems Using Scikit-learn Datasets
#### Estimator

## 5. Separating Training Data and Test Data
#### We will separate the training and test data ourselves and briefly practice the subsequent machine learning process.

### 1. Various Machine Learning Algorithms

Machine learning algorithms can be broadly divided into three types:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Machine learning is typically divided into supervised and unsupervised learning, depending on the presence or absence of labels (correct answers).

If the data is very complex, dimension reduction might be applied, and if there are significant components, principal component analysis may be used.

Moreover, depending on the type of data, even within regression, linear regression is used for predicting numerical data (continuous data), while logistic regression is used for classification (strictly speaking, binary classification).

Therefore, depending on the presence of correct answers, the type of data, characteristics, and problem definition, machine learning algorithms are used in a highly complex manner.

Reinforcement learning is a different type of algorithm from the supervised and unsupervised learning mentioned earlier. The system being trained is referred to as an agent, which observes the environment and acts on its own. The model learns to maximize the rewards it receives as a result. 

### 2. Machine Learning Algorithms Guided by Scikit-learn
https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
This Scikit-learn documentation provides a comprehensive list of classification and regression algorithms. Here is a summary of the types and numbers of these algorithms: 

Linear Models: These are used for regression tasks.

Linear and Quadratic Discriminant Analysis: Useful in statistical classification.

Kernel Ridge Regression: A method that combines ridge regression with the kernel trick.

Support Vector Machines (SVMs): These are used for both classification and regression tasks.

Stochastic Gradient Descent: Applicable to both classification and regression.

Nearest Neighbors: This includes algorithms for unsupervised nearest neighbors, classification, and regression.

Gaussian Processes: These offer methods for regression and classification.

Cross Decomposition: Techniques for regression.

Naive Bayes: Includes Gaussian Naive Bayes, Multinomial Naive Bayes, etc., mostly used for classification.

Decision Trees: Useful in both classification and regression.

Ensembles: This includes methods like Gradient Boosting, Random Forests, Bagging, etc., for both classification and regression.

Multiclass and Multioutput Algorithms: Cater to complex output predictions.

Feature Selection: Techniques to select features for modeling.

Semi-Supervised Learning: Algorithms that work with partially labeled data.

Isotonic Regression: A regression algorithm.

Probability Calibration: Techniques for calibrating classifiers.

Neural Network Models (Supervised): Includes multi-layer perceptron models for classification and regression.


These algorithms cover a wide range of machine learning tasks and are integral to solving various types of problems in both classification and regression domains

### 3. Hello Scikit-learn
Reference: 
https://scikit-learn.org/stable/index.html

https://youtu.be/rvVkVsG49uU

https://youtu.be/WCEXYvv-T5Q

ETL (Extract Transform Load) Functions in Scikit-Learn: Scikit-Learn doesn't have a specific function labeled as "ETL", but it does offer a variety of transformers for data preprocessing, dimensionality reduction, feature expansion, and feature extraction. These functions collectively can perform ETL-like tasks.

Model Classes in Scikit-Learn: Model classes in Scikit-Learn are typically represented as estimators. These include various supervised learning models like linear models, support vector machines, and ensemble methods, as well as unsupervised learning models like clustering algorithms and manifold learning methods.

Methods of Estimator Classes: Estimator classes in Scikit-Learn usually have methods like fit, predict, and score. Some estimators may also have transform methods, especially if they're used for data preprocessing.

APIs Performing Estimator and Transformer Functions: In Scikit-Learn, there are several APIs that can perform both estimator and transformer functions. These are often found in the preprocessing and feature extraction modules.

### 4. Key Modules in Scikit-learn: Data Representation Methods
https://scikit-learn.org/stable/modules/classes.html
This is a list of API References. 

When it comes to representing data, Scikit-learn provides ndarray from NumPy, DataFrame from Pandas, and Sparse Matrix from SciPy. 
Generally, we represent data in two ways: feature matrix and target vector. 

- Feature Matrix

Represents the input data.

Feature: Refers to individual observations in the data, represented as numeric, discrete, or boolean values. In the feature matrix, these are the values in columns.

Sample: Each piece of input data. In the feature matrix, these correspond to the rows.

n_samples: The number of rows (number of samples).

n_features: The number of columns (number of features).

X: The feature matrix is typically denoted as X.

The structure of [n_samples, n_features] is a 2D array, which can be represented using NumPy's ndarray, Pandas' DataFrame, or SciPy's Sparse Matrix.


- Target Vector

Represents the labels (answers) of the input data.

Target: Also known as the label or target value, it is what you aim to predict from the feature matrix.

n_samples: The length of the vector (number of labels).

In the target vector, there are no n_features.

y: The target vector is typically denoted as y.

The target vector is usually represented as a 1D vector, which can be represented using NumPy's ndarray or Pandas' Series.

(However, the target vector might not always be represented in 1D. In this node, all examples use a 1D vector.)

The n_samples of the feature matrix X and the target vector y must be the same.


### 4. Key Modules in Scikit-learn: Regression Model Practice


```python
import numpy as np
import matplotlib.pyplot as plt
r = np.random.RandomState(10)
x = 10 * r.rand(100)
y = 2 * x - 3 * r.rand(100)
plt.scatter(x,y)
```




    <matplotlib.collections.PathCollection at 0x7fbd1de59130>




    
![png](output_6_1.png)
    



```python
x.shape
```




    (100,)




```python
y.shape
```




    (100,)




```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model
```




    LinearRegression()




```python
# to use fit() method, you have to change x into matrix. 
X = x.reshape(100,1)
model.fit(X,y)

# now you completed training using input data and its label. Now, let's predict with new data. 
```




    LinearRegression()




```python
x_new = np.linspace(-1, 11, 100)
X_new = x_new.reshape(100,1)
y_new = model.predict(X_new)
```


```python
X_ = x_new.reshape(-1,1)
X_.shape
```




    (100, 1)




```python
plt.scatter(x, y, label='input data')
plt.plot(X_new, y_new, color='red', label='regression line')
```




    [<matplotlib.lines.Line2D at 0x7fbcf345cbb0>]




    
![png](output_13_1.png)
    


### 4. Key Modules in Scikit-learn: Dataset Modules


```python
from sklearn.datasets import load_wine
data = load_wine()
type(data)
```




    sklearn.utils.Bunch




```python
data.keys()
data.data
data.data.shape
data.data.ndim
data.target 
data.target.shape
data.feature_names
data.target_names
print(data.DESCR)
```

    .. _wine_dataset:
    
    Wine recognition dataset
    ------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 178 (50 in each of three classes)
        :Number of Attributes: 13 numeric, predictive attributes and the class
        :Attribute Information:
     		- Alcohol
     		- Malic acid
     		- Ash
    		- Alcalinity of ash  
     		- Magnesium
    		- Total phenols
     		- Flavanoids
     		- Nonflavanoid phenols
     		- Proanthocyanins
    		- Color intensity
     		- Hue
     		- OD280/OD315 of diluted wines
     		- Proline
    
        - class:
                - class_0
                - class_1
                - class_2
    		
        :Summary Statistics:
        
        ============================= ==== ===== ======= =====
                                       Min   Max   Mean     SD
        ============================= ==== ===== ======= =====
        Alcohol:                      11.0  14.8    13.0   0.8
        Malic Acid:                   0.74  5.80    2.34  1.12
        Ash:                          1.36  3.23    2.36  0.27
        Alcalinity of Ash:            10.6  30.0    19.5   3.3
        Magnesium:                    70.0 162.0    99.7  14.3
        Total Phenols:                0.98  3.88    2.29  0.63
        Flavanoids:                   0.34  5.08    2.03  1.00
        Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
        Proanthocyanins:              0.41  3.58    1.59  0.57
        Colour Intensity:              1.3  13.0     5.1   2.3
        Hue:                          0.48  1.71    0.96  0.23
        OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
        Proline:                       278  1680     746   315
        ============================= ==== ===== ======= =====
    
        :Missing Attribute Values: None
        :Class Distribution: class_0 (59), class_1 (71), class_2 (48)
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML Wine recognition datasets.
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    
    The data is the results of a chemical analysis of wines grown in the same
    region in Italy by three different cultivators. There are thirteen different
    measurements taken for different constituents found in the three types of
    wine.
    
    Original Owners: 
    
    Forina, M. et al, PARVUS - 
    An Extendible Package for Data Exploration, Classification and Correlation. 
    Institute of Pharmaceutical and Food Analysis and Technologies,
    Via Brigata Salerno, 16147 Genoa, Italy.
    
    Citation:
    
    Lichman, M. (2013). UCI Machine Learning Repository
    [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science. 
    
    .. topic:: References
    
      (1) S. Aeberhard, D. Coomans and O. de Vel, 
      Comparison of Classifiers in High Dimensional Settings, 
      Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of  
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Technometrics). 
    
      The data was used with many others for comparing various 
      classifiers. The classes are separable, though only RDA 
      has achieved 100% correct classification. 
      (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data)) 
      (All results using the leave-one-out technique) 
    
      (2) S. Aeberhard, D. Coomans and O. de Vel, 
      "THE CLASSIFICATION PERFORMANCE OF RDA" 
      Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of 
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Journal of Chemometrics).
    


### 5. Separating Training Data and Test Data



```python
import pandas as pd

pd.DataFrame(data.data, columns=data.feature_names)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>173</th>
      <td>13.71</td>
      <td>5.65</td>
      <td>2.45</td>
      <td>20.5</td>
      <td>95.0</td>
      <td>1.68</td>
      <td>0.61</td>
      <td>0.52</td>
      <td>1.06</td>
      <td>7.70</td>
      <td>0.64</td>
      <td>1.74</td>
      <td>740.0</td>
    </tr>
    <tr>
      <th>174</th>
      <td>13.40</td>
      <td>3.91</td>
      <td>2.48</td>
      <td>23.0</td>
      <td>102.0</td>
      <td>1.80</td>
      <td>0.75</td>
      <td>0.43</td>
      <td>1.41</td>
      <td>7.30</td>
      <td>0.70</td>
      <td>1.56</td>
      <td>750.0</td>
    </tr>
    <tr>
      <th>175</th>
      <td>13.27</td>
      <td>4.28</td>
      <td>2.26</td>
      <td>20.0</td>
      <td>120.0</td>
      <td>1.59</td>
      <td>0.69</td>
      <td>0.43</td>
      <td>1.35</td>
      <td>10.20</td>
      <td>0.59</td>
      <td>1.56</td>
      <td>835.0</td>
    </tr>
    <tr>
      <th>176</th>
      <td>13.17</td>
      <td>2.59</td>
      <td>2.37</td>
      <td>20.0</td>
      <td>120.0</td>
      <td>1.65</td>
      <td>0.68</td>
      <td>0.53</td>
      <td>1.46</td>
      <td>9.30</td>
      <td>0.60</td>
      <td>1.62</td>
      <td>840.0</td>
    </tr>
    <tr>
      <th>177</th>
      <td>14.13</td>
      <td>4.10</td>
      <td>2.74</td>
      <td>24.5</td>
      <td>96.0</td>
      <td>2.05</td>
      <td>0.76</td>
      <td>0.56</td>
      <td>1.35</td>
      <td>9.20</td>
      <td>0.61</td>
      <td>1.60</td>
      <td>560.0</td>
    </tr>
  </tbody>
</table>
<p>178 rows × 13 columns</p>
</div>




```python
X = data.data
y = data.target
```


```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
```


```python
model.fit(X, y)
```




    RandomForestClassifier()




```python
y_pred = model.predict(X)
```


```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(classification_report(y, y_pred))

print("accuracy = ", accuracy_score(y, y_pred))
# The reason why it gave the accuracy of 1.0 is because we used same data for fit() and prediction() method. 
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        59
               1       1.00      1.00      1.00        71
               2       1.00      1.00      1.00        48
    
        accuracy                           1.00       178
       macro avg       1.00      1.00      1.00       178
    weighted avg       1.00      1.00      1.00       178
    
    accuracy =  1.0



```python
from sklearn.datasets import load_wine
data = load_wine()
print(data.data.shape)
print(data.target.shape)
```

    (178, 13)
    (178,)



```python
X_train = data.data[:142]
X_test = data.data[142:]
print(X_train.shape, X_test.shape)
```

    (142, 13) (36, 13)



```python
y_train = data.target[:142]
y_test = data.target[142:]
print(y_train.shape, y_test.shape)
```

    (142,) (36,)



```python
# Now that we finished spliting train and test data, we will resume train and prediction.
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```




    RandomForestClassifier()




```python
y_pred = model.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score

print("accuracy=", accuracy_score(y_test, y_pred))
```

    accuracy= 0.9166666666666666



```python
# we could also use train_test_split()
from sklearn.model_selection import train_test_split

result = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
print(type(result))
print(len(result))
```

    <class 'list'>
    4



```python
print(result[0].shape)
print(result[1].shape)
print(result[2].shape)
print(result[3].shape)
# from top to bottom, they are feature matrix for training data, 
# feature matrix for test data, target vector for training data, 
# and target vector for test data
```

    (142, 13)
    (36, 13)
    (142,)
    (36,)



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python

```
