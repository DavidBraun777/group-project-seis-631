# Exploratory Data Analysis of User Behavior Dataset


This notebook explores the **User Behavior Dataset** by analyzing various factors such as device usage, 
battery drain, data usage, and user demographics. We will perform basic exploratory data analysis (EDA) to uncover trends, 
relationships, and insights within the data.

The following steps will be covered:
1. Data Overview
2. Descriptive Statistics
3. Data Visualization
4. Insights and Conclusion


## 1. Data Overview


```python

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('user_behavior_dataset.csv')

# Display the first few rows
data.head()

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
      <th>User ID</th>
      <th>Device Model</th>
      <th>Operating System</th>
      <th>App Usage Time (min/day)</th>
      <th>Screen On Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th>Number of Apps Installed</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>Gender</th>
      <th>User Behavior Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Google Pixel 5</td>
      <td>Android</td>
      <td>393</td>
      <td>6.4</td>
      <td>1872</td>
      <td>67</td>
      <td>1122</td>
      <td>40</td>
      <td>Male</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>OnePlus 9</td>
      <td>Android</td>
      <td>268</td>
      <td>4.7</td>
      <td>1331</td>
      <td>42</td>
      <td>944</td>
      <td>47</td>
      <td>Female</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Xiaomi Mi 11</td>
      <td>Android</td>
      <td>154</td>
      <td>4.0</td>
      <td>761</td>
      <td>32</td>
      <td>322</td>
      <td>42</td>
      <td>Male</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Google Pixel 5</td>
      <td>Android</td>
      <td>239</td>
      <td>4.8</td>
      <td>1676</td>
      <td>56</td>
      <td>871</td>
      <td>20</td>
      <td>Male</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>iPhone 12</td>
      <td>iOS</td>
      <td>187</td>
      <td>4.3</td>
      <td>1367</td>
      <td>58</td>
      <td>988</td>
      <td>31</td>
      <td>Female</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Descriptive Statistics


```python

# Check for missing values and data types
data.info()

# Generate descriptive statistics for numerical columns
data.describe()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 700 entries, 0 to 699
    Data columns (total 11 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   User ID                     700 non-null    int64  
     1   Device Model                700 non-null    object 
     2   Operating System            700 non-null    object 
     3   App Usage Time (min/day)    700 non-null    int64  
     4   Screen On Time (hours/day)  700 non-null    float64
     5   Battery Drain (mAh/day)     700 non-null    int64  
     6   Number of Apps Installed    700 non-null    int64  
     7   Data Usage (MB/day)         700 non-null    int64  
     8   Age                         700 non-null    int64  
     9   Gender                      700 non-null    object 
     10  User Behavior Class         700 non-null    int64  
    dtypes: float64(1), int64(7), object(3)
    memory usage: 60.3+ KB





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
      <th>User ID</th>
      <th>App Usage Time (min/day)</th>
      <th>Screen On Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th>Number of Apps Installed</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>User Behavior Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>700.00000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>350.50000</td>
      <td>271.128571</td>
      <td>5.272714</td>
      <td>1525.158571</td>
      <td>50.681429</td>
      <td>929.742857</td>
      <td>38.482857</td>
      <td>2.990000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>202.21688</td>
      <td>177.199484</td>
      <td>3.068584</td>
      <td>819.136414</td>
      <td>26.943324</td>
      <td>640.451729</td>
      <td>12.012916</td>
      <td>1.401476</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>302.000000</td>
      <td>10.000000</td>
      <td>102.000000</td>
      <td>18.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>175.75000</td>
      <td>113.250000</td>
      <td>2.500000</td>
      <td>722.250000</td>
      <td>26.000000</td>
      <td>373.000000</td>
      <td>28.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>350.50000</td>
      <td>227.500000</td>
      <td>4.900000</td>
      <td>1502.500000</td>
      <td>49.000000</td>
      <td>823.500000</td>
      <td>38.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>525.25000</td>
      <td>434.250000</td>
      <td>7.400000</td>
      <td>2229.500000</td>
      <td>74.000000</td>
      <td>1341.000000</td>
      <td>49.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>700.00000</td>
      <td>598.000000</td>
      <td>12.000000</td>
      <td>2993.000000</td>
      <td>99.000000</td>
      <td>2497.000000</td>
      <td>59.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Data Visualization


```python
# Distribution of App Usage Time
plt.figure(figsize=(10, 6))
sns.histplot(data['App Usage Time (min/day)'], kde=True, bins=30)
plt.title('Distribution of App Usage Time (min/day)')
plt.xlabel('App Usage Time (min/day)')
plt.ylabel('Frequency')
plt.show()

```


    
![png](output_7_0.png)
    



```python

# Screen On Time vs Battery Drain
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Screen On Time (hours/day)', y='Battery Drain (mAh/day)', hue='Operating System', data=data)
plt.title('Screen On Time vs Battery Drain')
plt.xlabel('Screen On Time (hours/day)')
plt.ylabel('Battery Drain (mAh/day)')
plt.legend(title='Operating System')
plt.show()

```


    
![png](output_8_0.png)
    



```python

# Age Distribution by User Behavior Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='User Behavior Class', y='Age', data=data)
plt.title('Age Distribution by User Behavior Class')
plt.xlabel('User Behavior Class')
plt.ylabel('Age')
plt.show()

```


    
![png](output_9_0.png)
    



```python

# Data Usage by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Data Usage (MB/day)', data=data)
plt.title('Data Usage by Gender')
plt.xlabel('Gender')
plt.ylabel('Data Usage (MB/day)')
plt.show()

```


    
![png](output_10_0.png)
    


## 4. Insights and Conclusion


In this exploratory data analysis, we observed various trends and patterns within the dataset, including:
- The distribution of app usage time among users,
- The relationship between screen on time and battery drain across operating systems,
- Age distributions for different user behavior classes,
- Data usage patterns based on gender.

These insights provide a foundational understanding of user behavior, which could be beneficial for further analysis 
or machine learning model development.

