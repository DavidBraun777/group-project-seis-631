# Exploratory data analysis in Python.

## Let us understand how to explore the data in python.



![alt text](https://moriohcdn.b-cdn.net/ff3cc511fb.png)


Image Credits: Morioh

## Introduction

**What is Exploratory Data Analysis ?**

Exploratory Data Analysis or (EDA) is understanding the data sets by summarizing their main characteristics often plotting them visually. This step is very important especially when we arrive at modeling the data in order to apply Machine learning. Plotting in EDA consists of Histograms, Box plot, Scatter plot and many more. It often takes much time to explore the data. Through the process of EDA, we can ask to define the problem statement or definition on our data set which is very important.

**How to perform Exploratory Data Analysis ?**

This is one such question that everyone is keen on knowing the answer. Well, the answer is it depends on the data set that you are working. There is no one method or common methods in order to perform EDA, whereas in this tutorial you can understand some common methods and plots that would be used in the EDA process.

**What data are we exploring today ?**


We obtained a fascinating dataset about mobile usage. The data-set can be downloaded from [here](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset). This dataset has 700 rows and 11 columns detailing various aspects of user interactions with their devices. It includes features like Device Model, Operating System, App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day), Number of Apps Installed, Data Usage (MB/day), Age, Gender, and User Behavior Class. In this tutorial, we’ll explore this data and prepare it for modeling. 



---



## 1. Importing the required libraries for EDA

Below are the libraries that are used in order to perform EDA (Exploratory data analysis) in this tutorial.


```python
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
%matplotlib inline     
sns.set(color_codes=True)
```



---



## 2. Loading the data into the data frame.

Loading the data into the pandas data frame is certainly one of the most important steps in EDA, as we can see that the value from the data set is comma-separated. So all we have to do is to just read the CSV into a data frame and pandas data frame does the job for us.

To get or load the dataset into the notebook, all I did was one trivial step. In Google Colab at the left-hand side of the notebook, you will find a > (greater than symbol). When you click that you will find a tab with three options, you just have to select Files. Then you can easily upload your file with the help of the Upload option. No need to mount to the google drive or use any specific libraries just upload the data set and your job is done. One thing to remember in this step is that uploaded files will get deleted when this runtime is recycled. This is how I got the data set into the notebook.

For most of the team we essentially did the same thing for this section. In the following sections we included what we each did with comments. Please note the sections and names.


```python
df = pd.read_csv("./user_behavior_dataset.csv")
# To display the top 5 rows 
df.head(5)               
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




```python
df.tail(5)                        # To display the botton 5 rows
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
      <th>695</th>
      <td>696</td>
      <td>iPhone 12</td>
      <td>iOS</td>
      <td>92</td>
      <td>3.9</td>
      <td>1082</td>
      <td>26</td>
      <td>381</td>
      <td>22</td>
      <td>Male</td>
      <td>2</td>
    </tr>
    <tr>
      <th>696</th>
      <td>697</td>
      <td>Xiaomi Mi 11</td>
      <td>Android</td>
      <td>316</td>
      <td>6.8</td>
      <td>1965</td>
      <td>68</td>
      <td>1201</td>
      <td>59</td>
      <td>Male</td>
      <td>4</td>
    </tr>
    <tr>
      <th>697</th>
      <td>698</td>
      <td>Google Pixel 5</td>
      <td>Android</td>
      <td>99</td>
      <td>3.1</td>
      <td>942</td>
      <td>22</td>
      <td>457</td>
      <td>50</td>
      <td>Female</td>
      <td>2</td>
    </tr>
    <tr>
      <th>698</th>
      <td>699</td>
      <td>Samsung Galaxy S21</td>
      <td>Android</td>
      <td>62</td>
      <td>1.7</td>
      <td>431</td>
      <td>13</td>
      <td>224</td>
      <td>44</td>
      <td>Male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>699</th>
      <td>700</td>
      <td>OnePlus 9</td>
      <td>Android</td>
      <td>212</td>
      <td>5.4</td>
      <td>1306</td>
      <td>49</td>
      <td>828</td>
      <td>23</td>
      <td>Female</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>





---



## 3. Dropping the duplicate rows, missing or null value, and irrelevant columns - Soad Ahmed


```python
df = df.drop(['Operating System','App Usage Time (min/day)','Battery Drain (mAh/day)'],axis=1)
df.head(5)
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
      <th>Screen On Time (hours/day)</th>
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
      <td>6.4</td>
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
      <td>4.7</td>
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
      <td>4.0</td>
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
      <td>4.8</td>
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
      <td>4.3</td>
      <td>58</td>
      <td>988</td>
      <td>31</td>
      <td>Female</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (700, 8)




```python
df.count()
```




    User ID                       700
    Device Model                  700
    Screen On Time (hours/day)    700
    Number of Apps Installed      700
    Data Usage (MB/day)           700
    Age                           700
    Gender                        700
    User Behavior Class           700
    dtype: int64




```python
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
```

    number of duplicate rows:  (0, 8)
    


```python
df = df.drop_duplicates()
df.head(5)
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
      <th>Screen On Time (hours/day)</th>
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
      <td>6.4</td>
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
      <td>4.7</td>
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
      <td>4.0</td>
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
      <td>4.8</td>
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
      <td>4.3</td>
      <td>58</td>
      <td>988</td>
      <td>31</td>
      <td>Female</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.count()
```




    User ID                       700
    Device Model                  700
    Screen On Time (hours/day)    700
    Number of Apps Installed      700
    Data Usage (MB/day)           700
    Age                           700
    Gender                        700
    User Behavior Class           700
    dtype: int64




```python
print(df.isnull().sum())
```

    User ID                       0
    Device Model                  0
    Screen On Time (hours/day)    0
    Number of Apps Installed      0
    Data Usage (MB/day)           0
    Age                           0
    Gender                        0
    User Behavior Class           0
    dtype: int64
    


```python
df = df.dropna()    # Dropping the missing values.
df.count()
```




    User ID                       700
    Device Model                  700
    Screen On Time (hours/day)    700
    Number of Apps Installed      700
    Data Usage (MB/day)           700
    Age                           700
    Gender                        700
    User Behavior Class           700
    dtype: int64




```python
print(df.isnull().sum())# After dropping the values
```

    User ID                       0
    Device Model                  0
    Screen On Time (hours/day)    0
    Number of Apps Installed      0
    Data Usage (MB/day)           0
    Age                           0
    Gender                        0
    User Behavior Class           0
    dtype: int64
    

## 4. Dropping irrelevant columns - Soad Ahmed


```python
df = df.rename(columns={"User ID": "ID", "Device Model": "Model", "Screen On Time (hours/day)": "Screen Time", "Data Usage (MB/day)": "Usage","User Behavior Class": "Behavior","Number of Apps Installed": "Apps Installed" })
df.head(5) 
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
      <th>ID</th>
      <th>Model</th>
      <th>Screen Time</th>
      <th>Apps Installed</th>
      <th>Usage</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Behavior</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Google Pixel 5</td>
      <td>6.4</td>
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
      <td>4.7</td>
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
      <td>4.0</td>
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
      <td>4.8</td>
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
      <td>4.3</td>
      <td>58</td>
      <td>988</td>
      <td>31</td>
      <td>Female</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df
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
      <th>ID</th>
      <th>Model</th>
      <th>Screen Time</th>
      <th>Apps Installed</th>
      <th>Usage</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Behavior</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Google Pixel 5</td>
      <td>6.4</td>
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
      <td>4.7</td>
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
      <td>4.0</td>
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
      <td>4.8</td>
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
      <td>4.3</td>
      <td>58</td>
      <td>988</td>
      <td>31</td>
      <td>Female</td>
      <td>3</td>
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
    </tr>
    <tr>
      <th>695</th>
      <td>696</td>
      <td>iPhone 12</td>
      <td>3.9</td>
      <td>26</td>
      <td>381</td>
      <td>22</td>
      <td>Male</td>
      <td>2</td>
    </tr>
    <tr>
      <th>696</th>
      <td>697</td>
      <td>Xiaomi Mi 11</td>
      <td>6.8</td>
      <td>68</td>
      <td>1201</td>
      <td>59</td>
      <td>Male</td>
      <td>4</td>
    </tr>
    <tr>
      <th>697</th>
      <td>698</td>
      <td>Google Pixel 5</td>
      <td>3.1</td>
      <td>22</td>
      <td>457</td>
      <td>50</td>
      <td>Female</td>
      <td>2</td>
    </tr>
    <tr>
      <th>698</th>
      <td>699</td>
      <td>Samsung Galaxy S21</td>
      <td>1.7</td>
      <td>13</td>
      <td>224</td>
      <td>44</td>
      <td>Male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>699</th>
      <td>700</td>
      <td>OnePlus 9</td>
      <td>5.4</td>
      <td>49</td>
      <td>828</td>
      <td>23</td>
      <td>Female</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>700 rows × 8 columns</p>
</div>



## 5. Detecting Outliers - Soad Ahmed


```python
sns.boxplot(x=df['Screen Time'])
```




    <Axes: xlabel='Screen Time'>




    
![png](output_33_1.png)
    



```python
sns.boxplot(x=df['Usage'])
```




    <Axes: xlabel='Usage'>




    
![png](output_34_1.png)
    



```python
sns.boxplot(x=df['Behavior'])
```




    <Axes: xlabel='Behavior'>




    
![png](output_35_1.png)
    


## 6. Plot against frequency (histogram)- Soad Ahmed


```python
plt.figure(figsize=(10, 6))
plt.hist(df['Usage'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Usage')
plt.xlabel('Usage')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
```


    
![png](output_37_0.png)
    



```python
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
```




    Text(0, 0.5, 'Frequency')




    
![png](output_38_1.png)
    



```python
plt.figure(figsize=(10, 6))
plt.hist(df['Usage'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Data Usage (MB/day)')
plt.xlabel('Usage')
plt.ylabel('Frequency')
```




    Text(0, 0.5, 'Frequency')




    
![png](output_39_1.png)
    


## 7. Plot against one another (scatter) - Soad Ahmed


```python
plt.figure(figsize=(10, 6))
plt.scatter(df['Screen Time'], df['Usage'], alpha=0.7)
plt.title('Screen Time vs Usage')
plt.xlabel('Screen Time')
plt.ylabel('Usage')
plt.grid(True)
plt.show()
```


    
![png](output_41_0.png)
    



```python
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Apps Installed'], alpha=0.7)  
plt.title('Apps Installed vs. Age') 
plt.xlabel('Age')
plt.ylabel('Apps Installed') 
plt.grid(True) 
plt.show() 
```


    
![png](output_42_0.png)
    




---



## 3. Checking the types of data - Cristian Zendejas


```python
# resetting the data for this section
df = pd.read_csv("./user_behavior_dataset.csv")
```

Checking the data types to make sure they are all in the formats we expect so we can use them.


```python
df.dtypes
```




    User ID                         int64
    Device Model                   object
    Operating System               object
    App Usage Time (min/day)        int64
    Screen On Time (hours/day)    float64
    Battery Drain (mAh/day)         int64
    Number of Apps Installed        int64
    Data Usage (MB/day)             int64
    Age                             int64
    Gender                         object
    User Behavior Class             int64
    dtype: object



## 4. Dropping irrelevant columns - Cristian Zendejas

#### I didn't think the user behavior class was valuable to us since we don't know the original definition of what this column means. So I decided to remove it.


```python
df = df.drop(['User Behavior Class'], axis=1)
df.head(5)
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
    </tr>
  </tbody>
</table>
</div>



## 5. Renaming the columns - Cristian Zendejas

In this instance, most of the column names are very confusing to read, so I just tweaked their column names. This is a good approach it improves the readability of the data set.

I decided to rename some of the columns to something simpler for visualization purposes.


```python
df = df.rename(columns={"Device Model": "Device", "Operating System": "OS", "Number of Apps Installed": "# of Apps Installed"})
df.head(5)
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
      <th>Device</th>
      <th>OS</th>
      <th>App Usage Time (min/day)</th>
      <th>Screen On Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th># of Apps Installed</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>Gender</th>
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
    </tr>
  </tbody>
</table>
</div>





---



## 6. Dropping the duplicate rows - Cristian Zendejas
This is often a handy thing to do because a huge data set as in this case contains more than 10, 000 rows often have some duplicate data which might be disturbing, so here I remove all the duplicate value from the data-set. 
This dataset we decided to use did not have much duplicated data if any at all. So most of the following processes didn't really change the data much.

```python
df.shape
```




    (700, 8)




```python
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
```

    number of duplicate rows:  (0, 10)
    

Now let us remove the duplicate data because it's ok to remove them.


```python
df.count()      # Used to count the number of rows
```




    User ID                       700
    Device                        700
    OS                            700
    App Usage Time (min/day)      700
    Screen On Time (hours/day)    700
    Battery Drain (mAh/day)       700
    # of Apps Installed           700
    Data Usage (MB/day)           700
    Age                           700
    Gender                        700
    dtype: int64



##### no duplicates for this data!


```python
df = df.drop_duplicates()
df.head(5)
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
      <th>Device</th>
      <th>OS</th>
      <th>App Usage Time (min/day)</th>
      <th>Screen On Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th># of Apps Installed</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>Gender</th>
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
    </tr>
  </tbody>
</table>
</div>




```python
# We can see that dropping duplicates made no difference.
df.count()
```




    User ID                       700
    Device                        700
    OS                            700
    App Usage Time (min/day)      700
    Screen On Time (hours/day)    700
    Battery Drain (mAh/day)       700
    # of Apps Installed           700
    Data Usage (MB/day)           700
    Age                           700
    Gender                        700
    dtype: int64





---



## 7. Dropping the missing or null values. - Cristian Zendejas

This is mostly similar to the previous step but in here all the missing values are detected and are dropped later. Now, this is not a good approach to do so, because many people just replace the missing values with the mean or the average of that column, but in this case, I just dropped that missing values. This is because there is nearly 100 missing value compared to 10, 000 values this is a small number and this is negligible so I just dropped those values.


```python
print(df.isnull().sum())
```

    User ID                       0
    Device                        0
    OS                            0
    App Usage Time (min/day)      0
    Screen On Time (hours/day)    0
    Battery Drain (mAh/day)       0
    # of Apps Installed           0
    Data Usage (MB/day)           0
    Age                           0
    Gender                        0
    dtype: int64
    

#### No null values either!


```python
df = df.dropna()    # Dropping the missing values.
df.count()
```




    User ID                       700
    Device                        700
    OS                            700
    App Usage Time (min/day)      700
    Screen On Time (hours/day)    700
    Battery Drain (mAh/day)       700
    # of Apps Installed           700
    Data Usage (MB/day)           700
    Age                           700
    Gender                        700
    dtype: int64



#### We can see that nothing changed.


```python
print(df.isnull().sum())   # After dropping the values
```

    User ID                       0
    Device                        0
    OS                            0
    App Usage Time (min/day)      0
    Screen On Time (hours/day)    0
    Battery Drain (mAh/day)       0
    # of Apps Installed           0
    Data Usage (MB/day)           0
    Age                           0
    Gender                        0
    dtype: int64
    



---



## 8. Detecting Outliers - Cristian Zendejas

An outlier is a point or set of points that are different from other points. Sometimes they can be very high or very low. It's often a good idea to detect and remove the outliers. Because outliers are one of the primary reasons for resulting in a less accurate model. Hence it's a good idea to remove them. The outlier detection and removing that I am going to perform is called IQR score technique. Often outliers can be seen with visualizations using a box plot. The technique of finding and removing outlier that I am performing in this assignment is taken help of a tutorial from[ towards data science](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba).


```python
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['App Usage Time (min/day)'], df['Screen On Time (hours/day)'])
ax.set_xlabel('App Usage Time (min/day)')
ax.set_ylabel('Screen On Time (hours/day)')
plt.show()
# we get some interesting visuals for this one. There seem to be distinct clusters for each range of app usage time.
```


    
![png](output_75_0.png)
    



```python
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['Age'], df['Screen On Time (hours/day)'])
ax.set_xlabel('Age')
ax.set_ylabel('Screen On Time (hours/day)')
plt.show()
# This plot didn't show much distinction between the variables.
```


    
![png](output_76_0.png)
    



```python
sns.boxplot(x=df['Screen On Time (hours/day)'])
# our data was also pretty consistent as we can see that there aren't any outliers!
```




    <Axes: xlabel='Screen On Time (hours/day)'>




    
![png](output_77_1.png)
    



```python
sns.boxplot(x=df['App Usage Time (min/day)'])
```




    <Axes: xlabel='App Usage Time (min/day)'>




    
![png](output_78_1.png)
    



```python
# select only numeric columns
numeric_df = df.select_dtypes(include=['number'])
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
```

    User ID                        349.50
    App Usage Time (min/day)       321.00
    Screen On Time (hours/day)       4.90
    Battery Drain (mAh/day)       1507.25
    # of Apps Installed             48.00
    Data Usage (MB/day)            968.00
    Age                             21.00
    dtype: float64
    

Don't worry about the above values because it's not important to know each and every one of them because it's just important to know how to use this technique in order to remove the outliers.


```python
df = numeric_df[~((numeric_df < (Q1 - 1.5 * IQR)) |(numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape
```




    (700, 7)



Only a few outliers were removed.



---



## 9. Plot different features against one another (scatter), against frequency (histogram) - Cristian Zendejas

### Histogram

Histogram refers to the frequency of occurrence of variables in an interval.


```python
df.Age.value_counts().nlargest(20).sort_index().plot(kind='bar', figsize=(10,5))
plt.title("Age & Screen Time")
plt.ylabel('Screen Time')
plt.xlabel('Age');
# I didn't see any big differences between age and screen time. No matter what age group there seems to be a consistent amount of screen time.
```


    
![png](output_86_0.png)
    


### Heat Maps - Cristian Zendejas

Heat Maps is a type of plot which is necessary when we need to find the dependent variables. One of the best way to find the relationship between the features can be done using heat maps. In the below heat map we know that the price feature depends mainly on the Engine Size, Horsepower, and Cylinders.


```python
plt.plot(df['# of Apps Installed'],df['App Usage Time (min/day)'])
#This plot proved to be very hard to read since the amount of apps are too close in value.
```




    [<matplotlib.lines.Line2D at 0x1795c4e2ff0>]




    
![png](output_88_1.png)
    




---




```python
sns.set(color_codes=True)
```


```python
# resetting the data for this section
df = pd.read_csv("user_behavior_dataset.csv")
```



---



## 4.  Dropping irrelevant columns - Mengyuan Cui


```python
# I recomend to drop the column User Behavior Class-classification of user behavior based on usage patterns (1 to 5).) because we do not
# know the patterns. 
df = df.drop(['User Behavior Class'], axis=1)
df.head(5)
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
    </tr>
  </tbody>
</table>
</div>



## 6. Dropping the duplicate rows - Mengyuan Cui


```python
df.shape
```




    (700, 10)




```python
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
```

    number of duplicate rows:  (0, 10)
    
Now let us remove the duplicate data because it's ok to remove them.

```python
df.count()
```




    User ID                       700
    Device Model                  700
    Operating System              700
    App Usage Time (min/day)      700
    Screen On Time (hours/day)    700
    Battery Drain (mAh/day)       700
    Number of Apps Installed      700
    Data Usage (MB/day)           700
    Age                           700
    Gender                        700
    dtype: int64




```python
df = df.drop_duplicates()
df.head(5)
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
    </tr>
  </tbody>
</table>
</div>




```python
df.count()
```




    User ID                       700
    Device Model                  700
    Operating System              700
    App Usage Time (min/day)      700
    Screen On Time (hours/day)    700
    Battery Drain (mAh/day)       700
    Number of Apps Installed      700
    Data Usage (MB/day)           700
    Age                           700
    Gender                        700
    dtype: int64



## 8. Detecting Outliers - Mengyuan Cui


```python
sns.boxplot(x=df['App Usage Time (min/day)'])
```




    <Axes: xlabel='App Usage Time (min/day)'>




    
![png](output_103_1.png)
    



```python
sns.boxplot(x=df['Screen On Time (hours/day)'])
```




    <Axes: xlabel='Screen On Time (hours/day)'>




    
![png](output_104_1.png)
    



```python
sns.boxplot(x=df['Battery Drain (mAh/day)'])
```




    <Axes: xlabel='Battery Drain (mAh/day)'>




    
![png](output_105_1.png)
    



```python

sns.boxplot(x=df['Number of Apps Installed'])
```




    <Axes: xlabel='Number of Apps Installed'>




    
![png](output_106_1.png)
    



```python
sns.boxplot(x=df['Data Usage (MB/day)'])
```




    <Axes: xlabel='Data Usage (MB/day)'>




    
![png](output_107_1.png)
    

Bar chart

```python
print(df.columns)
```

    Index(['User ID', 'Device Model', 'Operating System',
           'App Usage Time (min/day)', 'Screen On Time (hours/day)',
           'Battery Drain (mAh/day)', 'Number of Apps Installed',
           'Data Usage (MB/day)', 'Age', 'Gender'],
          dtype='object')
    


```python
df['Device Model'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of Mobile Device Usage?")
plt.ylabel('Number of Mobile Device')
plt.xlabel('User ID')
```




    Text(0.5, 0, 'User ID')




    
![png](output_110_1.png)
    



```python
df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 100], labels=['0-18', '19-30', '31-50', '50+'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Age Group', y='App Usage Time (min/day)', data=df)
plt.xlabel('Age Group')
plt.ylabel('App Usage Time (min/day)')
plt.title('Distribution of App Usage Time (min/day) by Age Group')
plt.show()
```


    
![png](output_111_0.png)
    




---



## 1. Data Overview - Matthew Henning


```python
# resetting the data
data = pd.read_csv('user_behavior_dataset.csv')
```


```python
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



## 2. Descriptive Statistics - Matthew Henning


```python
# Check for missing values and data types
data.info()
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
    


```python
# Generate descriptive statistics for numerical columns
data.describe()
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



## 3. Data Visualization - Matthew Henning


```python
# Distribution of App Usage Time
plt.figure(figsize=(10, 6))
sns.histplot(data['App Usage Time (min/day)'], kde=True, bins=30)
plt.title('Distribution of App Usage Time (min/day)')
plt.xlabel('App Usage Time (min/day)')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_120_0.png)
    



```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Screen On Time (hours/day)', y='Battery Drain (mAh/day)', hue='Operating System', data=data)
plt.title('Screen On Time vs Battery Drain')
plt.xlabel('Screen On Time (hours/day)')
plt.ylabel('Battery Drain (mAh/day)')
plt.legend(title='Operating System')
plt.show()
```


    
![png](output_121_0.png)
    



```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='User Behavior Class', y='Age', data=data)
plt.title('Age Distribution by User Behavior Class')
plt.xlabel('User Behavior Class')
plt.ylabel('Age')
plt.show()
```


    
![png](output_122_0.png)
    



```python
# Data Usage by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Data Usage (MB/day)', data=data)
plt.title('Data Usage by Gender')
plt.xlabel('Gender')
plt.ylabel('Data Usage (MB/day)')
plt.show()
```


    
![png](output_123_0.png)
    


## 4. Insights and Conclusion - Matthew Henning
In this exploratory data analysis, we observed various trends and patterns within the dataset, including:
- The distribution of app usage time among users,
- The relationship between screen on time and battery drain across operating systems,
- Age distributions for different user behavior classes,
- Data usage patterns based on gender.

These insights provide a foundational understanding of user behavior, which could be beneficial for further analysis 
or machine learning model development.


---



## Introduction - David Braun
**What is Exploratory Data Analysis ?**

Exploratory Data Analysis or (EDA) is understanding the data sets by summarizing their main characteristics often plotting them visually. This step is very important especially when we arrive at modeling the data in order to apply Machine learning. Plotting in EDA consists of Histograms, Box plot, Scatter plot and many more. It often takes much time to explore the data. Through the process of EDA, we can ask to define the problem statement or definition on our data set which is very important.**How to perform Exploratory Data Analysis ?**

This is one such question that everyone is keen on knowing the answer. Well, the answer is it depends on the data set that you are working. There is no one method or common methods in order to perform EDA, whereas in this tutorial you can understand some common methods and plots that would be used in the EDA process.**What data are we exploring today ?**

Since I am a huge fan of technology, I got a very beautiful data-set of cellphones from Kaggle. The data-set can be downloaded from [here](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset). To give a piece of brief information about the data set this data contains 700 rows and more than 10 columns which contains findings of the cellular usage as Device Model, Operating System, App Usage Time (min/day), and many more. So in this tutorial, we will explore the data and make it ready for modeling.
## 1. Importing the required libraries for EDA

Below are the libraries that are used in order to perform EDA (Exploratory data analysis) in this tutorial.

```python
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
%matplotlib inline     
sns.set(color_codes=True)
```

## 2. Loading the data into the data frame. - David Braun
Loading the data into the pandas data frame is certainly one of the most important steps in EDA, as we can see that the value from the data set is comma-separated. So all we have to do is to just read the CSV into a data frame and pandas data frame does the job for us.

To get or load the dataset into the notebook, all I did was one trivial step. In Google Colab at the left-hand side of the notebook, you will find a > (greater than symbol). When you click that you will find a tab with three options, you just have to select Files. Then you can easily upload your file with the help of the Upload option. No need to mount to the google drive or use any specific libraries just upload the data set and your job is done. One thing to remember in this step is that uploaded files will get deleted when this runtime is recycled. This is how I got the data set into the notebook.

```python
# resetting the data
df = pd.read_csv("./user_behavior_dataset.csv")
```


```python
df.head(5)
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




```python
df.tail(5)                        # To display the botton 5 rows
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
      <th>695</th>
      <td>696</td>
      <td>iPhone 12</td>
      <td>iOS</td>
      <td>92</td>
      <td>3.9</td>
      <td>1082</td>
      <td>26</td>
      <td>381</td>
      <td>22</td>
      <td>Male</td>
      <td>2</td>
    </tr>
    <tr>
      <th>696</th>
      <td>697</td>
      <td>Xiaomi Mi 11</td>
      <td>Android</td>
      <td>316</td>
      <td>6.8</td>
      <td>1965</td>
      <td>68</td>
      <td>1201</td>
      <td>59</td>
      <td>Male</td>
      <td>4</td>
    </tr>
    <tr>
      <th>697</th>
      <td>698</td>
      <td>Google Pixel 5</td>
      <td>Android</td>
      <td>99</td>
      <td>3.1</td>
      <td>942</td>
      <td>22</td>
      <td>457</td>
      <td>50</td>
      <td>Female</td>
      <td>2</td>
    </tr>
    <tr>
      <th>698</th>
      <td>699</td>
      <td>Samsung Galaxy S21</td>
      <td>Android</td>
      <td>62</td>
      <td>1.7</td>
      <td>431</td>
      <td>13</td>
      <td>224</td>
      <td>44</td>
      <td>Male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>699</th>
      <td>700</td>
      <td>OnePlus 9</td>
      <td>Android</td>
      <td>212</td>
      <td>5.4</td>
      <td>1306</td>
      <td>49</td>
      <td>828</td>
      <td>23</td>
      <td>Female</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Checking the types of data - David Braun
Here we check for the datatypes because sometimes the integer data could be stored as a string, if in that case, we have to convert that string to the integer data only then we can plot the data via a graph. Here, in this case, the data is already in integer format so nothing to worry.

```python
df.dtypes
```




    User ID                         int64
    Device Model                   object
    Operating System               object
    App Usage Time (min/day)        int64
    Screen On Time (hours/day)    float64
    Battery Drain (mAh/day)         int64
    Number of Apps Installed        int64
    Data Usage (MB/day)             int64
    Age                             int64
    Gender                         object
    User Behavior Class             int64
    dtype: object



## 4. Dropping irrelevant columns - David Braun
This step is certainly needed in every EDA because sometimes there would be many columns that we never use in such cases dropping is the only solution. In this case, the column User ID doesn't make any sense to me so I just dropped for this instance.

```python
df = df.drop(['User ID'], axis=1)
df.head(5)
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



## 5. Renaming the columns - David Braun
In this instance, most of the column names are very confusing to read, so I just tweaked their column names. This is a good approach it improves the readability of the data set.

```python
df = df.rename(columns={"Device Model": "Model", "Operating System": "OS", "App Usage Time (min/day)": "Apps (min/day)", "Screen On Time (hours/day)": "Screen Time (hours/day)","Battery Drain (mAh/day)": "Battery Drain (mAh/day)", "Number of Apps Installed": "# of Apps", "Data Usage (MB/day)": "Data Usage (MB/day)", "Age":"Age", "Gender":"Sex", "User Behavior Class":"User Level" })
df.head(5)
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
      <th>Model</th>
      <th>OS</th>
      <th>Apps (min/day)</th>
      <th>Screen Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th># of Apps</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>Sex</th>
      <th>User Level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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



## 6. Dropping the duplicate rows - David Braun
This is often a handy thing to do because a larger data set as in this case contains more than 700 rows may have some duplicate data which might be distruptive, so here I remove all the duplicate value from the data-set. For example prior to removing I had 700 rows of data but after removing the duplicates 700 data meaning that I had 0 of duplicate data.

```python
df.shape
```




    (700, 10)




```python
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
```

    number of duplicate rows:  (0, 10)
    
Now let us remove the duplicate data because it's ok to remove them.

```python

df.count()      # Used to count the number of rows
```




    Model                      700
    OS                         700
    Apps (min/day)             700
    Screen Time (hours/day)    700
    Battery Drain (mAh/day)    700
    # of Apps                  700
    Data Usage (MB/day)        700
    Age                        700
    Sex                        700
    User Level                 700
    dtype: int64


So seen above there are 11914 rows and we are removing 989 rows of duplicate data.

```python
df = df.drop_duplicates()
df.head(5)
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
      <th>Model</th>
      <th>OS</th>
      <th>Apps (min/day)</th>
      <th>Screen Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th># of Apps</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>Sex</th>
      <th>User Level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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




```python
df.count()
```




    Model                      700
    OS                         700
    Apps (min/day)             700
    Screen Time (hours/day)    700
    Battery Drain (mAh/day)    700
    # of Apps                  700
    Data Usage (MB/day)        700
    Age                        700
    Sex                        700
    User Level                 700
    dtype: int64



## 7. Dropping the missing or null values. - David Braun
This is mostly similar to the previous step but in here all the missing values are detected and are dropped later. Now, this is not a good approach to do so, because many people just replace the missing values with the mean or the average of that column, but in this case, I just dropped that missing values. This is because there are 0 missing values compared to 700 values this is a small number and this is non-existent, but I would have just dropped those values.

```python
print(df.isnull().sum())
```

    Model                      0
    OS                         0
    Apps (min/day)             0
    Screen Time (hours/day)    0
    Battery Drain (mAh/day)    0
    # of Apps                  0
    Data Usage (MB/day)        0
    Age                        0
    Sex                        0
    User Level                 0
    dtype: int64
    


```python
df = df.dropna()    # Dropping the missing values.
df.count()
```




    Model                      700
    OS                         700
    Apps (min/day)             700
    Screen Time (hours/day)    700
    Battery Drain (mAh/day)    700
    # of Apps                  700
    Data Usage (MB/day)        700
    Age                        700
    Sex                        700
    User Level                 700
    dtype: int64




```python
print(df.isnull().sum())   # After dropping the values
```

    Model                      0
    OS                         0
    Apps (min/day)             0
    Screen Time (hours/day)    0
    Battery Drain (mAh/day)    0
    # of Apps                  0
    Data Usage (MB/day)        0
    Age                        0
    Sex                        0
    User Level                 0
    dtype: int64
    

## 8. Detecting Outliers - David Braun
An outlier is a point or set of points that are different from other points. Sometimes they can be very high or very low. It's often a good idea to detect and remove the outliers. Because outliers are one of the primary reasons for resulting in a less accurate model. Hence it's a good idea to remove them. The outlier detection and removing that I am going to perform is called IQR score technique. Often outliers can be seen with visualizations using a box plot.

```python
sns.boxplot(x=df['Apps (min/day)'])
```




    <Axes: xlabel='Apps (min/day)'>




    
![png](output_163_1.png)
    



```python

sns.boxplot(x=df['# of Apps'])
```




    <Axes: xlabel='# of Apps'>




    
![png](output_164_1.png)
    



```python
sns.boxplot(x=df['Data Usage (MB/day)'])
```




    <Axes: xlabel='Data Usage (MB/day)'>




    
![png](output_165_1.png)
    



```python
sns.boxplot(x=df['Age'])
```




    <Axes: xlabel='Age'>




    
![png](output_166_1.png)
    



```python
# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])
```


```python
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
```

    Apps (min/day)              321.00
    Screen Time (hours/day)       4.90
    Battery Drain (mAh/day)    1507.25
    # of Apps                    48.00
    Data Usage (MB/day)         968.00
    Age                          21.00
    User Level                    2.00
    dtype: float64
    
Don't worry about the above values because it's not important to know each and every one of them because it's just important to know how to use this technique in order to remove the outliers.

```python
ndf = numeric_df[~((numeric_df < (Q1 - 1.5 * IQR)) |(numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
ndf.shape
```




    (700, 7)



## 9. Plot different features against one another (scatter), against frequency (histogram) - David Braun

### Histogram
Histogram refers to the frequency of occurrence of variables in an interval. In this case, there are mainly 10 different types of cell phone companies, but it is often important to know who has the most number of phones. To do this histogram is one of the trivial solutions which lets us know the total number of cell phones manufactured by a different company.

```python
df.Model.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of phones by model")
plt.ylabel('Number of cell phones')
plt.xlabel('Model');
```

### Heat Maps - David Braun
Heat Maps is a type of plot which is necessary when we need to find the dependent variables. One of the best way to find the relationship between the features can be done using heat maps. In the below heat map we know that the Age feature depends mainly on the ... .

```python
plt.figure(figsize=(10,5))
c= ndf.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c
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
      <th>Apps (min/day)</th>
      <th>Screen Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th># of Apps</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>User Level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Apps (min/day)</th>
      <td>1.000000</td>
      <td>0.950333</td>
      <td>0.956385</td>
      <td>0.955253</td>
      <td>0.942308</td>
      <td>0.004382</td>
      <td>0.970498</td>
    </tr>
    <tr>
      <th>Screen Time (hours/day)</th>
      <td>0.950333</td>
      <td>1.000000</td>
      <td>0.948983</td>
      <td>0.946975</td>
      <td>0.941322</td>
      <td>0.017232</td>
      <td>0.964581</td>
    </tr>
    <tr>
      <th>Battery Drain (mAh/day)</th>
      <td>0.956385</td>
      <td>0.948983</td>
      <td>1.000000</td>
      <td>0.961853</td>
      <td>0.932276</td>
      <td>-0.002722</td>
      <td>0.978587</td>
    </tr>
    <tr>
      <th># of Apps</th>
      <td>0.955253</td>
      <td>0.946975</td>
      <td>0.961853</td>
      <td>1.000000</td>
      <td>0.934800</td>
      <td>0.004034</td>
      <td>0.981255</td>
    </tr>
    <tr>
      <th>Data Usage (MB/day)</th>
      <td>0.942308</td>
      <td>0.941322</td>
      <td>0.932276</td>
      <td>0.934800</td>
      <td>1.000000</td>
      <td>0.003999</td>
      <td>0.946734</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.004382</td>
      <td>0.017232</td>
      <td>-0.002722</td>
      <td>0.004034</td>
      <td>0.003999</td>
      <td>1.000000</td>
      <td>-0.000563</td>
    </tr>
    <tr>
      <th>User Level</th>
      <td>0.970498</td>
      <td>0.964581</td>
      <td>0.978587</td>
      <td>0.981255</td>
      <td>0.946734</td>
      <td>-0.000563</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_177_1.png)
    


### Scatterplot - David Braun
We generally use scatter plots to find the correlation between two variables. Here the scatter plots are plotted between Screen Time (hours/day) and Apps (min/day) and we can see the plot below. With the plot given below, we can easily draw a trend line. These features provide a good scattering of points.

```python
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['Screen Time (hours/day)'], df['Apps (min/day)'])
ax.set_xlabel('Screen Time (hours/day)')
ax.set_ylabel('Apps (min/day)')
plt.show()
```

# Q: Do men or women install more apps on their phones - David Braun

### Description:
This bar chart compares the average number of apps installed by men and women. It helps us see who tends to use their phones for more apps.


```python
# Average number of apps installed by gender
gender_apps = df.groupby('Sex')['# of Apps'].mean()
gender_apps.plot(kind='bar', color=['blue', 'pink'], figsize=(8, 6))
plt.title('Average Number of Apps Installed by Gender')
plt.ylabel('Number of Apps Installed')
plt.show()
```


    
![png](output_183_0.png)
    


# Q: Who spends more time on their phone screens: men or women?

### Description: 
This chart shows the average number of hours men and women keep their screens on each day. It gives an idea of who might be more "screen-heavy."


```python
# Average screen time by gender
gender_screen_time = df.groupby('Sex')['Screen Time (hours/day)'].mean()
gender_screen_time.plot(kind='bar', color=['blue', 'pink'], figsize=(8, 6))
plt.title('Average Screen Time by Gender')
plt.ylabel('Screen Time (hours/day)')
plt.show()
```


    
![png](output_186_0.png)
    


# Q: Who spends more time using apps: men or women?

### Description:
This visualization looks at the time spent actively using apps each day, comparing men and women. It helps us see who might rely more on apps for activities.


```python
# Average app usage time by gender
gender_app_usage = df.groupby('Sex')['Apps (min/day)'].mean()
gender_app_usage.plot(kind='bar', color=['blue', 'pink'], figsize=(8, 6))
plt.title('Average App Usage Time by Gender')
plt.ylabel('App Usage Time (min/day)')
plt.show()
```


    
![png](output_189_0.png)
    


# Q: Does age affect how much battery men and women use?

### Description:
This scatter plot shows how battery drain changes as men and women get older. It helps us understand if younger or older users consume more battery.


```python
plt.figure(figsize=(8, 6))
plt.scatter(df[df['Sex'] == 'Male']['Age'], 
            df[df['Sex'] == 'Male']['Battery Drain (mAh/day)'], 
            alpha=0.7, label='Men', color='blue')

plt.scatter(df[df['Sex'] == 'Female']['Age'], 
            df[df['Sex'] == 'Female']['Battery Drain (mAh/day)'], 
            alpha=0.7, label='Women', color='pink')

plt.title('Age vs. Battery Drain by Gender')
plt.xlabel('Age')
plt.ylabel('Battery Drain (mAh/day)')
plt.legend()
plt.show()
```


    
![png](output_192_0.png)
    



```python
# Average usage metrics by gender
average_usage = df.groupby('Sex')[['Apps (min/day)', 'Screen Time (hours/day)', 'Battery Drain (mAh/day)']].mean()
print(average_usage)
```

# Q: Who uses their phone excessively based on specific thresholds?

### Description:
This bar chart identifies men and women who use their phones for more than 5 hours of screen time, 300 minutes of app usage, and 1500 mAh of battery drain per day. It highlights which gender has more "heavy users."


```python
# Define excessive usage thresholds
excessive_users = df[(df['Apps (min/day)'] > 300) & 
                       (df['Screen Time (hours/day)'] > 5) & 
                       (df['Battery Drain (mAh/day)'] > 1500)]

excessive_count = excessive_users['Sex'].value_counts()
excessive_count.plot(kind='bar', color=['blue', 'pink'], figsize=(8, 6))
plt.title('Excessive Phone Users by Gender')
plt.ylabel('Count')
plt.show()
```


    
![png](output_196_0.png)
    


# Q: Which operating system (Android or iOS) is more popular among men and women?

### Description:
This stacked bar chart shows how many men and women use Android or iOS. It highlights the preferences of each gender.


```python
# OS prevalence by gender
os_gender = df.groupby(['OS', 'Sex']).size().unstack()
os_gender.plot(kind='bar', stacked=True, figsize=(10, 6), color=['blue', 'pink'])
plt.title('Operating System Prevalence by Gender')
plt.ylabel('Count')
plt.show()
```


    
![png](output_199_0.png)
    


# Q: Does the type of operating system affect phone usage time?

### Description:
This bar chart compares the average screen time and app usage time (converted to hours) for Android and iOS users. It helps us see which OS might encourage less phone time.


```python
# Convert App Usage Time from minutes to hours
df['App Usage Time (hours/day)'] = df['Apps (min/day)'] / 60

# Group by Operating System and calculate mean values for Screen Time and App Usage Time
os_efficiency = df.groupby('OS')[['Screen Time (hours/day)', 'App Usage Time (hours/day)']].mean()

# Plot the comparison
os_efficiency.plot(kind='bar', figsize=(10, 6))
plt.title('Operating System Efficiency: Screen Time vs App Usage Time (in Hours)')
plt.ylabel('Average Usage Time (hours/day)')
plt.legend(['Screen Time (hours/day)', 'App Usage Time (hours/day)'])
plt.xticks(rotation=0)
plt.show()
```


    
![png](output_202_0.png)
    


**Hence the above are some of the steps involved in Exploratory data analysis, these are some general steps that you must follow in order to perform EDA. There are many more yet to come but for now, this is more than enough idea as to how to perform a good EDA given any data sets. Stay tuned for more updates.**

## Thank you.
