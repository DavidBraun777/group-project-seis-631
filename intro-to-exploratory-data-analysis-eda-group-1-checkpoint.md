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



Since I’m very interested in user behavior analysis, I obtained a fascinating dataset about mobile usage. The data-set can be downloaded from [here](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset). This dataset has 700 rows and 11 columns detailing various aspects of user interactions with their devices. It includes features like Device Model, Operating System, App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day), Number of Apps Installed, Data Usage (MB/day), Age, Gender, and User Behavior Class. In this tutorial, we’ll explore this data and prepare it for modeling. ​



---



## 1. Importing the required libraries for EDA


```python
import pandas as pd
df = pd.read_csv('user_behavior_dataset.csv')
df.tail(5)                       
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




```python
df.tail(5)
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



## 2. Checking the types of data


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





---



## 3. Dropping the duplicate rows, missing or null value, and irrelevant columns


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
df.shape
```




    (700, 8)




```python
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
```

    number of duplicate rows:  (0, 8)
    


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
    



---



## 5. Renaming the columns


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





---



## 4. Detecting Outliers


```python
# what other charts can we use to see outliers?
sns.boxplot(x=df['Screen Time'])
```




    <Axes: xlabel='Screen Time'>




    
![png](output_34_1.png)
    



```python
sns.boxplot(x=df['Usage'])
```




    <Axes: xlabel='Usage'>




    
![png](output_35_1.png)
    



```python
sns.boxplot(x=df['Behavior'])
```




    <Axes: xlabel='Behavior'>




    
![png](output_36_1.png)
    




---



## 5. Plot against frequency (histogram)

### Histogram


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


    
![png](output_40_0.png)
    



```python
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
```




    Text(0, 0.5, 'Frequency')




    
![png](output_41_1.png)
    



```python
plt.figure(figsize=(10, 6))
plt.hist(df['Usage'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Data Usage (MB/day)')
plt.xlabel('Usage')
plt.ylabel('Frequency')
```




    Text(0, 0.5, 'Frequency')




    
![png](output_42_1.png)
    


## 6. Plot against one another (scatter)

### Scatterplot


```python
plt.figure(figsize=(10, 6))
plt.scatter(df['Screen Time'], df['Usage'], alpha=0.7)
plt.title('Screen Time vs Usage')
plt.xlabel('Screen Time')
plt.ylabel('Usage')
plt.grid(True)
plt.show()
```


    
![png](output_45_0.png)
    



```python
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Apps Installed'], alpha=0.7)  
plt.title('Apps Installed vs. Age') 
plt.xlabel('Age')
plt.ylabel('Apps Installed') 
plt.grid(True) 
plt.show() 
```


    
![png](output_46_0.png)
    


**Hence the above are some of the steps involved in Exploratory data analysis, these are some general steps that you must follow in order to perform EDA. There are many more yet to come but for now, this is more than enough idea as to how to perform a good EDA given any data sets. Stay tuned for more updates.**

## Thank you.
