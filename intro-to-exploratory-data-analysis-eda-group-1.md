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



Since I am a huge fan of cars, I got a very beautiful data-set of cars from Kaggle. The data-set can be downloaded from [here](https://www.kaggle.com/CooperUnion/cardataset). To give a piece of brief information about the data set this data contains more of 10, 000 rows and more than 10 columns which contains features of the car such as Engine Fuel Type, Engine HP, Transmission Type, highway MPG, city MPG and many more. So in this tutorial, we will explore the data and make it ready for modeling.



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



## 3. Checking the types of data

Here we check for the datatypes because sometimes the MSRP or the price of the car would be stored as a string, if in that case, we have to convert that string to the integer data only then we can plot the data via a graph. Here, in this case, the data is already in integer format so nothing to worry.


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



## 4. Dropping irrelevant columns

This step is certainly needed in every EDA because sometimes there would be many columns that we never use in such cases dropping is the only solution. In this case, the columns such as Engine Fuel Type, Market Category, Vehicle style, Popularity, Number of doors, Vehicle Size doesn't make any sense to me so I just dropped for this instance.


```python
df = df.drop(['Data Usage (MB/day)', 'User Behavior Class', 'Gender'], axis=1)
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
      <th>Age</th>
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
      <td>40</td>
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
      <td>47</td>
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
      <td>42</td>
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
      <td>20</td>
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
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>





---



## 5. Renaming the columns

In this instance, most of the column names are very confusing to read, so I just tweaked their column names. This is a good approach it improves the readability of the data set.


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
      <th>Age</th>
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
      <td>40</td>
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
      <td>47</td>
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
      <td>42</td>
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
      <td>20</td>
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
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>





---



## 6. Dropping the duplicate rows

This is often a handy thing to do because a huge data set as in this case contains more than 10, 000 rows often have some duplicate data which might be disturbing, so here I remove all the duplicate value from the data-set. For example prior to removing I had 11914 rows of data but after removing the duplicates 10925 data meaning that I had 989 of duplicate data.


```python
df.shape
```




    (700, 8)




```python
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
```

    number of duplicate rows:  (0, 8)
    

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
    Age                           700
    dtype: int64



# no duplicates for this data!


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
      <th>Age</th>
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
      <td>40</td>
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
      <td>47</td>
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
      <td>42</td>
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
      <td>20</td>
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
      <td>31</td>
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
    Age                           700
    dtype: int64





---



## 7. Dropping the missing or null values.

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
    Age                           0
    dtype: int64
    

# No null values either!


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
    Age                           700
    dtype: int64



# We can see that nothing changed.


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
    Age                           0
    dtype: int64
    



---



## 8. Detecting Outliers

An outlier is a point or set of points that are different from other points. Sometimes they can be very high or very low. It's often a good idea to detect and remove the outliers. Because outliers are one of the primary reasons for resulting in a less accurate model. Hence it's a good idea to remove them. The outlier detection and removing that I am going to perform is called IQR score technique. Often outliers can be seen with visualizations using a box plot. Shown below are the box plot of MSRP, Cylinders, Horsepower and EngineSize. Herein all the plots, you can find some points are outside the box they are none other than outliers. The technique of finding and removing outlier that I am performing in this assignment is taken help of a tutorial from[ towards data science](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba).


```python
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['App Usage Time (min/day)'], df['Screen On Time (hours/day)'])
ax.set_xlabel('App Usage Time (min/day)')
ax.set_ylabel('Screen On Time (hours/day)')
plt.show()
```


    
![png](output_51_0.png)
    



```python
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['Age'], df['Screen On Time (hours/day)'])
ax.set_xlabel('Age')
ax.set_ylabel('Screen On Time (hours/day)')
plt.show()
```


    
![png](output_52_0.png)
    



```python
sns.boxplot(x=df['Screen On Time (hours/day)'])
```




    <Axes: xlabel='Screen On Time (hours/day)'>




    
![png](output_53_1.png)
    



```python
sns.boxplot(x=df['App Usage Time (min/day)'])
```




    <Axes: xlabel='App Usage Time (min/day)'>




    
![png](output_54_1.png)
    



```python
# what if we try using the z-score?
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
    Age                             21.00
    dtype: float64
    

Don't worry about the above values because it's not important to know each and every one of them because it's just important to know how to use this technique in order to remove the outliers.


```python
df = numeric_df[~((numeric_df < (Q1 - 1.5 * IQR)) |(numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape
```




    (700, 6)



Only a few outliers were removed.



---



## 9. Plot different features against one another (scatter), against frequency (histogram)

### Histogram

Histogram refers to the frequency of occurrence of variables in an interval. In this case, there are mainly 10 different types of car manufacturing companies, but it is often important to know who has the most number of cars. To do this histogram is one of the trivial solutions which lets us know the total number of car manufactured by a different company.


```python
df.Age.value_counts().nlargest(20).sort_index().plot(kind='bar', figsize=(10,5))
plt.title("Age & Screen Time")
plt.ylabel('Screen Time')
plt.xlabel('Age');
```


    
![png](output_62_0.png)
    


### Heat Maps

Heat Maps is a type of plot which is necessary when we need to find the dependent variables. One of the best way to find the relationship between the features can be done using heat maps. In the below heat map we know that the price feature depends mainly on the Engine Size, Horsepower, and Cylinders.


```python
plt.plot(df['# of Apps Installed'],df['App Usage Time (min/day)'])
```




    [<matplotlib.lines.Line2D at 0x14d9500f680>]




    
![png](output_64_1.png)
    



```python

```

### Scatterplot

We generally use scatter plots to find the correlation between two variables. Here the scatter plots are plotted between Horsepower and Price and we can see the plot below. With the plot given below, we can easily draw a trend line. These features provide a good scattering of points.

**Hence the above are some of the steps involved in Exploratory data analysis, these are some general steps that you must follow in order to perform EDA. There are many more yet to come but for now, this is more than enough idea as to how to perform a good EDA given any data sets. Stay tuned for more updates.**

## Thank you.
