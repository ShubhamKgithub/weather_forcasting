In [1]:#Importing needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
%matplotlib inline

In [5]:#Reading CSV file as weather_df and making date_time column as index of dataframe
weather_df = pd.read_csv("https://raw.githubusercontent.com/neetika6/Machine-Learning-Model-for-Weather-Forecasting/main/kanpur.csv" , parse_dates=['date_time'], index_col='date_ti

In [6]:weather_df.head()

In [7]:#Checking columns in our dataframe
weather_df.columns

Out[7]:Index(['maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour', 'uvIndex',
 'uvIndex.1', 'moon_illumination', 'moonrise', 'moonset', 'sunrise',
 'sunset', 'DewPointC', 'FeelsLikeC', 'HeatIndexC', 'WindChillC',
 'WindGustKmph', 'cloudcover', 'humidity', 'precipMM', 'pressure',
 'tempC', 'visibility', 'winddirDegree', 'windspeedKmph'],
 dtype='object')


In [8]:#Now shape
weather_df.shape

In [9]:weather_df.describe()

In [10]:#Checking if there is any null values in dataset
weather_df.isnull().any()

In [10]:#Checking if there is any null values in dataset
weather_df.isnull().any()

Out[10]:maxtempC False
mintempC False
totalSnow_cm False
sunHour False
uvIndex False
uvIndex.1 False
moon_illumination False
moonrise False
moonset False
sunrise False
sunset False
DewPointC False
FeelsLikeC False
HeatIndexC False
WindChillC False
WindGustKmph False
cloudcover False
humidity False
precipMM False
pressure False
tempC False
visibility False
winddirDegree False
windspeedKmph False
dtype: bool

In [11]:#Now lets separate the feature (i.e. temperature) to be predicted from the rest of the featured. weather_x stores the rest of the dataset while weather_y has temperature column.
weather_df_num=weather_df.loc[:,['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour','HeatIndexC', 'precipMM', 'pressure','windspeedKmph']]
weather_df_num.head()

In [12]:#Shape of new dataframe
weather_df_num.shape

Out[12]:(96432, 10)

In [13]:#Columns in new dataframe
weather_df_num.columns

Out[13]:Index(['maxtempC', 'mintempC', 'cloudcover', 'humidity', 'tempC', 'sunHour',
 'HeatIndexC', 'precipMM', 'pressure', 'windspeedKmph'],
 dtype='object')

In [19]:#Ploting all the column values
weather_df_num.plot(subplots=True, figsize=(25,20))

Out[19]:array([<AxesSubplot:xlabel='date_time'>, <AxesSubplot:xlabel='date_time'>,
 <AxesSubplot:xlabel='date_time'>, <AxesSubplot:xlabel='date_time'>,
 <AxesSubplot:xlabel='date_time'>, <AxesSubplot:xlabel='date_time'>,
 <AxesSubplot:xlabel='date_time'>, <AxesSubplot:xlabel='date_time'>,
 <AxesSubplot:xlabel='date_time'>], dtype=object)

In [21]:#Ploting all the column values for 1 year
weather_df_num['2019':'2020'].resample('D').fillna(method='pad').plot(subplots=True, figsize=(25,20))

Out[21]:array([<AxesSubplot:xlabel='date_time'>, <AxesSubplot:xlabel='date_time'>,
 <AxesSubplot:xlabel='date_time'>, <AxesSubplot:xlabel='date_time'>,
 <AxesSubplot:xlabel='date_time'>, <AxesSubplot:xlabel='date_time'>,
 <AxesSubplot:xlabel='date_time'>, <AxesSubplot:xlabel='date_time'>,
 <AxesSubplot:xlabel='date_time'>], dtype=object)

In[16]:weather_df_num.hist(bins=10,figsize=(15,15))

Out[16]:array([[<AxesSubplot:title={'center':'maxtempC'}>,
 <AxesSubplot:title={'center':'mintempC'}>,
 <AxesSubplot:title={'center':'cloudcover'}>],
 [<AxesSubplot:title={'center':'humidity'}>,
 <AxesSubplot:title={'center':'tempC'}>,
 <AxesSubplot:title={'center':'sunHour'}>],
 [<AxesSubplot:title={'center':'HeatIndexC'}>,
 <AxesSubplot:title={'center':'precipMM'}>,
 <AxesSubplot:title={'center':'pressure'}>],
 [<AxesSubplot:title={'center':'windspeedKmph'}>, <AxesSubplot:>,
 <AxesSubplot:>]], dtype=object)

In [17]:weth=weather_df_num['2019':'2020']
weth.head()

In [18]:weather_y=weather_df_num.pop("tempC")
weather_x=weather_df_num

In [ ]:# Now splitting the dataset into training and testing.

In [22]:train_X,test_X,train_y,test_y=train_test_split(weather_x,weather_y,test_size=0.2,random_state=4)

In [23]:train_X.shape

Out [23]:(77145, 9)

In [24]:train_y.shape

Out [24]:(77145,)

In [ ]:# train_x has all the features except temperature and train_y has the corresponding temperature for those features. in supervised machine learning we first feed the model with inpu

In [25]:train_y.head()

Out [25]:date_time
2012-03-13 07:00:00 22
2009-11-05 21:00:00 21
2017-10-11 22:00:00 30
2019-06-08 11:00:00 47
2019-03-06 05:00:00 18
Name: tempC, dtype: int64

In [30]:model=LinearRegression()
model.fit(train_X,train_y)

Out[30]:LinearRegression()

In [31]:prediction = model.predict(test_X)

In [32]:#calculating error
np.mean(np.absolute(prediction-test_y))

Out[32]:1.200473579409673

In [33]:print('Variance score: %.2f' % model.score(test_X, test_y))

Variance score: 0.96

In [34]:for i in range(len(prediction)):
 prediction[i]=round(prediction[i],2)
pd.DataFrame({'Actual':test_y,'Prediction':prediction,'diff':(test_y-prediction)})


In [35]:from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(train_X,train_y)

Out[35]:DecisionTreeRegressor(random_state=0)

In [36]:prediction2=regressor.predict(test_X)
np.mean(np.absolute(prediction2-test_y))


Out [36]:0.563013083078412

In [41]:print('Variance score: %.2f' % regr.score(test_X, test_y))

Variance score: 0.99

In [43]:from sklearn.metrics import r2_score

In [44]:print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,prediction ) )

Mean absolute error: 1.20
Residual sum of squares (MSE): 2.51
R2-score: 0.96

In [45]:print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction2 - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction2 - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,prediction2 ) )

Mean absolute error: 0.56
Residual sum of squares (MSE): 1.12
R2-score: 0.98

In [47]:print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction3 - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction3 - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,prediction3 ) )

Mean absolute error: 0.47
Residual sum of squares (MSE): 0.63
R2-score: 0.99