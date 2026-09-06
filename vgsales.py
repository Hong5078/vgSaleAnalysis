#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv('vgsales.csv')


# In[3]:


df.head(5)


# In[4]:


df.shape


# In[5]:


sports_games = df.loc[(df.Genre == 'Sports') & (df.Publisher == 'Electronic Arts')]
sports_games = sports_games.sort_values(by=['Rank'])
sports_games = sports_games.dropna(how="any")
sports_games.Year = sports_games.Year.astype(int)
sports_games


# In[6]:


sports_games_1 = sports_games[0:20]
heights = sports_games_1.Global_Sales
bars = sports_games_1.Platform
y_pos = range(len(sports_games_1.Global_Sales))
plt.bar(y_pos, heights)
# Rotation of the bars names
plt.xticks(y_pos, bars, rotation=90)
plt.show()


# In[7]:


ps4 = sports_games.loc[sports_games.Platform == 'PS4']
ps4_2016 = ps4.loc[ps4.Year == 2016]
ps4_2016


# In[8]:


heights = ps4_2016.Global_Sales
bars = ps4_2016.Rank
y_pos = range(len(ps4_2016.Global_Sales))
plt.bar(y_pos, heights)
# Rotation of the bars names
plt.xticks(y_pos, bars, rotation=90)
plt.show()


# ### Corelation between Ranking and North America Sales with respect to Year for EA_sports

# In[9]:


year_EA_sports = sports_games.groupby(['Year'], as_index=False).sum()
year_EA_sports


# In[10]:


heights = year_EA_sports.Global_Sales
bars = year_EA_sports.Year
y_pos = range(len(year_EA_sports.Global_Sales))
plt.bar(y_pos, heights)
# Rotation of the bars names
plt.xticks(y_pos, bars, rotation=90)
plt.show()


# In[11]:


plt.plot(year_EA_sports.Year, year_EA_sports.Global_Sales)
plt.show()


# In[12]:


plt.plot(year_EA_sports.Year, year_EA_sports.Rank / 3 ** 9)
plt.plot(year_EA_sports.Year, year_EA_sports.NA_Sales)
plt.xlabel("Year")
plt.ylabel("Rank")
plt.show()


# In[13]:


company = df.loc[df.Publisher == '1C Company']
company


# ### Total sales by every publisher from 1995 to 2016

# In[14]:


total_sales_1 = df.sort_values("Publisher")
total_sales_1


# In[15]:


total_sales = total_sales_1.iloc[:,5:]
total_sales.dropna(how='any', inplace = True)
total_sales


# In[16]:


total_sales_new = total_sales.groupby('Publisher', as_index=False).sum()
total_sales_new


# In[17]:


t3 = total_sales_new.sort_values("Global_Sales", ascending = False)
t3


# In[18]:


t3_1 = t3.iloc[:10, :]
t3_1


# In[19]:


heights = t3_1.Global_Sales
bars = t3_1.Publisher
y_pos = range(len(t3_1.Global_Sales))
plt.bar(y_pos, heights)
# Rotation of the bars names
plt.xticks(y_pos, bars, rotation=90, size=12)
plt.xlabel("Publisher")
plt.ylabel("Global sales")
plt.title("Top 10 Publishers in Global Sales from 1995 to 2016")
plt.show()


# ### Nintendo total Sales as per Genre 2015 

# In[20]:


nsp = df.loc[df.Publisher == 'Nintendo']
nsp = nsp.sort_values('Year')
nsp.dropna(how="any", inplace=True)
nsp


# In[21]:


def year_nin(y):
    return nsp.loc[nsp.Year == y]
y_2015 = year_nin(2015)
y_2015


# In[22]:


y_2015_g = y_2015.sort_values("Genre")
y_2015_g = y_2015_g.drop(['Year', 'Rank'], axis=1)
y_2015_g


# In[23]:


y15g_s = y_2015_g.groupby("Genre", as_index=False).sum()
y15g_s


# In[24]:


heights = y15g_s.Global_Sales
bars = y15g_s.Genre
y_pos = range(len(y15g_s.Global_Sales))
plt.bar(y_pos, heights)
# Rotation of the bars names
plt.xticks(y_pos, bars, rotation=90, size=12)
plt.xlabel("Genre")
plt.ylabel("Global sales")
plt.title("Total Nintendo sales by Genre in 2015")
plt.show()


# ### 1. Define - decision tree or random forest or new model
# ### 2. Fit - Capture patterns from provied data (train)
# ### 3. Prediction - model - test -result
# ### 4. Evaluate - accuracy

# In[25]:


pdf = df.loc[df.Publisher == "Nintendo"]
pdf


# In[26]:


pdf = pdf.sort_values("Year")
pdf


# In[27]:


pdf.dropna(how="any", inplace=True)
pdf


# In[28]:


pdf = pdf.groupby("Year", as_index=False).sum()
pdf


# In[29]:


plt.plot(pdf.Year,pdf.Global_Sales)
plt.show()


# ### Decision Tree Regression

# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
y = pdf.Global_Sales
pdf_features = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales']
X = pdf[pdf_features]
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=0)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    pdf_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)
    pdf_model.fit(train_X,train_y)
    pdf_preds = pdf_model.predict(val_X)
    mae = mean_absolute_error(val_y, pdf_preds)
    return(mae)
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean absolute error: %d" %(max_leaf_nodes,my_mae))


# In[31]:


pdf_model = DecisionTreeRegressor(max_leaf_nodes = 50, random_state=0)
pdf_model.fit(train_X,train_y)
pdf_preds = pdf_model.predict(val_X)
maed = mean_absolute_error(val_y, pdf_preds)
pdf_preds


# In[32]:


val_y1 = pd.DataFrame(val_y) 
val_y1


# In[33]:


val_X.index = range(9)
val_X


# In[34]:


pdf_preds_df = pd.DataFrame(data=pdf_preds, columns=['preds'])
pdf_preds_df


# In[35]:


val_X1 = pd.DataFrame(val_X, columns=['Year'])
val_X1.sort_values('Year', inplace=True)
val_X1


# ### Problem arised is that we gave a train_test_split algorithm to split our data into train and test. It chose randomly without considering timeseries. But still we got good predictions as seen in the graph below.

# In[36]:


plt.plot(val_X1.Year, val_y1)
plt.plot(val_X1.Year, pdf_preds_df.preds)
plt.xlabel('Year')
plt.ylabel('Global Sales')
plt.legend(['Original', 'Prediction'])
plt.show()


# In[37]:


# accuracy score
# import export_graphviz 
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(pdf_model, out_file ='tree.dot', feature_names =['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales'])


# ### Linear Regression

# In[38]:


from sklearn.linear_model import LinearRegression
l_model = LinearRegression()
l_model.fit(train_X,train_y)
preds_l = l_model.predict(val_X)
preds_l


# In[39]:


preds_l1 = pd.DataFrame(data=preds_l, columns=['preds'])
preds_l1


# In[40]:


val_y


# In[41]:


scorel = l_model.score(train_X,train_y)
mael = mean_absolute_error(val_y, preds_l)
print('Mean absolute error: \t', mael)
print('Linear Regression score: \t', scorel)


# ### Linear Regression is perfect for this data at 99% accuracy

# In[42]:


plt.plot(val_X1.Year, val_y1)
plt.plot(val_X1.Year, preds_l1.preds)
plt.xlabel('Year')
plt.ylabel('Global Sales')
plt.legend(['Original', 'Prediction'])
plt.show()


# ### Comparing Decision Tree Regression and Linear Regression

# In[43]:


plt.rcParams['lines.linewidth'] = 2
plt.rcParams["figure.figsize"] = [8, 6]
plt.plot(val_X1.Year, val_y1)
plt.plot(val_X1.Year, pdf_preds_df.preds)
plt.plot(val_X1.Year, preds_l1.preds)
plt.xlabel('Year')
plt.ylabel('Global Sales')
plt.legend(['Original', 'Decision Tree', 'Linear Regression'])
plt.show()


# ### Randaom Forest Regression

# In[44]:


from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=200, random_state=1)
forest_model.fit(train_X, train_y)
predsf = forest_model.predict(val_X)
predsf


# In[45]:


mae_forest = mean_absolute_error(val_y, predsf)
mae_forest


# In[46]:


preds_f = pd.DataFrame(data=predsf, columns=['preds'])
preds_f


# In[47]:


plt.plot(val_X1.Year, val_y1)
plt.plot(val_X1.Year, preds_f.preds)
plt.plot(val_X1.Year, pdf_preds_df.preds)
plt.xlabel('Year')
plt.ylabel('Global Sales')
plt.legend(['Original', 'Random forest', 'Decision tree'])
plt.show()


# In[48]:


print("MAE of Linear Regrression: %d \n MAE of Decision Tree Regressor: %d \n MAE of Random Forest Regressor: %d" %(mael, maed, mae_forest))


# In[ ]:




