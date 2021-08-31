# %%
# the full imports
import pandas as pd 
import numpy as np
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt

# %%
# the from imports
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

# %%
# imports data
dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")   
# allows large amounts of json code to be run
alt.data_transformers.enable('json')

# %%
# 1 - Shows corelation of year built to other categories
h_subset = dwellings_ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980']).sample(500)
# creates chart of all data points
sns.pairplot(h_subset, hue = 'before1980')
corr = h_subset.drop(columns = 'before1980').corr()

# %%
# 1 - saves correlation heatmap
sns.heatmap(corr).figure.savefig('1980_correlation.jpg')

# %%
# 1 - graph 1 for numbaths to compared to year built
(alt.Chart(dwellings_ml, title = 'Number of Baths Over the Years')
    .encode(
        x = alt.X('yrbuilt', scale = alt.Scale(zero = False)),
        y = alt.Y('numbaths', scale = alt.Scale(zero = False)),
        color = alt.Color('before1980:O' ,scale = alt.Scale(scheme='blueorange')))
        .mark_circle()).save('1980_numbaths_correlation.png')

# %%
# 1 - graph for number of floors compared to year built
(alt.Chart(dwellings_ml, title = 'Number of Floors Over the Years')
    .encode(
        x = alt.X('yrbuilt', scale = alt.Scale(zero = False)),
        y = alt.Y('stories', scale = alt.Scale(zero = False)),
        color = alt.Color('before1980:O' ,scale = alt.Scale(scheme='purplegreen')))
        .mark_circle()).save('1980_stories_correlation.png')

# %%
# 2 + 3 - creates new dataframe with revelant data
colorado_data = dwellings_denver.filter(['nbhd','abstrprd','livearea', 
'finbsmnt', 'basement', 'totunits','stories','sprice', 'deduct', 
'netprice', 'tasp', 'smonth', 'syear', 'yrbuilt',
'before1980'])

# %%
# 2 + 3 - creates new column in colorado dataframe for before1980
colorado_data['before1980'] = np.where(colorado_data['yrbuilt'] < 1980, 1, 0)

# %%
# 2 + 3 - creates a corelation chart that shows relationship to year built for colorado
# gets a sample size from data
colorado_h_subset = colorado_data.sample(500)
# creates correlation chart of all data points
sns.pairplot(colorado_h_subset, hue = 'before1980')
colorado_corr = colorado_h_subset.drop(columns = 'before1980').corr()

# %%
# 2 + 3 - creates and saves heatmap for colorado correlation
sns.heatmap(colorado_corr).figure.savefig('colorado_1980_correlation.jpg')

# %%
# 2 + 3 - creates chart showing the total units over the years in colorado
(alt.Chart(colorado_data, title = 'Denver Home Total Units over the years')
    .encode(
        x = alt.X('yrbuilt', scale = alt.Scale(zero = False)),
        y = alt.Y('totunits', scale = alt.Scale(zero = False)),
        color = alt.Color('before1980:O' ,scale = alt.Scale(scheme='redblue')))
        .mark_circle()).save('colorado_1980_tstories_correlation.png')

# %%
# 2 + 3 - Sets parameters for model, comparing total units to before1980
X_pred = colorado_data.drop(['yrbuilt', 'before1980'], axis = 1)
y_pred = colorado_data.before1980
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76) 

# %%
clf = GradientBoostingClassifier()
clf = clf.fit(X_train, y_train)
predict_p =  clf.predict(X_test)

(pd.Series(clf.feature_importances_, index=X_pred.columns)
   .nlargest(5)
   .plot(kind='barh')).figure.savefig('feature_importance_chart.jpg')

# %%
# 2 + 3 - creates confusion matrix and prints to markdown
model_confusion_matrix_results = pd.DataFrame(metrics.confusion_matrix(y_test, predict_p))
print(model_confusion_matrix_results.to_markdown())
metrics.plot_confusion_matrix(clf, X_test, y_test)

# %%
model_results = metrics.classification_report(y_test, predict_p)
print(model_results)

# %%
# 2 + 3 - shows the p values of each column
df_features = (pd.DataFrame(
        {'f_names': X_train.columns, 
        'f_values': clf.feature_importances_})
    .sort_values('f_values', ascending = False))
# %%

# %%
