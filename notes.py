# %%
# import sys
# !{sys.executable} -m pip install seaborn scikit-learn

# %%
# the full imports
import pandas as pd 
import numpy as np
import seaborn as sns
import altair as alt

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
# filters the dataset to just these columns with a sample of just 500 data points
h_subset = dwellings_ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980']).sample(500)
# creates chart of all data points
sns.pairplot(h_subset, hue = 'before1980')
corr = h_subset.drop(columns = 'before1980').corr()

# %%
# shows how corelated each variable is related to each other variable
sns.heatmap(corr)

# %%
# graphs yrbuilt to numbaths before 1980
(alt.Chart(dwellings_ml)
    .encode(
        x = alt.X('yrbuilt', scale = alt.Scale(zero = False)),
        y = alt.Y('numbaths', scale = alt.Scale(zero = False)),
        color = 'before1980:O')
        .mark_circle())

# %%
# creates data that is grouped by the year built and numbaths
dat_count = (dwellings_ml
    .groupby(['yrbuilt', 'numbaths'])
    .agg(count = ('nocars', 'size'))
    .reset_index())

# %%
# use alt.color to adjust color scale
(alt.Chart(dat_count)
    .encode(x = 'yrbuilt',
    y = 'numbaths',
    color = alt.Color('count'))
    .mark_rect())

# %%
X_pred = dwellings_ml.drop(['yrbuilt', 'before1980'], axis = 1)
y_pred = dwellings_ml.before1980
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76) 

# %%
# now we use X_train and y_train to build a model.  
# https://scikit-learn.org/stable/modules/tree.html#classification
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# now we use X_train and y_train to build a model.  
# %%
# https://scikit-learn.org/stable/modules/tree.html#classification
clf = GradientBoostingClassifier()
clf = clf.fit(X_train, y_train)
predict_p =  clf.predict(X_test)
# %%
print(metrics.confusion_matrix(y_test, predict_p))
metrics.plot_confusion_matrix(clf, X_test, y_test)
# %%
print(metrics.classification_report(y_test, predict_p))
# %%
# shows the p-values
df_features = (pd.DataFrame(
        {'f_names': X_train.columns, 
        'f_values': clf.feature_importances_})
    .sort_values('f_values', ascending = False))
# %%
# %%
import sys
!{sys.executable} -m pip install seaborn scikit-learn
# %%
# the full imports
import pandas as pd 
import numpy as np
import seaborn as sns
import altair as alt
# %%
# the from imports
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
# %%
dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")   
alt.data_transformers.enable('json')
# %%
h_subset = dwellings_ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980']).sample(500)
sns.pairplot(h_subset, hue = 'before1980')
corr = h_subset.drop(columns = 'before1980').corr()
# %%
sns.heatmap(corr)
# %%
(alt.Chart(dwellings_ml)
    .encode(
        x = alt.X('yrbuilt', scale=alt.Scale(zero=False), axis=alt.Axis(format='.0f')), 
        y = alt.Y('numbaths', scale=alt.Scale(zero=False)), 
        color = 'before1980:O')
    .mark_circle()
)
# %%
dat_count = (dwellings_ml
    .groupby(['yrbuilt', 'numbaths'])
    .agg(count = ('nocars', 'size'))
    .reset_index())
# %%
# not quite working yet
(alt.Chart(dat_count)
    .encode(
        alt.X('yrbuilt:O',
            scale = alt.Scale(zero=False),
            axis=alt.Axis(format='.0f')), 
        alt.Y('numbaths:O',scale = alt.Scale(zero=False)), 
        color = alt.Color('count', 
            scale=alt.Scale(type='log')))
    .mark_rect()
    .properties(width=400)
)
# %%
(alt.Chart(dat_count)
    .encode(
        alt.X('yrbuilt:O',
            scale = alt.Scale(zero=False),
            axis=alt.Axis(format='.0f')), 
        alt.Y('numbaths',scale = alt.Scale(zero=False)))
    .mark_boxplot(size = 3)
    .properties(width=650)
)
# %%
base = (alt.Chart(dwellings_ml)
    .encode(
    alt.X('yrbuilt', bin=alt.Bin(step = 1),
        axis=alt.Axis(format='.0f')),
    alt.Y('numbaths', bin=alt.Bin(step=1)),
        color=alt.Color('count()', 
            scale=alt.Scale(type='log')))
)
base.mark_rect() 
# %%
base = (alt.Chart(dwellings_ml)
    .encode(
    alt.X('yrbuilt', bin=alt.Bin(step = 5),
        axis=alt.Axis(format='.0f')),
    alt.Y('numbaths', bin=alt.Bin(step=1)),
        color=alt.Color('count()', 
            scale=alt.Scale(type='log')))
)
base.mark_rect() 
# %%
X_pred = dwellings_ml.drop(['yrbuilt', 'before1980'], axis = 1)
y_pred = dwellings_ml.before1980
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76) 
# now we use X_train and y_train to build a model.  
# %%
# https://scikit-learn.org/stable/modules/tree.html#classification
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
# %%