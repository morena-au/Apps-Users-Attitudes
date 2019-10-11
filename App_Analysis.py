# %% [markdown]
# # Apps Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as stm
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

Apps_clean = pd.read_csv(r'./Apps_clean.csv')
# Unamed equal previous indexing

# %%
# V2 differences in frequency between utilitarian and hedonic
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True,
                        sharey=True, figsize=(15, 10))
fig.suptitle('V2 (Utilitarian vs. Hedonic) & V4 (Frequncy [1++, 7--])',
             fontsize=15, fontweight='bold')
sns.boxplot(x='V2', y='V4_1_Interview', data=Apps_clean,
            ax=axs[0])
axs[0].set_title('First Interview')
sns.boxplot(x='V2', y='V4_2_Interview', data=Apps_clean,
            ax=axs[1])
axs[1].set_title('Second Interview')
sns.boxplot(x='V2', y='V4_3_Interview', data=Apps_clean,
            ax=axs[2])
axs[2].set_title('Third Interview')
sns.boxplot(x='V2', y='V4_4_Interview', data=Apps_clean,
            ax=axs[3])
axs[3].set_title('Forth Interview')
fig.subplots_adjust(hspace=0.8)
plt.show()

# %%
# Dependent variable distribution
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True,
                        sharey=True, figsize=(15, 10))
fig.suptitle('V4 (Frequncy [1++, 7--]) Univariate Distribution',
             fontsize=15, fontweight='bold')
sns.distplot(Apps_clean['V4_1_Interview'], bins=np.arange(1, 9),
             hist_kws=dict(ec="k"), kde=False, ax=axs[0])
axs[0].set_title('First Interview')
sns.distplot(Apps_clean['V4_2_Interview'], bins=np.arange(1, 9),
             hist_kws=dict(ec="k"), kde=False, ax=axs[1])
axs[1].set_title('Second Interview')
sns.distplot(Apps_clean['V4_3_Interview'], bins=np.arange(1, 9),
             hist_kws=dict(ec="k"), kde=False, ax=axs[2])
axs[2].set_title('Third Interview')
sns.distplot(Apps_clean['V4_4_Interview'], bins=np.arange(1, 9),
             hist_kws=dict(ec="k"), kde=False, ax=axs[3])
axs[3].set_title('Forth Interview')
fig.subplots_adjust(hspace=0.8)
plt.show()

# %% [markdown]
# ### MODEL 1: Linear Regression between V4 and V2

# %%
# create a dummy variable
V2_enc = OneHotEncoder(categories='auto', drop='first')
V2_enc = V2_enc.fit(Apps_clean['V2'].values.reshape(-1, 1))
V2_enc.categories_
# 1 if ith apps is hedonic
# 0 if ith apps is utilitarian
# b0 interpreted as the average frequency among utilitarian apps
# b0 + b1 average frequency among hendonic apps
X = V2_enc.transform(Apps_clean['V2'].values.reshape(-1, 1)).toarray()

# Using statsmodels.api

model = stm.OLS(Apps_clean['V4_1_Interview'], stm.add_constant(X)).fit()
# P-values are very low. This indicates that there is statistical
# evidence of a difference in average frequency between V2
# (utilitarian vs. hedonic apps), but it does not explain variance
print(model.summary())

# %%
# Check if they change with the time/they.
# Does not seems the case looking at the graphs
# run Linear Regression models:
for i in ['V4_1_Interview', 'V4_2_Interview',
          'V4_3_Interview', 'V4_4_Interview']:
  model = stm.OLS(Apps_clean[i], stm.add_constant(X)).fit()
  print(model.summary())

# %% [markdown]
# ### MODEL 2: Linear Regression between V4 and V3

# %%
# V3 (free vs paid for)
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, 
                        sharey=True, figsize=(15, 10))
fig.suptitle('V3 (Free vs. Paid for) & V4 (Frequncy [1++, 7--])',
             fontsize=15, fontweight='bold')
sns.boxplot(x='V3', y='V4_1_Interview', data=Apps_clean,
            ax=axs[0])
axs[0].set_title('First Interview')
sns.boxplot(x='V3', y='V4_2_Interview', data=Apps_clean,
            ax=axs[1])
axs[1].set_title('Second Interview')
sns.boxplot(x='V3', y='V4_3_Interview', data=Apps_clean,
            ax=axs[2])
axs[2].set_title('Third Interview')
sns.boxplot(x='V3', y='V4_4_Interview', data=Apps_clean,
            ax=axs[3])
axs[3].set_title('Forth Interview')
fig.subplots_adjust(hspace=0.8)
plt.show()

# %%
# create a dummy variable
V3_enc = OneHotEncoder(categories='auto', drop='first')
V3_enc = V3_enc.fit(Apps_clean['V3'].values.reshape(-1, 1))
V3_enc.categories_
# 1 if ith apps is paid for
# 0 if ith apps is free
# b0 interpreted as the average frequency among free apps
# b0 + b1 average frequency among paid for apps
del X
X = V3_enc.transform(Apps_clean['V3'].values.reshape(-1, 1)).toarray()

# create the model using statsmodels.api
for i in ['V4_1_Interview', 'V4_2_Interview',
          'V4_3_Interview', 'V4_4_Interview']:
  model = stm.OLS(Apps_clean[i], stm.add_constant(X)).fit()
  print(model.summary())

# p-value is small so there is statistical evidence of a difference
# in average frequency between V3 (free vs paid for apps)
# the relation does not change with time

# %% [markdown]
# ### MODEL 3: Linear Regression between V4 and V6

# %%
pd.value_counts(Apps_clean['V6_2_Interview'])

# %%
# V6 univariate distribution
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True,
                        sharey=True, figsize=(15, 10))
fig.suptitle('V6 Number of functions [1--, 10++] - Univariate Distribution',
             fontsize=15, fontweight='bold')
sns.distplot(Apps_clean['V6_1_Interview'], bins=np.arange(1, 12),
             hist_kws=dict(ec="k"), kde=False, ax=axs[0])
axs[0].set_title('First Interview')
sns.distplot(Apps_clean['V6_2_Interview'], bins=np.arange(1, 12),
             hist_kws=dict(ec="k"), kde=False, ax=axs[1])
axs[1].set_title('Second Interview')
sns.distplot(Apps_clean['V6_3_Interview'], bins=np.arange(1, 12),
             hist_kws=dict(ec="k"), kde=False, ax=axs[2])
axs[2].set_title('Third Interview')
sns.distplot(Apps_clean['V6_4_Interview'], bins=np.arange(1, 12),
             hist_kws=dict(ec="k"), kde=False, ax=axs[3])
axs[3].set_title('Forth Interview')
fig.subplots_adjust(hspace=0.8)
plt.show()

# %%
hexplot = sns.jointplot('V6_1_Interview', 'V4_1_Interview', data=Apps_clean,
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V6 Number of functions [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %%
# create the model using statsmodels.api
model = stm.OLS(Apps_clean['V4_1_Interview'],
                stm.add_constant(Apps_clean['V6_1_Interview'])).fit()
print(model.summary())

# P-value and low R-squarred: the regression model has significant variables
# but explains little of the variability.
# The trend indicates that the predictor vairable still provides information
# about the response even though data points fall
# further from the regression line.

# %% [markdown]
# ### MODEL 4: Linear Regression between V4 and V10 extended functions
hexplot = sns.jointplot(Apps_clean['V10'], Apps_clean['V4_2_Interview'],
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-0.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V10 Extended functions [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %%
# it needs to be standardize otherwise not useful
# feature engineering binarize (always - not always)
pd.value_counts(Apps_clean['V10'])

# %% [markdown]
# ### MODEL 5: Linear Regression between V4 and V11 cross app use [1--, 10++]

# %%
# feature engineering binarize (always - not always)
pd.value_counts(Apps_clean['V11'])

# %%
hexplot = sns.jointplot(Apps_clean['V11'], Apps_clean['V4_2_Interview'],
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-0.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V11 Cross Apps Use [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %% [markdown]
# ### MODEL 6: Linear Regression between V4 and V12 Satisfaction [1--, 10++]
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, 
                        sharey=True, figsize=(15, 10))
fig.suptitle('V12 Satisfaction [1--, 10++] - Univariate Distribution',
             fontsize=15, fontweight='bold')
sns.distplot(Apps_clean['V12_1_Interview'], bins=np.arange(1, 12),
             hist_kws=dict(ec="k"), kde=False, ax=axs[0])
axs[0].set_title('First Interview')
sns.distplot(Apps_clean['V12_2_Interview'], bins=np.arange(1, 12),
             hist_kws=dict(ec="k"), kde=False, ax=axs[1])
axs[1].set_title('Second Interview')
sns.distplot(Apps_clean['V12_3_Interview'], bins=np.arange(1, 12),
             hist_kws=dict(ec="k"), kde=False, ax=axs[2])
axs[2].set_title('Third Interview')
sns.distplot(Apps_clean['V12_4_Interview'], bins=np.arange(1, 12),
             hist_kws=dict(ec="k"), kde=False, ax=axs[3])
axs[3].set_title('Forth Interview')
fig.subplots_adjust(hspace=0.8)
plt.show()

# Distribution doesn't change much with the time

# %%
hexplot = sns.jointplot(Apps_clean['V12_1_Interview'],
                        Apps_clean['V4_1_Interview'], kind='hex')
cbar_ax = hexplot.fig.add_axes([-0.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V12 Satisfaction [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %%
# create the model using statsmodels.api
model = stm.OLS(Apps_clean['V4_1_Interview'],
                stm.add_constant(Apps_clean['V12_1_Interview'])).fit()
print(model.summary())

# %% [markdown]
# ### MODEL 7: Linear Regression between V4 and V13 Future use [1--, 10++]
pd.value_counts(Apps_clean['V13'])
hexplot = sns.jointplot(Apps_clean['V13'], Apps_clean['V4_4_Interview'],
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-0.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V13 Future Use [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %%
# create the model using statsmodels.api
model = stm.OLS(Apps_clean['V4_1_Interview'],
                stm.add_constant(Apps_clean['V13'])).fit()
print(model.summary())

# is it relevant though? Aren't we measuring the same thing

# %% [markdown]
# ### MODEL 7: Linear Regression between V4 and V14 recommendation [1--,10++]
pd.value_counts(Apps_clean['V14'])

# %%
sns.kdeplot(Apps_clean['V12_1_Interview'], shade=True)
sns.kdeplot(Apps_clean['V13'], shade=True)
sns.kdeplot(Apps_clean['V14'], shade=True)
plt.title('Univariate distribution of V12, V13, V14')
plt.show()

# %%
# So I expect the same behaviour of V12, V13, V14
# Joint probability distribution
hexplot = sns.jointplot(Apps_clean['V14'], Apps_clean['V4_1_Interview'],
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-0.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V14 Reccomendation [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %%
# create the model using statsmodels.api
model = stm.OLS(Apps_clean['V4_1_Interview'],
                stm.add_constant(Apps_clean['V14'])).fit()
print(model.summary())

# %% [markdown]
# ### CORRELATION MAP

# %%
# index, identifiers and categorical variables
X = Apps_clean.drop(['Unnamed: 0', 'Probanden_ID__lfdn__AppNr',
                     'Probanden_ID__lfdn',
                     'Datum_1_Interview', 'Datum_2_Interview',
                     'Datum_3_Interview',
                     'Datum_4_Interview', 'V1', 'V01', 'V2', 'V3',
                     'Miss_row_%', 'Days_Between_2_and_1_Interview',
                     'Days_Between_3_and_2_Interview',
                     'Days_Between_4_and_3_Interview'], axis=1)
corr = X.corr()

fig, ax = plt.subplots(figsize=(25, 15))
sns.heatmap(corr, center=0, linewidths=0.5)
plt.show()

# V17, V18, V19, V20, V21
# Statistically different means (V18 weaker than other)
# expect the same behaviour

# %% [markdown]
# ### MODEL 8: Multi variate statistics and mediation relationship

# 1. check assumptions that observations are time independent
# V4
Apps_clean['D4_2_1'] = Apps_clean['V4_2_Interview'] -
                       Apps_clean['V4_1_Interview']
Apps_clean['D4_3_2'] = Apps_clean['V4_3_Interview'] - Apps_clean['V4_2_Interview']
Apps_clean['D4_4_3'] = Apps_clean['V4_4_Interview'] - Apps_clean['V4_3_Interview']
Apps_clean['D4_4_1'] = Apps_clean['V4_4_Interview'] - Apps_clean['V4_1_Interview']

fig, axs = plt.subplots(4, sharex=True, sharey=True)
fig.suptitle('V4 (frequency) difference between interviews',
             fontsize=18, fontweight='bold')
sns.boxplot(Apps_clean['D4_2_1'], ax=axs[0])
axs[0].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D4_2_1'])),
                 fontsize=9)
sns.boxplot(Apps_clean['D4_3_2'], ax=axs[1])
axs[1].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D4_3_2'])),
                 fontsize=9)
sns.boxplot(Apps_clean['D4_4_3'], ax=axs[2])
axs[2].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D4_4_3'])),
                 fontsize=9)
sns.boxplot(Apps_clean['D4_4_1'], ax=axs[3])
axs[3].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D4_4_1'])),
                 fontsize=9)
fig.subplots_adjust(top=0.85, hspace=1.5)
plt.show()

pd.value_counts(Apps_clean['D4_4_3'])
# COMMENTS: 
# Most of the observations 50% didn't change with time
# Biggest differences bewteen the second and first interview where 
# where the frequency for 20% is increased by 1

fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, 
                        sharey=True, figsize=(15, 10))
fig.suptitle('V4 Difference - Univariate Distribution',
             fontsize=15, fontweight='bold')
sns.distplot(Apps_clean['D4_2_1'], bins=np.arange(-6, 7),
             hist_kws=dict(ec="k"), kde=False, ax=axs[0])
axs[0].set_title('V4 difference between 2 and 1 interview', fontsize=8)
sns.distplot(Apps_clean['D4_3_2'], bins=np.arange(-6, 7),
             hist_kws=dict(ec="k"), kde=False, ax=axs[1])
axs[1].set_title('V4 difference between 3 and 2 interview', fontsize=8)
sns.distplot(Apps_clean['D4_4_3'], bins=np.arange(-6, 7),
             hist_kws=dict(ec="k"), kde=False, ax=axs[2])
axs[2].set_title('V4 difference between 4 and 3 interview', fontsize=8)
sns.distplot(Apps_clean['D4_4_1'], bins=np.arange(-6, 7),
             hist_kws=dict(ec="k"), kde=False, ax=axs[3])
axs[3].set_title('V4 difference between 4 and 1 interview', fontsize=8)
fig.subplots_adjust(hspace=0.8)
plt.show()

# V6
Apps_clean['D6_2_1'] = Apps_clean['V6_2_Interview'] - Apps_clean['V6_1_Interview']
Apps_clean['D6_3_2'] = Apps_clean['V6_3_Interview'] - Apps_clean['V6_2_Interview']
Apps_clean['D6_4_3'] = Apps_clean['V6_4_Interview'] - Apps_clean['V6_3_Interview']
Apps_clean['D6_4_1'] = Apps_clean['V6_4_Interview'] - Apps_clean['V6_1_Interview']

fig, axs = plt.subplots(4, sharex=True, sharey=True)
fig.suptitle('V6 (function) difference between interviews',
             fontsize=18, fontweight='bold')
sns.boxplot(Apps_clean['D6_2_1'], ax=axs[0])
axs[0].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D6_2_1'])),
                 fontsize=9)
sns.boxplot(Apps_clean['D6_3_2'], ax=axs[1])
axs[1].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D6_3_2'])),
                 fontsize=9)
sns.boxplot(Apps_clean['D6_4_3'], ax=axs[2])
axs[2].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D6_4_3'])),
                 fontsize=9)
sns.boxplot(Apps_clean['D6_4_1'], ax=axs[3])
axs[3].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D6_4_1'])),
                 fontsize=9)
fig.subplots_adjust(top=0.85, hspace=1.5)
plt.show()

# COMMENTS:
# Most of the observations didn't change with the time
# biggest difference are within the 2 and 3 interview.
# In the second interview 20% observations increase frequency by 1
# and another 20% decrease by 1
# In the third inteview 20% observation increase by 1
# this accumulative effect is incorporate in D6_4_1

# %% [Markdown]
# ### MODEL 1a: V6_1, V2 ---> V4_1

# Model V6 on V4 at the starting point during the interview 1
# (since first interview good baseline since 50% of observation
# does not change in the following interviews)
# Mediating for V2 using simple mediation model by Baron and Kenny's

# ## STEP 1: V6_1 on V4_1

model1 = stm.OLS(Apps_clean['V4_1_Interview'],
                stm.add_constant(Apps_clean['V6_1_Interview'])).fit()
print(model1.summary())

# V6_1 coef is significant one unit in increase of functionality
# brings -0.2277 decrease in frequency
# (Remember with frequency less is more)
# TODO consider to invert Frequency so it is more intuitive

## STEP 2: V6_1 on V2
# check mediator effect

# Create dummy variable for V2
V2 = V2_enc.transform(Apps_clean['V2'].values.reshape(-1, 1)).toarray()
# 1 if ith apps is hedonic
# 0 if ith apps is utilitarian

logit2 = stm.Logit(V2, stm.add_constant(Apps_clean['V6_1_Interview'])).fit()
print(logit2.summary())
# V6_1_Interview is significant with p = 0.009
# TODO check interpretation of logit function

## STEP 3: V6_1, V2 ---> V4_1

del X
X = np.hstack([Apps_clean['V6_1_Interview'].values.reshape(-1, 1), V2])

model3 = stm.OLS(Apps_clean['V4_1_Interview'],
                stm.add_constant(X)).fit()
print(model3.summary())

# Coefs are all significant --> Partial mediation
# direct effect of number of functions on frequency
model1.params[1]
# Indirect effect
logit2.params[1]*model3.params[2]
# Partial mediation
abs(model1.params[1]) > abs(model3.params[1])
# Total effect
model3.params[1] + (logit2.params[1]*model3.params[2])

# %% [Markdown]
# ### Model 1b: D6_4_1, V2 ---> D4_4_1
# Model the difference in number of functions onto the difference in frequency
# (Since differences within interviews seems to be commulative)
# (thus changes does not seems to level out >> check further on outliers)
# mediating for V2

# TODO
# STEP 2: V3 (y) --> V2
logit = stm.Logit(V3, stm.add_constant(V2)).fit()
print(logit.summary())

# %%
# STEP 2 (b): V2 (y) <-- V3
logit = stm.Logit(V2, stm.add_constant(V3)).fit()
print(logit.summary())

# %%
# STEP 3: V4 <--- V2, V3
X = np.hstack([V2, V3])

model = stm.OLS(Apps_clean['V4_1_Interview'], stm.add_constant(X)).fit()
print(model.summary())

# %%
# Model V2, V4, V10

# STEP 1: V4 <--- V10
model = stm.OLS(Apps_clean['V4_1_Interview'],
                stm.add_constant(Apps_clean['V10'].\
                                 values.reshape(-1, 1))).fit()
print(model.summary())

# %%
# STEP 2: V2 <--- V10
logit = stm.Logit(V2,
                  stm.add_constant(Apps_clean['V10'].\
                                   values.reshape(-1, 1))).fit()
print(logit.summary())

# %%
# STEP 3: V4 <--- V2, V10
del X
X = np.hstack([V2, Apps_clean['V10'].values.reshape(-1, 1)])
model = stm.OLS(Apps_clean['V4_1_Interview'], stm.add_constant(X)).fit()
print(model.summary())

# %%
# STEP 4: V3 <--- V10
logit = stm.Logit(V3, stm.add_constant(Apps_clean['V10'].\
                  values.reshape(-1, 1))).fit()
print(logit.summary())

# %%
# STEP 5: V4 <--- V2, V3, V10
del X
X = np.hstack([V2, V3, Apps_clean['V10'].values.reshape(-1, 1)])
model = stm.OLS(Apps_clean['V4_1_Interview'], stm.add_constant(X)).fit()
print(model.summary())

# %%
# STEP 5: V4 <--- STEP 5 RESIDUALS, V17
del X
X = np.hstack([model.resid.values.reshape(-1, 1),
               Apps_clean['V17_2_Interview'].values.reshape(-1, 1)])
model = stm.OLS(Apps_clean['V4_1_Interview'], stm.add_constant(X)).fit()
print(model.summary())

## code after the meeting 10.10.2019

