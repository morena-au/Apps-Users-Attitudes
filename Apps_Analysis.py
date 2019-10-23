# %% [markdown]
# # Apps Analysis

# %% [markdown]
# ## 0. Data preprocessing

# %%
# Load library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.mediation import Mediation
import researchpy as rp
from statsmodels.formula.api import ols


# Manage warnings
pd.set_option('mode.chained_assignment', None)

# Import excel files
Data_1 = pd.read_excel(r'./Datasets/Data_1.xlsx')
Data_2 = pd.read_excel(r'./Datasets/Data_2.xlsx')
Data_3 = pd.read_excel(r'./Datasets/Data_3.xlsx')
Data_4 = pd.read_excel(r'./Datasets/Data_4.xlsx')
Personality = pd.read_excel(r'./Datasets/Part_Personality.xlsx')

# Create a new ID column common to all datasets
for i in [Data_1, Data_2, Data_3, Data_4, Personality]:
    # concatenate 'Probanden_ID' and 'lfdn' to create a new unique variable
    # which identify the combination participant-app
    i['Probanden_ID__lfdn'] = i.Probanden_ID.astype(str) + '__' + \
                                  i.lfdn.astype(str)

# Create a new ID column which uniquely identify observations
# on Data_1, Data_2, Data_3, Data_4

# (PS. There are persons with the same initials and apps number
# so the progressive number is necessary)
for i in [Data_1, Data_2, Data_3, Data_4]:
    # concatenate 'Probanden_ID', 'lfdn' and 'AppNr' to create a new variable
    # which identify uniquely each observations
    i['Probanden_ID__lfdn__AppNr'] = i.Probanden_ID.astype(str) + '__' + \
                                     i.lfdn.astype(str) + '__' + \
                                     i.AppNr.astype(str)
    # delete redundant columns
    i.drop(['AppNr', 'lfdn', 'Probanden_ID'], axis=1, inplace=True)

# Check if V1 is the same in all 4 datasets for the combination
# participant + progressive number + app number

# list of dataframes with only the columns under investigation
dataframe_list = [Data_1[['Probanden_ID__lfdn__AppNr', 'V1']],
                  Data_2[['Probanden_ID__lfdn__AppNr', 'V1']],
                  Data_3[['Probanden_ID__lfdn__AppNr', 'V1']],
                  Data_4[['Probanden_ID__lfdn__AppNr', 'V1']]]

# reduce cumulatively compute a function on a list and return the result
V1_merge = reduce(lambda left, right: pd.merge(left, right,
                                               on='Probanden_ID__lfdn__AppNr'),
                  dataframe_list)

# Check all columns against the first column using 'eq'
V1_merge = V1_merge.fillna('-')
# 'all' make sure that all the value along the columns are true
# if sum equal to number rows then V1 is the same for all datasets
sum(V1_merge.filter(like='V1').eq(V1_merge.iloc[:, 1],
                                  axis=0).all(axis=1)) == Data_1.shape[0]

# Since they are all the same drop V1 from other datasets
for i in [Data_2, Data_3, Data_4]:
    i.drop('V1', axis=1, inplace=True)

# Rename cols before merging adding the number of the interview
# which is refering too
for num, data in enumerate([Data_1, Data_2, Data_3, Data_4]):
    for col_name in ['Datum', 'Interviewer', 'V4', 'V5', 'V6', 'V12',
                     'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23']:
        if any(data.columns == col_name):
            data.rename(columns={col_name: col_name +
                                 '_' + str(num + 1) +
                                 '_Interview'}, inplace=True)

# Merge datasets
for i in [Data_2, Data_3, Data_4]:
    i.drop('Probanden_ID__lfdn', axis=1, inplace=True)

dataframe_list = [Data_1, Data_2, Data_3, Data_4]

Apps = reduce(lambda left, right: pd.merge(left, right,
                                           on='Probanden_ID__lfdn__AppNr'),
              dataframe_list)

# Reorder columns
col = ['Probanden_ID__lfdn__AppNr', 'Probanden_ID__lfdn', 'Datum_1_Interview',
       'Datum_2_Interview', 'Datum_3_Interview', 'Datum_4_Interview',
       'Interviewer_1_Interview', 'Interviewer_2_Interview',
       'Interviewer_3_Interview', 'Interviewer_4_Interview', 'V1',
       'V01', 'V2', 'V3', 'V4_1_Interview', 'V4_2_Interview', 'V4_3_Interview',
       'V4_4_Interview', 'V5_1_Interview', 'V5_2_Interview', 'V5_3_Interview',
       'V5_4_Interview', 'V6_1_Interview', 'V6_2_Interview', 'V6_3_Interview',
       'V6_4_Interview', 'V7', 'V8', 'V9', 'V10', 'V11', 'Kombi',
       'V12_1_Interview', 'V12_2_Interview', 'V12_3_Interview',
       'V12_4_Interview', 'V13', 'V14', 'V17_2_Interview',
       'V17_3_Interview', 'V17_4_Interview', 'V18_2_Interview',
       'V18_3_Interview', 'V18_4_Interview', 'V19_2_Interview',
       'V19_3_Interview', 'V19_4_Interview', 'V20_2_Interview',
       'V20_3_Interview', 'V20_4_Interview', 'V21_2_Interview',
       'V21_3_Interview', 'V21_4_Interview', 'V22_2_Interview',
       'V23_2_Interview']

Apps = Apps[col]

# Calculate the percentage of missing for each row
# delete the row with high number of missing values(>80%)
values_over_rows = Apps.apply(lambda x: x.count(), axis=1)
Apps['Miss_row_%'] = 100 - values_over_rows/len(Apps.columns)*100
Apps = Apps[Apps['Miss_row_%'] < 80]

# Calculate the percentage of missing for each columns
# delete the columns with high number of missing values (>80%)
miss_cols_pct = pd.DataFrame(100-(Apps.apply(lambda x: x.count(), axis=0) /
                                  Apps.shape[0])*100)
col_del = list(miss_cols_pct[miss_cols_pct[0] > 80].index)  # V8, V9, Kombi
Apps.drop(col_del, axis=1, inplace=True)

# Format date
# Datum_1 filter out the date with a point between them
typo_row_index = []
typo_date = []

for row in list(Apps.index):
    # convert date on the format '03.01.2018'
    if len(str(Apps.loc[row, 'Datum_1_Interview'])) == 10:
        Apps.loc[row, 'Datum_1_Interview'] = \
            datetime.strptime(Apps.loc[row, 'Datum_1_Interview'],
                              '%d.%m.%Y')
    # convert date on the format '43037'
    elif len(str(Apps.loc[row, 'Datum_1_Interview'])) == 5:
        Apps.loc[row, 'Datum_1_Interview'] = \
            datetime.fromordinal(datetime(1900, 1, 1).toordinal() +
                                 int(Apps.loc[row, 'Datum_1_Interview']) - 2)
    elif pd.isna(Apps.loc[row, 'Datum_1_Interview']):
        pass  # do nothing
    else:  # write down the typos
        typo_row_index.append(row)
        typo_date.append(Apps.loc[row, 'Datum_1_Interview'])

# Adjust for typos
np.unique(typo_date)
for row in typo_row_index:
    Apps.loc[row, 'Datum_1_Interview'] = datetime(2017, 12, 20)

# Adjst date_2_interview
typo_row_index = []
typo_date = []

for row in list(Apps.index):
    # convert date on the format '03.01.2018'
    if len(str(Apps.loc[row, 'Datum_2_Interview'])) == 10:
        Apps.loc[row, 'Datum_2_Interview'] = \
            datetime.strptime(Apps.loc[row, 'Datum_2_Interview'],
                              '%d.%m.%Y')
    # if already in timestamp format
    elif isinstance(Apps.loc[row, 'Datum_2_Interview'], datetime):
        pass  # do nothing
    elif pd.isna(Apps.loc[row, 'Datum_2_Interview']):
        pass  # do nothing
    else:  # write down the typos
        typo_row_index.append(row)
        typo_date.append(Apps.loc[row, 'Datum_2_Interview'])

# Adjst for date_3_interview
typo_row_index = []
typo_date = []

for row in list(Apps.index):
    # If it is already an instance of datetime do nothing
    if isinstance(Apps.loc[row, 'Datum_3_Interview'], datetime):
        pass
    # convert date on the format '03.01.2018'
    elif len(str(Apps.loc[row, 'Datum_3_Interview'])) == 10 and \
            Apps.loc[row, 'Datum_3_Interview'].find('.') > 0:
        Apps.loc[row, 'Datum_3_Interview'] = \
            datetime.strptime(Apps.loc[row, 'Datum_3_Interview'],
                              '%d.%m.%Y')
    # convert date on the format '17/05/2018'
    elif len(str(Apps.loc[row, 'Datum_3_Interview'])) == 10 and \
            Apps.loc[row, 'Datum_3_Interview'].find('/') > 0:
        Apps.loc[row, 'Datum_3_Interview'] = \
            datetime.strptime(Apps.loc[row, 'Datum_3_Interview'],
                              '%d/%m/%Y')
    # If it is already in timestamp pass
    elif pd.isna(Apps.loc[row, 'Datum_3_Interview']):
        pass  # do nothing
    else:  # write down any other typos
        typo_row_index.append(row)
        typo_date.append(Apps.loc[row, 'Datum_3_Interview'])

# Adjst for date_4_interview
typo_row_index = []
typo_date = []

for row in list(Apps.index):
    # It is already an instance of datetime pass
    if isinstance(Apps.loc[row, 'Datum_4_Interview'], datetime):
        pass
    # convert date on the format '03.01.2018'
    elif len(str(Apps.loc[row, 'Datum_4_Interview'])) == 10 and \
            Apps.loc[row, 'Datum_4_Interview'].find('.') > 0:
        Apps.loc[row, 'Datum_4_Interview'] = \
            datetime.strptime(Apps.loc[row, 'Datum_4_Interview'],
                              '%d.%m.%Y')
    # convert date on the format '17/05/2018'
    elif len(str(Apps.loc[row, 'Datum_4_Interview'])) == 10 and \
            Apps.loc[row, 'Datum_4_Interview'].find('/') > 0:
        Apps.loc[row, 'Datum_4_Interview'] = \
            datetime.strptime(Apps.loc[row, 'Datum_4_Interview'],
                              '%d/%m/%Y')
    # If it is missing
    elif pd.isna(Apps.loc[row, 'Datum_4_Interview']):
        pass
    else:  # write down the typos
        typo_row_index.append(row)
        typo_date.append(Apps.loc[row, 'Datum_4_Interview'])


# Check if the interviews are progressive
Apps['Days_Between_2_and_1_Interview'] = Apps['Datum_2_Interview'] - \
                                                Apps['Datum_1_Interview']

# correct the typo with year 2108 in Datum_2_Interview
Apps.loc[Apps['Datum_2_Interview'] > datetime(2020, 1, 1),
         'Datum_2_Interview'] = datetime(2018, 4, 11)

# Update calculation
Apps['Days_Between_2_and_1_Interview'] = Apps['Datum_2_Interview'] - \
                                                Apps['Datum_1_Interview']

# Transform the calculation into numbers of days
Apps['Days_Between_2_and_1_Interview'] = \
        Apps['Days_Between_2_and_1_Interview'].apply(lambda x:
                                                     x.total_seconds() /
                                                     60/60/24)

# Difference between 3 and 2 interview
Apps['Days_Between_3_and_2_Interview'] = Apps['Datum_3_Interview'] - \
                                                Apps['Datum_2_Interview']

# Spot typos
pd.unique(Apps.loc[Apps['Datum_3_Interview'] > datetime(2020, 1, 1),
                   'Datum_3_Interview'])
# (2918, 2, 3, 0, 0)
Apps.loc[Apps['Datum_3_Interview'] > datetime(2500, 1, 1),
         'Datum_3_Interview'] = datetime(2018, 2, 3)

# (2020, 5, 17, 0, 0)
Apps.loc[Apps['Datum_3_Interview'] > datetime(2020, 1, 1),
         'Datum_3_Interview'] = datetime(2018, 5, 17)

# Did we have collection of data also during the 2019?
Apps.loc[Apps['Datum_3_Interview'] > datetime(2019, 1, 1),
         'Datum_3_Interview'] = datetime(2018, 5, 17)

Apps['Days_Between_3_and_2_Interview'] = Apps['Datum_3_Interview'] - \
                                                Apps['Datum_2_Interview']

# Into numbers of days
Apps['Days_Between_3_and_2_Interview'] = \
        Apps['Days_Between_3_and_2_Interview'].apply(lambda x:
                                                     x.total_seconds() /
                                                     60/60/24)

# Difference between 4 and 3 interview
Apps['Days_Between_4_and_3_Interview'] = Apps['Datum_4_Interview'] - \
                                                Apps['Datum_3_Interview']

# Spot typos
pd.unique(Apps.loc[Apps['Datum_4_Interview'] > datetime(2020, 1, 1),
                   'Datum_4_Interview'])

# datetime(3018, 3, 4, 0, 0)
Apps.loc[Apps['Datum_4_Interview'] > datetime(2500, 1, 1),
         'Datum_4_Interview'] = datetime(2018, 3, 4)

# Update calculation
Apps['Days_Between_4_and_3_Interview'] = Apps['Datum_4_Interview'] - \
                                                Apps['Datum_3_Interview']

# Find the date after 01/01/2019
np.unique(Apps.loc[Apps['Datum_4_Interview'] > datetime(2019, 1, 1),
          'Datum_4_Interview'])

# Change 2019-07-09 to 2018-07-09
Apps.loc[Apps['Datum_4_Interview'] > datetime(2019, 1, 1),
         'Datum_4_Interview'] = datetime(2018, 7, 9)

# Update the calcualtion
Apps['Days_Between_4_and_3_Interview'] = Apps['Datum_4_Interview'] - \
                                                Apps['Datum_3_Interview']

# Trasform the calculation into numbers of days
Apps['Days_Between_4_and_3_Interview'] = \
        Apps['Days_Between_4_and_3_Interview'].apply(lambda x:
                                                     x.total_seconds() /
                                                     60/60/24)

# %%
# Plot the different frequency graphs
descriptive_2_1 = pd.DataFrame.describe(Apps['Days_Between_2_and_1_Interview'])
descriptive_3_2 = pd.DataFrame.describe(Apps['Days_Between_3_and_2_Interview'])
descriptive_4_3 = pd.DataFrame.describe(Apps['Days_Between_4_and_3_Interview'])

fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(15, 10))
fig.suptitle('Days between 2 consecutive interviews',
             fontsize=15, fontweight='bold')
sns.boxplot(Apps['Days_Between_2_and_1_Interview'].dropna(),
            ax=axs[0])
axs[0].set_title('{}'.format(descriptive_2_1), fontsize=9, fontweight='bold')
sns.boxplot(Apps['Days_Between_3_and_2_Interview'].dropna(),
            ax=axs[1])
axs[1].set_title('{}'.format(descriptive_3_2), fontsize=9, fontweight='bold')
sns.boxplot(Apps['Days_Between_4_and_3_Interview'].dropna(),
            ax=axs[2])
axs[2].set_title('{}'.format(descriptive_4_3), fontsize=9, fontweight='bold')
fig.subplots_adjust(top=0.85, hspace=0.8)
plt.show()

# %%
# KK_5__278 >> 2018 - 2017 - 2017 - 2018
# chenge the middle two dates
Apps.loc[Apps['Datum_2_Interview'] == datetime(2017, 5, 13),
         'Datum_2_Interview'] = datetime(2018, 5, 13)
Apps['Days_Between_2_and_1_Interview'] = Apps['Datum_2_Interview'] - \
                                                Apps['Datum_1_Interview']
# Into numbers of days
Apps['Days_Between_2_and_1_Interview'] = \
        Apps['Days_Between_2_and_1_Interview'].apply(lambda x:
                                                     x.total_seconds() /
                                                     60/60/24)

# Continuos to fix typos
# Think what to do do with the missing date
# (assumption there is a progression within the interviews so keep it)
# TODO

# Delete Interviewer columns
Apps.drop(['Interviewer_1_Interview', 'Interviewer_2_Interview',
           'Interviewer_3_Interview', 'Interviewer_4_Interview'],
          axis=1, inplace=True)

# Delete rows with both V1 and V01 missing, thus no apps listed
# (V01 can be create base on value in V1)

V1_V01_miss_idx = list(Apps.loc[(Apps['V1'].isna()) &
                       (Apps['V01'].isna())].index)

Apps.drop(V1_V01_miss_idx, axis=0, inplace=True)

# check V2 frequency and missing
pd.value_counts(Apps['V2'])

# Eliminate 3,4, 6 values
V2_outbound_idx = list(Apps.loc[Apps['V2'].isin([3, 4, 6])].index)
Apps.drop(V2_outbound_idx, axis=0, inplace=True)

# V3 check missing and frequency
pd.value_counts(Apps['V3'])
V3_outbound_idx = list(Apps.loc[Apps['V3'] == 5].index)
Apps.drop(V3_outbound_idx, axis=0, inplace=True)

# V4_1_Interview check missing and frequency (1++ - 7--)
pd.value_counts(Apps['V4_1_Interview'])
V4_1_outbound_idx = list(Apps.loc[Apps['V4_1_Interview'] == 11].index)
Apps.drop(V4_1_outbound_idx, axis=0, inplace=True)

# V4_2_Interview check missing and frequency (1++ - 7--)
pd.value_counts(Apps['V4_2_Interview'])
V4_2_outbound_idx = list(Apps.loc[(Apps['V4_2_Interview'] > 7)].index)
Apps.drop(V4_2_outbound_idx, axis=0, inplace=True)

# V4_3_Interview check missing and frequency (1++ - 7--)
pd.value_counts(Apps['V4_3_Interview'])
V4_3_outbound_idx = list(Apps.loc[(Apps['V4_3_Interview'] > 7)].index)
Apps.drop(V4_3_outbound_idx, axis=0, inplace=True)

# V4_4_Interview check missing and frequency (1++ - 7--)
pd.value_counts(Apps['V4_4_Interview'])
V4_4_outbound_idx = list(Apps.loc[(Apps['V4_4_Interview'] > 7)].index)
Apps.drop(V4_4_outbound_idx, axis=0, inplace=True)

# V5 > different unit of measurement together delete columns

Apps.drop(['V5_1_Interview', 'V5_2_Interview',
           'V5_3_Interview', 'V5_4_Interview'], axis=1, inplace=True)

# V6 (1 -10) - check out of bound values
Apps.loc[:, ['V6_1_Interview', 'V6_2_Interview',
             'V6_3_Interview', 'V6_4_Interview']].apply(pd.Series.value_counts)

for i in ['V6_1_Interview', 'V6_2_Interview',
          'V6_3_Interview', 'V6_4_Interview']:
    V6_outbound_idx = list(Apps.loc[(Apps[i] > 10)].index)
    Apps.drop(V6_outbound_idx, axis=0, inplace=True)

# V7 delete
Apps.drop(['V7'], axis=1, inplace=True)

# V10 extended function (1 - 10)
pd.value_counts(Apps['V10'])
V10_outbound_idx = list(Apps.loc[(Apps['V10'] > 10)].index)
Apps.drop(V10_outbound_idx, axis=0, inplace=True)

# V11 use across apps
pd.value_counts(Apps['V11'])
V11_outbound_idx = list(Apps.loc[(Apps['V11'] > 10)].index)
Apps.drop(V11_outbound_idx, axis=0, inplace=True)

# V12 satisfaction (1-- - 10++)
Apps.loc[:, ['V12_1_Interview', 'V12_2_Interview',
             'V12_3_Interview',
             'V12_4_Interview']].apply(pd.Series.value_counts)

# V12 has 6 " " need to use coerce to transform it to null
Apps['V12_2_Interview'] = pd.to_numeric(Apps['V12_2_Interview'],
                                        errors='coerce')

# Transform categorical to float
Apps['V12_4_Interview'] = Apps['V12_4_Interview'].astype(float)

# Show summary
pd.DataFrame.describe(Apps.loc[:, ['V12_1_Interview', 'V12_2_Interview',
                                   'V12_3_Interview',
                                   'V12_4_Interview']], include='all')

for i in ['V12_1_Interview', 'V12_2_Interview',
          'V12_3_Interview', 'V12_4_Interview']:
    V12_outbound_idx = list(Apps.loc[(Apps[i] > 10)].index)
    Apps.drop(V12_outbound_idx, axis=0, inplace=True)

# V13 future use (1-- - 10++)
pd.value_counts(Apps['V13'])

# Delete outbounds observations
V13_outbound_idx = list(Apps.loc[(Apps['V13'] > 10)].index)
Apps.drop(V13_outbound_idx, axis=0, inplace=True)

# V14 future use (1-- - 10++)
pd.value_counts(Apps['V14'])

# Delete outboud observation
V14_outbound_idx = list(Apps.loc[(Apps['V14'] > 10)].index)
Apps.drop(V14_outbound_idx, axis=0, inplace=True)

# V17 habit
cols = ['V17_2_Interview', 'V17_3_Interview', 'V17_4_Interview']
Apps[cols].apply(pd.Series.value_counts)

# coerce " " (string na) to NaN (numeric na)
Apps[cols] = Apps[cols].apply(pd.to_numeric, errors='coerce')

Apps[cols].apply(lambda x: sum(x.isna()), axis=0)

# V18 Flow
cols = ['V18_2_Interview', 'V18_3_Interview', 'V18_4_Interview']
Apps[cols].apply(pd.Series.value_counts)

# coerce " " (string na) to NaN (numeric na)
Apps[cols] = Apps[cols].apply(pd.to_numeric, errors='coerce')
Apps[cols].apply(lambda x: sum(x.isna()), axis=0)

# V19 Confirmation
cols = ['V19_2_Interview', 'V19_3_Interview', 'V19_4_Interview']
Apps[cols].apply(pd.Series.value_counts)

# coerce " " (string na) to NaN (numeric na)
Apps[cols] = Apps[cols].apply(pd.to_numeric, errors='coerce')
Apps[cols].apply(lambda x: sum(x.isna()), axis=0)

# V20
cols = ['V20_2_Interview', 'V20_3_Interview', 'V20_4_Interview']
Apps[cols].apply(pd.Series.value_counts)
Apps[cols].apply(lambda x: sum(x.isna()), axis=0)

# V21
cols = ['V21_2_Interview', 'V21_3_Interview', 'V21_4_Interview']
Apps[cols].apply(pd.Series.value_counts)
Apps[cols].apply(lambda x: sum(x.isna()), axis=0)

# delete V22 V23
Apps.drop(['V22_2_Interview', 'V23_2_Interview'], axis=1, inplace=True)

# Drop rows with na
Apps_clean = Apps.dropna()

# Create the final csv file
# Apps_clean.to_csv('Apps_clean.csv')
# Read the file into the workspace
# Apps_clean = pd.read_csv(r'./Apps_clean.csv')

# %% [markdown]
# ## 1. Exploratory Data Analysis

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

# %% [markdown]
# **COMMENT**: <br />
# - Median across interviews is the same <br />
# - Hedonic apps (2.0) are in average more frequenctly used

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
# **COMMENT**: <br />
# - The distribution of frequency seems to be the same across interviews

# %%
# create a dummy variable
V2_enc = OneHotEncoder(categories='auto', drop='first')
V2_enc = V2_enc.fit(Apps_clean['V2'].values.reshape(-1, 1))
V2_enc.categories_
# 1 if ith apps is hedonic
# 0 if ith apps is utilitarian

X = V2_enc.transform(Apps_clean['V2'].values.reshape(-1, 1)).toarray()

model = sm.OLS(Apps_clean['V4_1_Interview'], sm.add_constant(X)).fit()
print(model.summary())

# %% [markdown]
# *b0* interpreted as the average frequency among utilitarian apps. <br />
# *b0 + b1* average frequency among hendonic apps. <br />
# P-values are very low. This indicates that there is statistical
# evidence of a difference in average frequency between V2
# (utilitarian vs. hedonic apps), but it does not explain variance
# R^2 is low.

# %%
# Check if they change with the time.
# (Does not seems to change looking at the graphs)
# run Linear Regression models:
for i in ['V4_1_Interview', 'V4_2_Interview',
          'V4_3_Interview', 'V4_4_Interview']:
    model = sm.OLS(Apps_clean[i], sm.add_constant(X)).fit()
    print(model.summary())

# %% [markdown]
# The coefficients does not significantly change across interview

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

# %% [markdown]
# 1. Median does not change across interviews
# 2. Paid for apps (2.0) are more frequently used

# %%
# create a dummy variable
V3_enc = OneHotEncoder(categories='auto', drop='first')
V3_enc = V3_enc.fit(Apps_clean['V3'].values.reshape(-1, 1))
V3_enc.categories_
# 1 if ith apps is paid for
# 0 if ith apps is free

del X
X = V3_enc.transform(Apps_clean['V3'].values.reshape(-1, 1)).toarray()

# create the model using statsmodels.api
for i in ['V4_1_Interview', 'V4_2_Interview',
          'V4_3_Interview', 'V4_4_Interview']:
    model = sm.OLS(Apps_clean[i], sm.add_constant(X)).fit()
    print(model.summary())

# %% [Markdown]
# b0 interpreted as the average frequency among free apps
# b0 + b1 average frequency among paid for apps
# p-value is small so there is statistical evidence of a difference
# in average frequency between V3 (free vs paid for apps).
# The relation does not change with time.

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


# %% [markdown]
# V6 distribution does not seems to change across interviews

# %%
hexplot = sns.jointplot('V6_1_Interview', 'V4_1_Interview', data=Apps_clean,
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V6 Number of functions [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()


# %% [markdown]
# **COMMENT**: <br />
# Apps with more functions tend to be used more.

# %%
# create the model using statsmodels.api
model = sm.OLS(Apps_clean.loc[:, 'V4_1_Interview'],
               sm.add_constant(Apps_clean.loc[:, 'V6_1_Interview'])).fit()
print(model.summary())

# %% [markdown]
# P-value and R-squarred are low: the regression model has significant
# variables but explains little of the variability.
# The trend indicates that the predictor vairable still provides information
# about the response even though data points fall
# further from the regression line.

# %% [markdown]
# Regression between V4 and V10 extended functions

# %%
hexplot = sns.jointplot(Apps_clean['V10'], Apps_clean['V4_2_Interview'],
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-0.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V10 Extended functions [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %% [markdown]
# V10 cannot explain variability since almost all in 1. <br />
# feature engineering binarize (always - not always)

# %% [markdown]
# Linear Regression between V4 and V11 cross app use [1--, 10++]

# %%
hexplot = sns.jointplot(Apps_clean['V11'], Apps_clean['V4_2_Interview'],
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-0.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V11 Cross Apps Use [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %% [markdown]
# V11 cannot explain variability since almost all in 1. <br />
# feature engineering binarize (always - not always)

# %% [markdown]
# Linear Regression between V4 and V12 Satisfaction [1--, 10++]

# %%
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

# %% [markdown]
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
model = sm.OLS(Apps_clean.loc[:, 'V4_1_Interview'],
               sm.add_constant(Apps_clean.loc[:, 'V12_1_Interview'])).fit()
print(model.summary())

# %% [markdown]
# **COMMENTS**:<br />
# - Statistic evidence different means
# - Don't change over time (look distribution)
# - People more satisfy tend to use the app moren

# %% [markdown]
# Linear Regression between V4 and V13 Future use \[1--, 10++\]

# %%
pd.value_counts(Apps_clean['V13'])
hexplot = sns.jointplot(Apps_clean['V13'], Apps_clean['V4_4_Interview'],
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([-0.05, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.title('V13 Future Use [1--, 10++] vs V4 Frequency [1++, 7--] ',
          x=20, y=1.8, fontsize=15, fontweight='bold')
plt.show()

# %% [markdown]
# High 10s in V3

# %%
# create the model using statsmodels.api
model = sm.OLS(Apps_clean.loc[:, 'V4_1_Interview'],
               sm.add_constant(Apps_clean.loc[:, 'V13'])).fit()
print(model.summary())

# %% [markdown]
# **COMMENT**: <br />
# - Statistic evidence different means <br />
# - More future use tend to use the app more

# %% [markdown]
# Linear Regression between V4 and V14 recommendation \[1--,10++\]

# %%
sns.kdeplot(Apps_clean['V12_1_Interview'], shade=True)
sns.kdeplot(Apps_clean['V13'], shade=True)
sns.kdeplot(Apps_clean['V14'], shade=True)
plt.title('Univariate distribution of V12, V13, V14')
plt.show()

# %% [markdown]
# **COMMENT**: <br />
# Univariate distribution shows higher values for V12, V13, V14
# with a weaker effect for V14

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

# %% [markdown]
# Higher values indicates more frequently used apps.

# %%
# create the model using statsmodels.api
model = sm.OLS(Apps_clean.loc[:, 'V4_1_Interview'],
               sm.add_constant(Apps_clean.loc[:, 'V14'])).fit()
print(model.summary())

# %% [markdown]
# There is statistic evidence different means
# Higher recommendation brings higher use frequency

# %% [markdown]
# ### CORRELATION MAP

# %%
# index, identifiers and categorical variables
X = Apps_clean.drop(['Probanden_ID__lfdn__AppNr',
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

# %% [markdown]
# **COMMENT**:<br />
# V17, V18, V19, V20, V21 show positive correlation between the
# dependent variables and negative correlation with V4.
# V18 weaker than other.

# %% [markdown]
# ## 2. MODELLING
# Multi variate statistics and mediation relationship

# %% [markdown]
# #### Check assumptions that observations are time independent
# %%
# V4
Apps_clean.loc[:, 'D4_2_1'] = Apps_clean.loc[:, 'V4_2_Interview'] - \
                              Apps_clean.loc[:, 'V4_1_Interview']
Apps_clean.loc[:, 'D4_3_2'] = Apps_clean.loc[:, 'V4_3_Interview'] - \
                              Apps_clean.loc[:, 'V4_2_Interview']
Apps_clean.loc[:, 'D4_4_3'] = Apps_clean.loc[:, 'V4_4_Interview'] - \
                              Apps_clean.loc[:, 'V4_3_Interview']
Apps_clean.loc[:, 'D4_4_1'] = Apps_clean.loc[:, 'V4_4_Interview'] - \
                              Apps_clean.loc[:, 'V4_1_Interview']

fig, axs = plt.subplots(4, sharex=True, sharey=True, figsize=(15, 15))
fig.suptitle('V4 (frequency) difference between interviews',
             fontsize=18, fontweight='bold')
sns.boxplot(Apps_clean['D4_2_1'], ax=axs[0])
axs[0].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D4_2_1'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D4_3_2'], ax=axs[1])
axs[1].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D4_3_2'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D4_4_3'], ax=axs[2])
axs[2].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D4_4_3'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D4_4_1'], ax=axs[3])
axs[3].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D4_4_1'])),
                 fontsize=10)
fig.subplots_adjust(top=0.85, hspace=1.5)
plt.show()

# %% [markdown]
# **COMMENTS**: <br />
# Most of the observations 50% didn't change with time
# Biggest differences bewteen the second and first interview where
# the frequency of 20% obs is increased by 1.

# %%
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True,
                        sharey=True, figsize=(15, 10))
fig.suptitle('V4 Difference - Univariate Distribution',
             fontsize=15, fontweight='bold')
sns.distplot(Apps_clean['D4_2_1'], bins=np.arange(-6, 7),
             hist_kws=dict(ec="k"), kde=False, ax=axs[0])
axs[0].set_title('V4 difference between 2 and 1 interview', fontsize=9)
sns.distplot(Apps_clean['D4_3_2'], bins=np.arange(-6, 7),
             hist_kws=dict(ec="k"), kde=False, ax=axs[1])
axs[1].set_title('V4 difference between 3 and 2 interview', fontsize=9)
sns.distplot(Apps_clean['D4_4_3'], bins=np.arange(-6, 7),
             hist_kws=dict(ec="k"), kde=False, ax=axs[2])
axs[2].set_title('V4 difference between 4 and 3 interview', fontsize=9)
sns.distplot(Apps_clean['D4_4_1'], bins=np.arange(-6, 7),
             hist_kws=dict(ec="k"), kde=False, ax=axs[3])
axs[3].set_title('V4 difference between 4 and 1 interview', fontsize=9)
fig.subplots_adjust(hspace=0.8)
plt.show()

# %%
# V6
Apps_clean['D6_2_1'] = Apps_clean['V6_2_Interview'] - \
                       Apps_clean['V6_1_Interview']
Apps_clean['D6_3_2'] = Apps_clean['V6_3_Interview'] - \
                       Apps_clean['V6_2_Interview']
Apps_clean['D6_4_3'] = Apps_clean['V6_4_Interview'] - \
                       Apps_clean['V6_3_Interview']
Apps_clean['D6_4_1'] = Apps_clean['V6_4_Interview'] - \
                       Apps_clean['V6_1_Interview']

fig, axs = plt.subplots(4, sharex=True, sharey=True, figsize=(15, 15))
fig.suptitle('V6 (function) difference between interviews',
             fontsize=18, fontweight='bold')
sns.boxplot(Apps_clean['D6_2_1'], ax=axs[0])
axs[0].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D6_2_1'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D6_3_2'], ax=axs[1])
axs[1].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D6_3_2'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D6_4_3'], ax=axs[2])
axs[2].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D6_4_3'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D6_4_1'], ax=axs[3])
axs[3].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D6_4_1'])),
                 fontsize=10)
fig.subplots_adjust(top=0.85, hspace=1.5)
plt.show()

# %% [markdown]
# **COMMENTS**: <br />
# Most of the observations didn't change with the time.
# Biggest difference are within the 2 and 3 interview.
# In the second interview 20% observations increase frequency by 1
# and another 20% decrease by 1.
# In the third inteview 20% observation increase by 1.
# This accumulative effect is incorporate in D6_4_1.

# %%
# V12
Apps_clean['D12_2_1'] = Apps_clean['V12_2_Interview'] - \
                       Apps_clean['V12_1_Interview']
Apps_clean['D12_3_2'] = Apps_clean['V12_3_Interview'] - \
                       Apps_clean['V12_2_Interview']
Apps_clean['D12_4_3'] = Apps_clean['V12_4_Interview'] - \
                       Apps_clean['V12_3_Interview']
Apps_clean['D12_4_1'] = Apps_clean['V12_4_Interview'] - \
                       Apps_clean['V12_1_Interview']

fig, axs = plt.subplots(4, sharex=True, sharey=True, figsize=(15, 15))
fig.suptitle('V12 (satisfaction) difference between interviews',
             fontsize=18, fontweight='bold')
sns.boxplot(Apps_clean['D12_2_1'], ax=axs[0])
axs[0].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D12_2_1'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D12_3_2'], ax=axs[1])
axs[1].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D12_3_2'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D12_4_3'], ax=axs[2])
axs[2].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D12_4_3'])),
                 fontsize=10)
sns.boxplot(Apps_clean['D12_4_1'], ax=axs[3])
axs[3].set_title('{}'.format(pd.DataFrame.describe(Apps_clean['D12_4_1'])),
                 fontsize=10)
fig.subplots_adjust(top=0.85, hspace=1.5)
plt.show()

# %% [markdown]
# **COMMENT**: <br />
# There are some "level off" effects.
# Investigate further the relationship (TODO)

# %% [markdown]
# #### MODEL 1a: V6_1, V2 ---> V4_1
# **STEP** **1**: V6_1 on V4_1

# %%
# Model V6 on V4 at the starting point, during the interview 1
# (since first interview good baseline since 50% of observation
# does not change in the following interviews)
# Mediating for V2 using simple mediation model by Baron and Kenny's
model1 = sm.OLS(Apps_clean.loc[:, 'V4_1_Interview'],
                sm.add_constant(Apps_clean.loc[:, 'V6_1_Interview'].
                                values.reshape(-1, 1))).fit()
print(model1.summary())

# %% [markdown]
# **COMMENT**: <br />
# *V6_1* coef is significant one unit in increase of functionality
# brings -0.2277 decrease in frequency (Remember with frequency less is more)
# (TODO) consider to invert Frequency so it is more intuitive

# %% [markdown]
# **STEP** **2**: V6_1 on V2

# %%
# Check mediator effect
# Create dummy variable for V2
V2 = V2_enc.transform(Apps_clean.loc[:, 'V2'].values.reshape(-1, 1)).toarray()
# 1 if ith apps is hedonic
# 0 if ith apps is utilitarian

logit2 = sm.Logit(V2, sm.add_constant(Apps_clean.
                                      loc[:, 'V6_1_Interview'].
                                      values.reshape(-1, 1))).fit()
print(logit2.summary())

# %% [markdown]
# **COMMENT**: <br />
# *V6_1_Interview* is significant with p = 0.009. <br />
# The intercept is the estimated log odds (log(p/1-p)) of an app with a number
# of functions (*V6*) of zero being an hedonic apps (*V2*).<br />
# The conditional logit of being an hedonic app when the number of functions
# is held to 5: $b0 + b1*V6 >> -0,3297 + (0,0356*5)$<br />
# For 1 unit increase in the number function, the expected change in the log
# odds of being an hedonic app is $b1 = 0,0356§.

# %% [markdown]
# **STEP** **3**: V6_1, V2 ---> V4_1

# %%
del X
X = np.hstack([Apps_clean['V6_1_Interview'].values.reshape(-1, 1), V2])

model3 = sm.OLS(Apps_clean.loc[:, 'V4_1_Interview'],
                sm.add_constant(X)).fit()
print(model3.summary())

# %% [markdown]
# Coefs are all significant --> Partial mediation
# %%
# Direct effect of number of functions on frequency
model1.params[1]
# %%
# Indirect effect
logit2.params[1]*model3.params[2]
# %%
# Partial mediation coef V6 on step 1 greater then coef V6 step 3
abs(model1.params[1]) > abs(model3.params[1])
# %%
# Total effect
model3.params[1] + (logit2.params[1]*model3.params[2])

# %%
# Using mediation model in stasmodels (Imai, Keele, Tingley (2010))
# Regression model for the outcome
Apps_clean['V2'] = V3_enc.transform(Apps_clean['V2'].
                                    values.reshape(-1, 1)).toarray()

outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview + \
                                     V2',
                                    data=Apps_clean)

# Regression model for the mediator variable
mediator_model = sm.GLM.from_formula('V2 ~ V6_1_Interview ',
                                     data=Apps_clean,
                                     family=sm.families.Binomial())

med = Mediation(outcome_model, mediator_model, 'V6_1_Interview', 'V2')

med_result = med.fit()

med_result.summary()

# %% [markdown]
# **COMMENT**:<br />
# The *ACME* (Average Causal Mediation Effect) is not statistically
# distinct from zero.

# %% [markdown]
# #### Model 1b: D6_4_1, V2 ---> D4_4_1 <br />
# Model the difference in number of functions (*V6*) onto the
# difference in frequency (*V4*) mediating for *V2*.
# (Since differences within interviews seems to be commulative,
# thus changes does not seems to level out >> check further on outliers
# to spot trends within time. TODO)

# %% [markdown]
# **STEP** **1**: D6_4_1 on D4_4_1

# %%
model1 = sm.OLS(Apps_clean.loc[:, 'D4_4_1'],
                sm.add_constant(Apps_clean.loc[:, 'D6_4_1'].
                                values.reshape(-1, 1))).fit()
print(model1.summary())

# %% [markdown]
# **COMMENT**:<br />
# Statistical evidence means between *D4_4_1* and *D6_4_1*. <br />
# Increments of function reduce frequency values,
# which translate that the apps is used more (and viceversa)

# %% [markdown]
# **STEP** **2**: D6_4_1 on V2

# %%
# Check mediator effect

logit2 = sm.Logit(V2, sm.add_constant(Apps_clean.
                                      loc[:, 'D6_4_1'].
                                      values.reshape(-1, 1))).fit()
# 1 if ith apps is hedonic
# 0 if ith apps is utilitarian
print(logit2.summary())

# %% [markdown]
# **COMMENT**: <br />
# With a $p-value = 0,119$ we fail to reject the null hypothesis.
# Since *D6_4_1* is not significant, it cannot mediate anything.

# %% [markdown]
# #### MODEL 2a
# **STEP** **1**: V6_1, V12_1, V2, V3 ---> V4_1

# %%
del X
V3 = V3_enc.transform(Apps_clean['V3'].values.reshape(-1, 1)).toarray()
X = np.hstack([Apps_clean.loc[:, 'V6_1_Interview'].values.reshape(-1, 1),
              Apps_clean.loc[:, 'V12_1_Interview'].values.reshape(-1, 1),
              V2, V3])

model1 = sm.OLS(Apps_clean.loc[:, 'V4_1_Interview'],
                sm.add_constant(X)).fit()
print(model1.summary())

# %% [markdown]
# **COMMENT**: <br />
# Coefs are all statisticaly significant.
# *V3* lower significance (information already covered by V2).

# %% [markdown]
# **STEP** **2**: mediation effects. <br />
# Reference Imai, Keele, Tingley (2010)

# %%
Apps_clean['V3'] = V3_enc.transform(Apps_clean['V3'].
                                    values.reshape(-1, 1)).toarray()

# V6*V2
# Regression model for the outcome
outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview + \
                                     V12_1_Interview + V3 + V2',
                                    data=Apps_clean)

# Regression model for the mediator variable
mediator_model = sm.GLM.from_formula('V2 ~ V6_1_Interview + \
                                       V12_1_Interview + V3',
                                     data=Apps_clean,
                                     family=sm.families.Binomial())

med = Mediation(outcome_model, mediator_model, 'V6_1_Interview', 'V2')

med_result = med.fit()

med_result.summary()

# %% [markdown]
# **COMMENT**: <br />
# *ACME* (Average Causal Mediation Effect). <br />
# *ADE* (Average Direct Effect). <br />
# The results above demonstrate that the ACME is not statistically distinct
# from zero, or no mediation. The average direct effect (ADE) is negative
# and statistically notable. The total effect is statistically notable and it
# is driven by ADE.

# %%
# V12 * V2
# Regression model for the outcome
outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview + \
                                     V12_1_Interview + V3 + V2',
                                    data=Apps_clean)

# Regression model for the mediator variable
mediator_model = sm.GLM.from_formula('V2 ~ V6_1_Interview + \
                                       V12_1_Interview + V3',
                                     data=Apps_clean,
                                     family=sm.families.Binomial())

med = Mediation(outcome_model, mediator_model, 'V12_1_Interview', 'V2')
med_result = med.fit()
med_result.summary()

# %% [markdown]
# **COMMENT**: <br />
# Only direct effect is statistically significant

# %%
# V6 * V3
# Regression model for the outcome
outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview + \
                                     V12_1_Interview + V3 + V2',
                                    data=Apps_clean)

# Regression model for the mediator variable
mediator_model = sm.GLM.from_formula('V3 ~ V6_1_Interview + \
                                       V12_1_Interview + V2',
                                     data=Apps_clean,
                                     family=sm.families.Binomial())

med = Mediation(outcome_model, mediator_model, 'V12_1_Interview', 'V3')
med_result = med.fit()
med_result.summary()

# %% [markdown]
# **COMMENT**: <br />
# Only direct effect is significant

# %%
# V12 * V3
# Regression model for the outcome
outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview + \
                                     V12_1_Interview + V3 + V2',
                                    data=Apps_clean)

# Regression model for the mediator variable
mediator_model = sm.GLM.from_formula('V3 ~ V6_1_Interview + \
                                       V12_1_Interview + V2',
                                     data=Apps_clean,
                                     family=sm.families.Binomial())

med = Mediation(outcome_model, mediator_model, 'V12_1_Interview', 'V3')
med_result = med.fit()
med_result.summary()

# %% [markdown]
# **COMMENT**: <br />
# Only Average Direct Effect statistically significant

# %%
# V2 * V3
# Regression model for the outcome
outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview + \
                                     V12_1_Interview + V3 + V2',
                                    data=Apps_clean)

# Regression model for the mediator variable
mediator_model = sm.GLM.from_formula('V3 ~ V6_1_Interview + \
                                       V12_1_Interview + V2',
                                     data=Apps_clean,
                                     family=sm.families.Binomial())

med = Mediation(outcome_model, mediator_model, 'V2', 'V3')
med_result = med.fit()
med_result.summary()

# %% [markdown]
# **COMMENT**: <br />
# With a $p-value = 0.014$ also the Average Causal Mediation Effect is
# statistically significant.

# %%
# V6 * V12
# Regression model for the outcome
outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview + \
                                     V12_1_Interview + V3 + V2',
                                    data=Apps_clean)

# Regression model for the mediator variable
mediator_model = sm.OLS.from_formula('V12_1_Interview ~ V6_1_Interview + \
                                       V3 + V2',
                                     data=Apps_clean)

med = Mediation(outcome_model, mediator_model, 'V6_1_Interview',
                'V12_1_Interview')
med_result = med.fit()
med_result.summary()

# %% [markdown]
# **COMMENT**:<br />
# With a $p-value=0.0$ the Average Direct Effect and the Average Causal
# Mediation Effect are both statistically significant

# %% [markdown]
# #### MODEL 2b: A modereted mediation analysis<br />

# %%
# Add novelty seeking.
clean_NS = pd.merge(Apps_clean, Personality.loc[:, ['Probanden_ID__lfdn',
                                                    'NovSeek']],
                    on='Probanden_ID__lfdn')

# NovSeek avereage between [1--, 10++]
pd.DataFrame.describe(clean_NS['NovSeek'])

# %%
# The midiation effect is computed for people with high novelty seeking
# V6*V2
# Regression model for the outcome
outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview*NovSeek \
                                     + V12_1_Interview + V3 + V2*NovSeek',
                                    data=clean_NS)

# Regression model for the mediator variable
mediator_model = sm.GLM.from_formula('V2 ~ V6_1_Interview*NovSeek + \
                                       V12_1_Interview + V3',
                                     data=clean_NS,
                                     family=sm.families.Binomial())
moderators = {'NovSeek': 10}
med = Mediation(outcome_model, mediator_model, 'V6_1_Interview', 'V2',
                moderators=moderators).fit()
med.summary()

# %% [markdown]
# **COMMENT**: <br />
# Only direct effect is significant

# %%
outcome_model = sm.OLS.from_formula('V4_1_Interview ~ V6_1_Interview + \
                                     V12_1_Interview + V3 + V2 + NovSeek',
                                    data=clean_NS)

# Regression model for the mediator variable
mediator_model = sm.OLS.from_formula('NovSeek ~ V6_1_Interview + \
                                       V12_1_Interview + V3 + V2',
                                     data=clean_NS)

med = Mediation(outcome_model, mediator_model, 'V6_1_Interview',
                'NovSeek').fit()
med.summary()

# %% [markdown]
# **COMMENT**: <br />
# Only direct effect is significant.

# %% [Markdown]
# #### Missing Values Analysis

# %%
Apps_missing = Apps.drop(list(Apps_clean.index))

# Difference in V6 for the missing and clean dataset
V6_clean_1 = pd.DataFrame.describe(Apps_clean['V6_1_Interview'])
V6_miss_1 = pd.DataFrame.describe(Apps_missing['V6_1_Interview'])

fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 13))
fig.suptitle('V6 (frequency) within clean and missing observations',
             fontsize=15, fontweight='bold')
sns.boxplot(Apps_clean['V6_1_Interview'], ax=axs[0])
axs[0].set_title('{}'.format(V6_clean_1), fontsize=10)
sns.boxplot(Apps_missing['V6_1_Interview'], ax=axs[1])
axs[1].set_title('{}'.format(V6_miss_1), fontsize=10)
fig.subplots_adjust(top=0.85, hspace=0.8)
plt.show()

# %%
# Difference in V12 for the missing and clean dataset
V12_clean_1 = pd.DataFrame.describe(Apps_clean['V12_1_Interview'])
V12_miss_1 = pd.DataFrame.describe(Apps_missing['V12_1_Interview'])

fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 13))
fig.suptitle('V12 (satisfaction) within clean and missing observations',
             fontsize=15, fontweight='bold')
sns.boxplot(Apps_clean['V12_1_Interview'], ax=axs[0])
axs[0].set_title('{}'.format(V12_clean_1), fontsize=10)
sns.boxplot(Apps_missing['V12_1_Interview'], ax=axs[1])
axs[1].set_title('{}'.format(V12_miss_1), fontsize=10)
fig.subplots_adjust(top=0.84, hspace=0.5)
plt.show()

# %% [markdown]
# **COMMENT**: <br />
# No difference between the two datasets.

# %% [Markdown]
# #### Missing Values Analysis

# %%
Apps_missing = Apps.drop(list(Apps_clean.index))

# Difference in V6 for the missing and clean dataset
V6_clean_1 = pd.DataFrame.describe(Apps_clean['V6_1_Interview'])
V6_miss_1 = pd.DataFrame.describe(Apps_missing['V6_1_Interview'])

fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 13))
fig.suptitle('V6 (frequency) within clean and missing observations',
             fontsize=15, fontweight='bold')
sns.boxplot(Apps_clean['V6_1_Interview'], ax=axs[0])
axs[0].set_title('{}'.format(V6_clean_1), fontsize=10)
sns.boxplot(Apps_missing['V6_1_Interview'], ax=axs[1])
axs[1].set_title('{}'.format(V6_miss_1), fontsize=10)
fig.subplots_adjust(top=0.85, hspace=0.8)
plt.show()

# %%
# Difference in V12 for the missing and clean dataset
V12_clean_1 = pd.DataFrame.describe(Apps_clean['V12_1_Interview'])
V12_miss_1 = pd.DataFrame.describe(Apps_missing['V12_1_Interview'])

fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 13))
fig.suptitle('V12 (satisfaction) within clean and missing observations',
             fontsize=15, fontweight='bold')
sns.boxplot(Apps_clean['V12_1_Interview'], ax=axs[0])
axs[0].set_title('{}'.format(V12_clean_1), fontsize=10)
sns.boxplot(Apps_missing['V12_1_Interview'], ax=axs[1])
axs[1].set_title('{}'.format(V12_miss_1), fontsize=10)
fig.subplots_adjust(top=0.84, hspace=0.5)
plt.show()

# %% [markdown]
# **COMMENT**: <br />
# No difference between the two datasets.


# MAJOR VARIBLES CLEANING Interactive python
'''
%reset_selective -f outbound
%reset_selective -f descriptive
%reset_selective -f col
%reset_selective -f typo
%reset_selective -f enc
%reset_selective -f ax
%reset_selective -f row
%reset_selective -f V
'''
# %reset_selective -f outbound
# %reset_selective -f descriptive
# %reset_selective -f col
# %reset_selective -f typo
# %reset_selective -f enc
# %reset_selective -f ax
# %reset_selective -f row
# %reset_selective -f V

# NUMBER OF COMPLETE ANSWERS FOR V12 AND V6 ACROSS INTERVIEWS
Apps_clean.shape[0]  # 2977

# V2
pd.value_counts(Apps_clean['V2'])
# V3
pd.value_counts(Apps_clean['V3'])
# Description V6
pd.DataFrame.describe(Apps_clean.loc[:, ['V6_1_Interview', 'V6_2_Interview',
                                         'V6_3_Interview', 'V6_4_Interview']])
# Description V12
pd.DataFrame.describe(Apps_clean.loc[:, ['V12_1_Interview', 'V12_2_Interview',
                                         'V12_3_Interview',
                                         'V12_4_Interview']])

# V2 differences in frequency between utilitarian and hedonic v4
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

# In general: frequency for V2, V3 [1, 2, 3 missing]
Apps_missing[['V2', 'V3']] = Apps_missing[['V2', 'V3']].fillna('missing')
# V2 missing dataset
pd.value_counts(Apps_missing['V2'])
# V3 missing dataset
pd.value_counts(Apps_missing['V3'])

# Detect groups with 1, 2, 3 non missing for V4, V6, V12

Apps_missing['V4_Not_Miss'] = Apps_missing.loc[:, ['V4_1_Interview',
                                                   'V4_2_Interview',
                                                   'V4_3_Interview',
                                                   'V4_4_Interview']
                                               ].apply(lambda x:
                                                       x.count(),
                                                       axis=1)
'''
4 > no missing
3 > 1 missing
2 > 2 missing
1 > 3 missing
0 > all missing
'''
pd.value_counts(Apps_missing['V4_Not_Miss'])

# count consecutive
# for 1 > 3 missing = 90 count number on the first row should be 90
Apps_missing[Apps_missing['V4_Not_Miss'] == 1].loc[:,  # all consecutive
                                                   'V4_1_Interview'].count()

# they are all consecutive

# for 2 > 2 missing = 88 count number on the first 2 rows should be 88
# add a new columns which count the number of values on the first two
# interviews
# axis: 0 or ‘index’: apply function to each column.
# axis: 1 or ‘columns’: apply function to each row.

Apps_missing['V4_1_2_count'
             ] = Apps_missing[Apps_missing['V4_Not_Miss'] == 2
                              ].loc[:, ['V4_1_Interview',
                                        'V4_2_Interview']
                                    ].apply(lambda x:
                                            x.count(),
                                            axis=1)
# delete the non consecutive ones (5 rows)
pd.value_counts(Apps_missing['V4_1_2_count'])

# for 3 > 1 missing = 273 count number on the first 3 rows should be 273
# 167 contiguos
Apps_missing['V4_1_2_3_count'
             ] = Apps_missing[Apps_missing['V4_Not_Miss'] == 3
                              ].loc[:, ['V4_1_Interview',
                                        'V4_2_Interview',
                                        'V4_3_Interview']
                                    ].apply(lambda x:
                                            x.count(),
                                            axis=1)
pd.value_counts(Apps_missing['V4_1_2_3_count'])

Apps_missing['V6_Not_Miss'] = Apps_missing.loc[:, ['V6_1_Interview',
                                                   'V6_2_Interview',
                                                   'V6_3_Interview',
                                                   'V6_4_Interview']
                                               ].apply(lambda x:
                                                       x.count(),
                                                       axis=1)
pd.value_counts(Apps_missing['V6_Not_Miss'])

# for 1 > 3 missing = 194 count number on the first row should be 194
# all consecutive
Apps_missing[Apps_missing['V6_Not_Miss'] == 1].loc[:, 'V6_1_Interview'].count()

Apps_missing['V6_1_2_count'
             ] = Apps_missing[Apps_missing['V6_Not_Miss'] == 2
                              ].loc[:, ['V6_1_Interview',
                                        'V6_2_Interview']
                                    ].apply(lambda x:
                                            x.count(),
                                            axis=1)
pd.value_counts(Apps_missing['V6_1_2_count'])

Apps_missing['V6_1_2_3_count'
             ] = Apps_missing[Apps_missing['V6_Not_Miss'] == 3
                              ].loc[:, ['V6_1_Interview',
                                        'V6_2_Interview',
                                        'V6_3_Interview']
                                    ].apply(lambda x:
                                            x.count(),
                                            axis=1)
pd.value_counts(Apps_missing['V6_1_2_3_count'])

# V12
Apps_missing['V12_Not_Miss'] = Apps_missing.loc[:, ['V12_1_Interview',
                                                    'V12_2_Interview',
                                                    'V12_3_Interview',
                                                    'V12_4_Interview']
                                                ].apply(lambda x:
                                                        x.count(),
                                                        axis=1)
pd.value_counts(Apps_missing['V12_Not_Miss'])

# for 1 > 3 missing = 171 count number on the first row should be 171
Apps_missing[Apps_missing['V12_Not_Miss'] == 1].loc[:,
                                                    'V12_1_Interview'].count()

Apps_missing['V12_1_2_count'
             ] = Apps_missing[Apps_missing['V12_Not_Miss'] == 2
                              ].loc[:, ['V12_1_Interview',
                                        'V12_2_Interview']
                                    ].apply(lambda x:
                                            x.count(),
                                            axis=1)
pd.value_counts(Apps_missing['V12_1_2_count'])

Apps_missing['V12_1_2_3_count'
             ] = Apps_missing[Apps_missing['V12_Not_Miss'] == 3
                              ].loc[:, ['V12_1_Interview',
                                        'V12_2_Interview',
                                        'V12_3_Interview']
                                    ].apply(lambda x:
                                            x.count(),
                                            axis=1)
pd.value_counts(Apps_missing['V12_1_2_3_count'])

# Group 1:
group_1 = Apps_missing[(Apps_missing['V2'] != 'missing') &  # V2 not missing
                       (Apps_missing['V3'] != 'missing') &  # V3 not missing
                       (Apps_missing['V4_Not_Miss'] == 1) &  # V4 not missing=1
                       (Apps_missing['V6_Not_Miss'] == 1) &
                       # 1st interview not missing
                       (pd.notna(Apps_missing['V6_1_Interview'])) &
                       (Apps_missing['V12_Not_Miss'] == 1) &
                       (pd.notna(Apps_missing['V12_1_Interview']))]
# Group 2:
group_2 = Apps_missing[(Apps_missing['V2'] != 'missing') &
                       (Apps_missing['V3'] != 'missing') &
                       # V4 has 2 consecutive
                       (Apps_missing['V4_1_2_count'] == 2) &
                       # V6 has 2 consecutive
                       (Apps_missing['V6_1_2_count'] == 2) &
                       # V12 has 2 consecutive
                       (Apps_missing['V12_1_2_count'] == 2)]

# Group 3:
group_3 = Apps_missing[(Apps_missing['V2'] != 'missing') &
                       (Apps_missing['V3'] != 'missing') &
                       # V4 with 3 consecutive
                       (Apps_missing['V4_1_2_3_count'] == 3) &
                       (Apps_missing['V6_1_2_3_count'] == 3) &
                       (Apps_missing['V12_1_2_3_count'] == 3)]

# For all
fig, ax = plt.subplots()
# title, number and legend
sns.countplot(x='V2', hue='V3', data=Apps_clean, ax=ax)
plt.show()

# General for missing
fig, ax = plt.subplots()
sns.countplot(x='V2', hue='V3', data=Apps_missing, ax=ax)
plt.show()

# V12 Satisfaction distribution over groups and interviews
fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15, 10))
fig.suptitle('V12 Satisfaction [1--, 10++] -\
 Univariate Distribution across Groups',
             fontsize=15, fontweight='bold')
axs[0].set_title('First Interview')
sns.kdeplot(Apps_clean['V12_1_Interview'], shade=True, ax=axs[0],
            label='Group 4')
sns.kdeplot(group_3['V12_1_Interview'], shade=True, ax=axs[0],
            label='Group 3')
sns.kdeplot(group_2['V12_1_Interview'], shade=True, ax=axs[0],
            label='Group 2')
sns.kdeplot(group_1['V12_1_Interview'], shade=True, ax=axs[0],
            label='Group 1')
axs[1].set_title('Second Interview')
sns.kdeplot(Apps_clean['V12_2_Interview'], shade=True, ax=axs[1],
            label='Group 4')
sns.kdeplot(group_3['V12_2_Interview'], shade=True, ax=axs[1],
            label='Group 3')
sns.kdeplot(group_2['V12_2_Interview'], shade=True, ax=axs[1],
            label='Group 2')
axs[2].set_title('Third Interview')
sns.kdeplot(Apps_clean['V12_3_Interview'], shade=True, ax=axs[2],
            label='Group 4')
sns.kdeplot(group_3['V12_3_Interview'], shade=True, ax=axs[2],
            label='Group 3')
axs[3].set_title('Third Interview')
sns.kdeplot(Apps_clean['V12_4_Interview'], shade=True, ax=axs[3],
            label='Group 4')
fig.subplots_adjust(hspace=0.8)
plt.legend()
plt.show()

# COMMENT:
# Group 1 has not difference
# Group 2 has more values at the lower scale (< 5)
# there is not coherence between the groups, it is really hard to make a
# conclusions

# V6 number of functions distribution over groups and interviews
fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15, 10))
fig.suptitle('V6 # Functions [1--, 10++] -\
 Univariate Distribution across Groups',
             fontsize=15, fontweight='bold')
axs[0].set_title('First Interview')
sns.kdeplot(Apps_clean['V6_1_Interview'], shade=True, ax=axs[0],
            label='Group 4')
sns.kdeplot(group_3['V6_1_Interview'], shade=True, ax=axs[0],
            label='Group 3')
sns.kdeplot(group_2['V6_1_Interview'], shade=True, ax=axs[0],
            label='Group 2')
sns.kdeplot(group_1['V6_1_Interview'], shade=True, ax=axs[0],
            label='Group 1')
axs[1].set_title('Second Interview')
sns.kdeplot(Apps_clean['V6_2_Interview'], shade=True, ax=axs[1],
            label='Group 4')
sns.kdeplot(group_3['V6_2_Interview'], shade=True, ax=axs[1],
            label='Group 3')
sns.kdeplot(group_2['V6_2_Interview'], shade=True, ax=axs[1],
            label='Group 2')
axs[2].set_title('Third Interview')
sns.kdeplot(Apps_clean['V6_3_Interview'], shade=True, ax=axs[2],
            label='Group 4')
sns.kdeplot(group_3['V6_3_Interview'], shade=True, ax=axs[2],
            label='Group 3')
axs[3].set_title('Third Interview')
sns.kdeplot(Apps_clean['V6_4_Interview'], shade=True, ax=axs[3],
            label='Group 4')
fig.subplots_adjust(hspace=0.8)
plt.legend()
plt.show()

# COMMENT
# Number of functionalities group 2 higher picks on high values
# group 1 picks around 7.5
# there is not a real trend on the data...hight number of functions
# brings at some drop out.

# V2 differences in frequency between utilitarian and hedonic v4
# GROUP 1
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True,
                        sharey=True, figsize=(15, 10))
fig.suptitle('GROUP 1: V2 (Utilitarian vs. Hedonic) & \
                V4 (Frequncy [1++, 7--])',
             fontsize=15, fontweight='bold')
sns.boxplot(x='V2', y='V4_1_Interview', data=group_1,
            ax=axs[0])
axs[0].set_title('First Interview')
sns.boxplot(x='V2', y='V4_2_Interview', data=group_1,
            ax=axs[1])
axs[1].set_title('Second Interview')
sns.boxplot(x='V2', y='V4_3_Interview', data=group_1,
            ax=axs[2])
axs[2].set_title('Third Interview')
sns.boxplot(x='V2', y='V4_4_Interview', data=group_1,
            ax=axs[3])
axs[3].set_title('Forth Interview')
fig.subplots_adjust(hspace=0.8)
plt.show()

# V2 differences in frequency between utilitarian and hedonic v4
# GROUP 2
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True,
                        sharey=True, figsize=(15, 10))
fig.suptitle('GROUP 2: V2 (Utilitarian vs. Hedonic) & \
                V4 (Frequncy [1++, 7--])',
             fontsize=15, fontweight='bold')
sns.boxplot(x='V2', y='V4_1_Interview', data=group_2,
            ax=axs[0])
axs[0].set_title('First Interview')
sns.boxplot(x='V2', y='V4_2_Interview', data=group_2,
            ax=axs[1])
axs[1].set_title('Second Interview')
sns.boxplot(x='V2', y='V4_3_Interview', data=group_2,
            ax=axs[2])
axs[2].set_title('Third Interview')
sns.boxplot(x='V2', y='V4_4_Interview', data=group_2,
            ax=axs[3])
axs[3].set_title('Forth Interview')
fig.subplots_adjust(hspace=0.8)
plt.show()

# COMMENT
# Duringthe second interview decrese in used frequency

# V2 differences in frequency between utilitarian and hedonic v4
# GROUP 3
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True,
                        sharey=True, figsize=(15, 10))
fig.suptitle('GROUP 3: V2 (Utilitarian vs. Hedonic) & \
                V4 (Frequncy [1++, 7--])',
             fontsize=15, fontweight='bold')
sns.boxplot(x='V2', y='V4_1_Interview', data=group_3,
            ax=axs[0])
axs[0].set_title('First Interview')
sns.boxplot(x='V2', y='V4_2_Interview', data=group_3,
            ax=axs[1])
axs[1].set_title('Second Interview')
sns.boxplot(x='V2', y='V4_3_Interview', data=group_3,
            ax=axs[2])
axs[2].set_title('Third Interview')
sns.boxplot(x='V2', y='V4_4_Interview', data=group_3,
            ax=axs[3])
axs[3].set_title('Forth Interview')
fig.subplots_adjust(hspace=0.8)
plt.show()

# COMMENT
# Duringthe second interview decrese in used frequency

# ANOVA CALCULATION
# Create a dataset with GROUPS, V2, V4, V6, V12, V17, V18
# 1. Check if V17 and V18 are relevant
'''
# for group 1 > V17 2, 3, 4 are all missing
# for group 2 > V17 only 2 with 0 missing
#             > V18 only 2 with 1 missing
# for group 3 > V18 only 3 with 1 missing
#              > 17 only 3 with 1 missing
'''

# keep keys
group_4 = Apps_clean.drop(['Datum_1_Interview', 'Datum_2_Interview',
                           'Datum_3_Interview',
                           'Datum_4_Interview', 'V1', 'V3', 'V10', 'V11',
                           'V13', 'V14',
                           'V19_2_Interview', 'V19_3_Interview',
                           'V19_4_Interview',
                           'V20_2_Interview', 'V20_3_Interview',
                           'V20_4_Interview',
                           'V21_2_Interview', 'V21_3_Interview',
                           'V21_4_Interview',
                           'Miss_row_%', 'Days_Between_2_and_1_Interview',
                           'Days_Between_3_and_2_Interview',
                           'Days_Between_4_and_3_Interview', 'D4_2_1',
                           'D4_3_2', 'D4_4_3',
                           'D4_4_1', 'D6_2_1', 'D6_3_2', 'D6_4_3', 'D6_4_1',
                           'D12_2_1', 'D12_3_2', 'D12_4_3', 'D12_4_1', 'V01'],
                          axis=1)

group_4['Group'] = 4

for num, i in enumerate([group_1, group_2, group_3]):
    i.drop(['Datum_1_Interview', 'Datum_2_Interview',
            'Datum_3_Interview',
            'Datum_4_Interview', 'V1', 'V3', 'V10', 'V11',
            'V13', 'V14',
            'V19_2_Interview', 'V19_3_Interview',
            'V19_4_Interview',
            'V20_2_Interview', 'V20_3_Interview',
            'V20_4_Interview',
            'V21_2_Interview', 'V21_3_Interview',
            'V21_4_Interview',
            'Miss_row_%', 'Days_Between_2_and_1_Interview',
            'Days_Between_3_and_2_Interview',
            'Days_Between_4_and_3_Interview', 'V4_Not_Miss',
            'V4_1_2_count', 'V4_1_2_3_count', 'V6_Not_Miss',
            'V6_1_2_count', 'V6_1_2_3_count',
            'V12_Not_Miss', 'V12_1_2_count', 'V12_1_2_3_count',
            'V01'],
           axis=1, inplace=True)

    i['Group'] = num + 1

# reduce cumulatively compute a function on a list and return the result
dataframe_list = [group_1, group_2, group_3, group_4]
groups = reduce(lambda left, right: pd.DataFrame.append(left, right),
                dataframe_list)

# Getting summary statistics

# G4 vs G1 for V6_1_Interview
G4_G1 = groups.loc[(groups['Group'] == 4) | (groups['Group'] == 1), :]
rp.summary_cont(G4_G1['V6_1_Interview'].groupby(G4_G1['Group']))

# ANOVA with scipy.stats
stats.f_oneway(G4_G1['V6_1_Interview'][G4_G1['Group'] == 1],
               G4_G1['V6_1_Interview'][G4_G1['Group'] == 4])

# COMMENT
# F-statistic= 1.16239 and p-value=0,281. There is not significant
# effect of group on V6.

# Groups are not with the same size
# More robust test is the non-parametric Kruskal-Wallis H-test (1952)

stats.kruskal(G4_G1['V6_1_Interview'][G4_G1['Group'] == 1],
              G4_G1['V6_1_Interview'][G4_G1['Group'] == 4])


# COMMENT
# KruskalResult = 0.37171 and p-value = 0.5420 There is not significant
# effect of groups 1 and 4 on V6.

# ANOVA with statsmodels
'''
# Let's Dummy the groups
group_enc = OneHotEncoder(categories='auto', drop='first')
# not include 1 of the groups. The group 1 will get captured in the
# model's intercept and is the base (control) group.
group_enc = group_enc.fit(G4_G1['Group'].values.reshape(-1, 1))
group_enc.categories_
# 0 if ith app is in Group 1
# 1 if ith app is in Group 4

G4_G1['Group'] = group_enc.transform(G4_G1['Group'
                                          ].values.reshape(-1, 1)).toarray()

G4_G1['Group'] = group_enc.inverse_transform(G4_G1['Group'
                                                  ].values.reshape(-1, 1))
'''

# 'C' takes care of creating a dummy. The group 1 will get captured in the
# model's intercept and is the base (control) group.
results = ols('V6_1_Interview ~ C(Group)', data=G4_G1).fit()
results.summary()

# COMMENT:
# The intercept group is group 1. At he bottom there are tests to check model
# assumptions.
# F-stats = 1.162 and p=0.281. This tells us that there is not a significant
# differences in the group means. The coefs, are the difference in mean between
# the control group and the respective group listed.
# The intercept is the mean for the group 1 and 6.88-0.33 = 6.55 is the mean
# for group 2.
# To test betwen groups, we need to do some post-hoc testing where we compare
# all groups
# against each other.

# ANOVA TABLE
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table

# COOMENT
# the C(Group) ro is the between groups effect which is the overall
# experimental effect. The sum of squares = 8.40 is how much variance is
# explained by our model. The current model does not exaplain a significant
# amount with p > 0.05. The residual row is the unsystematic variation in
# the data = 22056 unexplained variance.

# G4 vs G1 for V2
# Two-Way tables or contingency tables
group_V2 = pd.crosstab(index=G4_G1['Group'],
                       columns=G4_G1['V2'])
group_V2

# Calculate the Chi2 for the contingency table
obs = np.array([[1, 43, 33], [4, 1560, 1417]])
chi2, p_value, dof, exp = stats.chi2_contingency(obs)
p_value

# Create the final csv file
# Apps_clean.to_csv('Apps_clean.csv', index = False)

group_1.to_csv('group_1.csv', index = False)
group_2.to_csv('group_2.csv', index = False)
group_3.to_csv('group_3.csv', index = False)
group_4.to_csv('group_4.csv', index = False)
groups.to_csv('groups.csv', index = False)
# Apps.to_csv('Apps.csv', index = False)
