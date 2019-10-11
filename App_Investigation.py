# %% [markdown]
# # Apps Analysis

# %%
# Load library
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

# %%
# Define a new workspace and import excel files
Data_1 = pd.read_excel(r'./Data_1.xlsx')
Data_2 = pd.read_excel(r'./Data_2.xlsx')
Data_3 = pd.read_excel(r'./Data_3.xlsx')
Data_4 = pd.read_excel(r'./Data_4.xlsx')
Personality = pd.read_excel(r'./Part_Personality.xlsx')

# %% [markdown]
# ## 1. Data description:

# %% [markdown]
# ### 1.1 Data breakdown structure:
# PDF

# %% [markdown]
# ## 2. Data manipulation:
#
# ### 2.1 Merge datasets
# %% [markdown]
# #### Create a new ID column common to all datasets:

# %%
# Create a new participant ID column (made of Probanden_ID__lfdn)
# Useful to merge with Personality
for i in [Data_1, Data_2, Data_3, Data_4, Personality]:
        # concatenate 'Probanden_ID' and 'lfdn' to create a new unique variable
        # which identify the combination participant-app
        i['Probanden_ID__lfdn'] = i.Probanden_ID.astype(str) + '__' + \
                                  i.lfdn.astype(str)

# %% [markdown]
# #### Create a new ID column common to Data_1, Data_2, Data_3, Data_4:

# %%
# Merge the 4 datasets base on Probanden_ID__lfdn__AppNr
# There are persons with the same initials and progressive number
for i in [Data_1, Data_2, Data_3, Data_4]:
        # concatenate 'Probanden_ID' and 'AppNr' to create a new variable
        # which identify uniquelly the combination participant-app
        i['Probanden_ID__lfdn__AppNr'] = i.Probanden_ID.astype(str) + '__' + \
                                         i.lfdn.astype(str) + '__' + \
                                         i.AppNr.astype(str)
        # delete redundant columns
        i.drop(['AppNr', 'lfdn', 'Probanden_ID'], axis=1, inplace=True)

# %%
# Check if V1 is the same in all 4 datasets for the combination
# participant + app

# reduce cumulatively  compute a function on a list and return the result
dataframe_list = [Data_1[['Probanden_ID__lfdn__AppNr', 'V1']],
                  Data_2[['Probanden_ID__lfdn__AppNr', 'V1']],
                  Data_3[['Probanden_ID__lfdn__AppNr', 'V1']],
                  Data_4[['Probanden_ID__lfdn__AppNr', 'V1']]]

V1_merge = reduce(lambda left, right: pd.merge(left, right,
                                               on='Probanden_ID__lfdn__AppNr'),
                  dataframe_list)

# check all columns against the first column using eq
# if they are all equal to the first column than is True
V1_merge = V1_merge.fillna('-')
sum(V1_merge.filter(like='V1').eq(V1_merge.iloc[:, 1], axis=0).all(axis=1))

# %%
# drop V1 from other datasets
for i in [Data_2, Data_3, Data_4]:
        i.drop('V1', axis=1, inplace=True)

# %%
# rename cols before merging
for num, data in enumerate([Data_1, Data_2, Data_3, Data_4]):
        for col_name in ['Datum', 'Interviewer', 'V4', 'V5', 'V6', 'V12',
                         'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23']:
                if any(data.columns == col_name):
                        data.rename(columns={col_name: col_name +
                                             '_' + str(num + 1) +
                                             '_Interview'}, inplace=True)

# %%
# merge datasets
for i in [Data_2, Data_3, Data_4]:
        i.drop('Probanden_ID__lfdn', axis=1, inplace=True)

dataframe_list = [Data_1, Data_2, Data_3, Data_4]

Apps = reduce(lambda left, right: pd.merge(left, right,
                                           on='Probanden_ID__lfdn__AppNr'),
              dataframe_list)

# reorder columns
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

# Apps
Apps.head()
# %%
# Number of (rows, columns) in Apps
Apps.shape

# %% [markdown]
# ### 2.2 Clean and format datasets
# Calculate the percentage of missing for each row
# delete the row with high number of missing values(>80%)
# First screening
values_over_rows = Apps.apply(lambda x: x.count(), axis=1)
Apps['Miss_row_%'] = 100 - values_over_rows/len(Apps.columns)*100

Apps = Apps[Apps['Miss_row_%'] < 80]
plt.hist(Apps['Miss_row_%'])
# Calculate the percentage of missing for each columns
# delete the row with high number of missing values

# First screening
miss_cols_pct = pd.DataFrame(100-(Apps.apply(lambda x: x.count(), axis=0) /
                                  Apps.shape[0])*100)

col_del = list(miss_cols_pct[miss_cols_pct[0] > 80].index)
Apps.drop(col_del, axis=1, inplace=True)

# %%
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
                                             int(Apps.loc
                                                 [row, 'Datum_1_Interview']) -
                                             2)
        elif pd.isna(Apps.loc[row, 'Datum_1_Interview']):
                pass
        else:  # do nothing
                typo_row_index.append(row)
                typo_date.append(Apps.loc[row, 'Datum_1_Interview'])

# Adjust for typos
np.unique(typo_date)
for row in typo_row_index:
        Apps.loc[row, 'Datum_1_Interview'] = datetime(2017, 12, 20)

# Adjst for date_2_interview
typo_row_index = []
typo_date = []

for row in list(Apps.index):
        # convert date on the format '03.01.2018'
        if len(str(Apps.loc[row, 'Datum_2_Interview'])) == 10:
                Apps.loc[row, 'Datum_2_Interview'] = \
                        datetime.strptime(Apps.loc[row, 'Datum_2_Interview'],
                                          '%d.%m.%Y')
        # if already timestamp pass
        elif isinstance(Apps.loc[row, 'Datum_2_Interview'], datetime):
                pass
        elif pd.isna(Apps.loc[row, 'Datum_2_Interview']):
                pass
        else:  # do nothing
                typo_row_index.append(row)
                typo_date.append(Apps.loc[row, 'Datum_2_Interview'])

# Adjst for date_3_interview
typo_row_index = []
typo_date = []

for row in list(Apps.index):
        # convert date on the format '03.01.2018'
        if isinstance(Apps.loc[row, 'Datum_3_Interview'], datetime):
                pass
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
        # if already timestamp pass

        elif pd.isna(Apps.loc[row, 'Datum_3_Interview']):
                pass
        else:  # do nothing
                typo_row_index.append(row)
                typo_date.append(Apps.loc[row, 'Datum_3_Interview'])

# Adjst for date_3_interview
typo_row_index = []
typo_date = []

for row in list(Apps.index):
        # convert date on the format '03.01.2018'
        if isinstance(Apps.loc[row, 'Datum_4_Interview'], datetime):
                pass
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
        # if already timestamp pass

        elif pd.isna(Apps.loc[row, 'Datum_4_Interview']):
                pass
        else:  # do nothing
                typo_row_index.append(row)
                typo_date.append(Apps.loc[row, 'Datum_4_Interview'])


# Check if date are progressive
Apps['Days_Between_2_and_1_Interview'] = Apps['Datum_2_Interview'] - \
                                                Apps['Datum_1_Interview']

# correct the typo with year 2108 in Datum_2_Interview
Apps.loc[Apps['Datum_2_Interview'] > datetime(2020, 1, 1),
         'Datum_2_Interview'] = datetime(2018, 4, 11)

Apps['Days_Between_2_and_1_Interview'] = Apps['Datum_2_Interview'] - \
                                                Apps['Datum_1_Interview']

# Into numbers of days
Apps['Days_Between_2_and_1_Interview'] = \
        Apps['Days_Between_2_and_1_Interview'].apply(lambda x:
                                                     x.total_seconds() /
                                                     60/60/24)


# make a frequency plot of the days between the 2nd and first interview
pd.value_counts(Apps['Days_Between_2_and_1_Interview'])

# # Difference between 3 and 2 interview
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

# Do we have collection of data also during the 2019???
Apps.loc[Apps['Datum_3_Interview'] > datetime(2019, 1, 1),
         'Datum_3_Interview'] = datetime(2018, 5, 17)

Apps['Days_Between_3_and_2_Interview'] = Apps['Datum_3_Interview'] - \
                                                Apps['Datum_2_Interview']

pd.value_counts(Apps['Days_Between_3_and_2_Interview'])

# Into numbers of days
Apps['Days_Between_3_and_2_Interview'] = \
        Apps['Days_Between_3_and_2_Interview'].apply(lambda x:
                                                     x.total_seconds() /
                                                     60/60/24)

# # Difference between 4 and 3 interview
Apps['Days_Between_4_and_3_Interview'] = Apps['Datum_4_Interview'] - \
                                                Apps['Datum_3_Interview']

# Spot typos
pd.unique(Apps.loc[Apps['Datum_4_Interview'] > datetime(2020, 1, 1),
                   'Datum_4_Interview'])


# datetime(3018, 3, 4, 0, 0)
Apps.loc[Apps['Datum_4_Interview'] > datetime(2500, 1, 1),
         'Datum_4_Interview'] = datetime(2018, 3, 4)

Apps['Days_Between_4_and_3_Interview'] = Apps['Datum_4_Interview'] - \
                                                Apps['Datum_3_Interview']

np.unique(Apps.loc[Apps['Datum_4_Interview'] > datetime(2019, 1, 1),
          'Datum_4_Interview'])

Apps.loc[Apps['Datum_4_Interview'] > datetime(2019, 1, 1),
         'Datum_4_Interview'] = datetime(2018, 7, 9)

Apps['Days_Between_4_and_3_Interview'] = Apps['Datum_4_Interview'] - \
                                                Apps['Datum_3_Interview']

pd.value_counts(Apps['Days_Between_4_and_3_Interview'])

# Into numbers of days
Apps['Days_Between_4_and_3_Interview'] = \
        Apps['Days_Between_4_and_3_Interview'].apply(lambda x:
                                                     x.total_seconds() /
                                                     60/60/24)
# %%
# # Plot the different frequency graph and eliminate observations
# with negative number of days between following interview

descriptive_2_1 = pd.DataFrame.describe(Apps['Days_Between_2_and_1_Interview'])
descriptive_3_2 = pd.DataFrame.describe(Apps['Days_Between_3_and_2_Interview'])
descriptive_4_3 = pd.DataFrame.describe(Apps['Days_Between_4_and_3_Interview'])

fig, axs = plt.subplots(3, sharex=True, sharey=True)
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

# KK_5__278 >> 2018 - 2017 - 2017 - 2018
# chenge the middle two dates
Apps.loc[Apps['Days_Between_2_and_1_Interview'] < 0,
         'Days_Between_2_and_1_Interview']

# Ajust negative date
Apps.loc[Apps['Datum_2_Interview'] == datetime(2017, 5, 13),
         'Probanden_ID__lfdn__AppNr']
Apps.loc[Apps['Datum_2_Interview'] == datetime(2017, 5, 13),
         'Datum_2_Interview'] = datetime(2018, 5, 13)
Apps['Days_Between_2_and_1_Interview'] = Apps['Datum_2_Interview'] - \
                                                Apps['Datum_1_Interview']
# Into numbers of days
Apps['Days_Between_2_and_1_Interview'] = \
        Apps['Days_Between_2_and_1_Interview'].apply(lambda x:
                                                     x.total_seconds() /
                                                     60/60/24)

# make assumption >> typos error adjust in a second time

# think what to do do with the missing date
# (assumption there is a progression within the interviews so keep it)

# report the missing values by column
sum(pd.isna(Apps['Datum_4_Interview']))

# Delete Interviewer columns
Apps.drop(['Interviewer_1_Interview', 'Interviewer_2_Interview',
           'Interviewer_3_Interview', 'Interviewer_4_Interview'],
          axis=1, inplace=True)

# Delete rows with both V1 and V01 missing >> no apps listed
# since V01 can be create base on value in V1

V1_V01_miss_idx = list(Apps.loc[(Apps['V1'].isna()) &
                       (Apps['V01'].isna())].index)

Apps.drop(V1_V01_miss_idx, axis=0, inplace=True)

# check V2 frequency and missing
pd.value_counts(Apps['V2'])

# eliminate 3,4, 6 values
V2_outbound_idx = list(Apps.loc[Apps['V2'].isin([3, 4, 6])].index)
Apps.drop(V2_outbound_idx, axis=0, inplace=True)

sum(Apps['V2'].isna())
# # Think what to do with the missing
# Ignore missing models, and graphs will delete the intire row
# TODO

# V3 check missing and frequency
pd.value_counts(Apps['V3'])
V3_outbound_idx = list(Apps.loc[Apps['V3'] == 5].index)
Apps.drop(V3_outbound_idx, axis=0, inplace=True)

# First strategy delete all missing values and
# check the number of observations left
# TODO
sum(Apps['V3'].isna())

# V4_1_Interview check missing and frequency (1++ - 7--)
pd.value_counts(Apps['V4_1_Interview'])
V4_1_outbound_idx = list(Apps.loc[Apps['V4_1_Interview'] == 11].index)
Apps.drop(V4_1_outbound_idx, axis=0, inplace=True)
sum(pd.isna(Apps['V4_1_Interview']))

# V4_2_Interview check missing and frequency (1++ - 7--)
pd.value_counts(Apps['V4_2_Interview'])
V4_2_outbound_idx = list(Apps.loc[(Apps['V4_2_Interview'] > 7)].index)
Apps.drop(V4_2_outbound_idx, axis=0, inplace=True)
sum(pd.isna(Apps['V4_2_Interview']))

# V4_3_Interview check missing and frequency (1++ - 7--)
pd.value_counts(Apps['V4_3_Interview'])
V4_3_outbound_idx = list(Apps.loc[(Apps['V4_3_Interview'] > 7)].index)
Apps.drop(V4_3_outbound_idx, axis=0, inplace=True)
sum(pd.isna(Apps['V4_3_Interview']))

# V4_4_Interview check missing and frequency (1++ - 7--)
pd.value_counts(Apps['V4_4_Interview'])
V4_4_outbound_idx = list(Apps.loc[(Apps['V4_4_Interview'] > 7)].index)
Apps.drop(V4_4_outbound_idx, axis=0, inplace=True)
sum(pd.isna(Apps['V4_4_Interview']))

# V5_1_Interview check missing and frequency
# V5_1, V5_2, V5_4 >> object (they have some string on it)
# V5_4 >> numeric
# different unit of measurement together delete columns
# potentially check for the correlation with V4

Apps.drop(['V5_1_Interview', 'V5_2_Interview',
           'V5_3_Interview', 'V5_4_Interview'], axis=1, inplace=True)

# V6 (1 -10)
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
sum(pd.isna(Apps['V10']))

# V11 use across apps
pd.value_counts(Apps['V11'])
V11_outbound_idx = list(Apps.loc[(Apps['V11'] > 10)].index)
Apps.drop(V11_outbound_idx, axis=0, inplace=True)
sum(pd.isna(Apps['V11']))

# V12 satisfaction (1-- - 10++)
Apps.loc[:, ['V12_1_Interview', 'V12_2_Interview',
             'V12_3_Interview',
             'V12_4_Interview']].apply(pd.Series.value_counts)

# V12 has 6 " " need to use coerce to transform it to null
# V12_2 and V12_4 equal to categorical transform to numeric
Apps['V12_2_Interview'] = pd.to_numeric(Apps['V12_2_Interview'],
                                        errors='coerce')

pd.DataFrame.describe(Apps.loc[:, ['V12_1_Interview', 'V12_2_Interview',
                                   'V12_3_Interview',
                                   'V12_4_Interview']], include='all')

for i in ['V12_1_Interview', 'V12_2_Interview',
          'V12_3_Interview', 'V12_4_Interview']:
        V12_outbound_idx = list(Apps.loc[(Apps[i] > 10)].index)
        Apps.drop(V12_outbound_idx, axis=0, inplace=True)

# missing values are progressivly growing
sum(pd.isna(Apps['V12_4_Interview']))

# V13 future use (1-- - 10++)
pd.value_counts(Apps['V13'])

V13_outbound_idx = list(Apps.loc[(Apps['V13'] > 10)].index)
Apps.drop(V13_outbound_idx, axis=0, inplace=True)

# V14 future use (1-- - 10++)
pd.value_counts(Apps['V14'])
sum(pd.isna(Apps['V14']))

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
Apps.shape

# # ANALYSIS
# drop rows with na
Apps_clean = Apps.dropna()

# write down csv file so then you cantinue from here
# Apps_clean.to_csv('Apps_clean.csv')

# V2
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
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

# Investigate the differences in frequency between utilitarian and hedonic
# apps, ignoring the other variables for the moment.

# dependent variable distribution
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
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

# create a dummy variable
V2_enc = OneHotEncoder(categories='auto', drop='first')
V2_enc = V2_enc.fit(Apps_clean['V2'].values.reshape(-1, 1))
V2_enc.categories_
# 1 if ith apps is hedonic
# 0 if ith apps is utilitarian
# b0 interpreted as the average frequency among utilitarian apps
# b0 + b1 average frequency among hendonic apps
X = V2_enc.transform(Apps_clean['V2'].values.reshape(-1, 1)).toarray()

# run Linear Regression models
linear_reg = LinearRegression().fit(X, Apps_clean['V4_1_Interview'])
# Average frequency for utilitarian apps
Intercept = linear_reg.intercept_

# average frequency for hedonic apps
V2_hedonic = linear_reg.coef_
Intercept + V2_hedonic

# Using statsmodels.api

model = stm.OLS(Apps_clean['V4_1_Interview'], stm.add_constant(X)).fit()
# P-values are very low. This indicates that there is statistical
# evidence of a difference in average frequency between V2
# (utilitarian vs. hedonic apps)
print(model.summary())

# check if they change with the time/they do not really change with time
# does not seems the case looking at the graphs
# run Linear Regression models
model = stm.OLS(Apps_clean['V4_4_Interview'], stm.add_constant(X)).fit()
print(model.summary())

# V3 (free vs paid for)
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
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

# create a dummy variable
V3_enc = OneHotEncoder(categories='auto', drop='first')
V3_enc = V3_enc.fit(Apps_clean['V3'].values.reshape(-1, 1))
V3_enc.categories_
# 1 if ith apps is paid for
# 0 if ith apps is free
# b0 interpreted as the average frequency among free apps
# b0 + b1 average frequency among paid for apps
X = V3_enc.transform(Apps_clean['V3'].values.reshape(-1, 1)).toarray()

# create the model using statsmodels.api
for i in ['V4_1_Interview', 'V4_2_Interview',
          'V4_3_Interview', 'V4_4_Interview']:
        model = stm.OLS(Apps_clean[i], stm.add_constant(X)).fit()
        print(model.summary())

# p-value is small so there is statistical evidence of a difference
# in average frequency between V3 (free vs paid for apps)
# the relation does not change with time

# STEP 2: regress the independent variable on the medietor varaible
# TODO

# V6
# TODO

# STEP 3: Regress the independent variable and the mediator

# V10 extended functions
with sns.axes_style('white'):
        sns.jointplot(Apps_clean['V10'], Apps_clean['V4_2_Interview'],
                      kind='hex', color='k')
plt.show()

# it needs to be standardize otherwise not useful
# feature engineering binarize (always - not always)
pd.value_counts(Apps_clean['V10'])

# V11 cross app use [1--, 10++]
# feature engineering binarize (always - not always)
pd.value_counts(Apps_clean['V11'])

hexplot = sns.jointplot(Apps_clean['V11'], Apps_clean['V4_2_Interview'],
                        kind='hex')
cbar_ax = hexplot.fig.add_axes([.01, .25, .02, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.show()

# %%
