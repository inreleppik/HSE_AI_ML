import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def CalcOutliers(df_num):
    # Вычисление первого (Q1) и третьего (Q3) квартилей
    Q1 = np.percentile(df_num, 25)
    Q3 = np.percentile(df_num, 75)
    IQR = Q3 - Q1  # Межквартильный размах
    
    # Определение границ выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Определение выбросов
    outliers_lower = [x for x in df_num if x < lower_bound]
    outliers_higher = [x for x in df_num if x > upper_bound]
    outliers_total = [x for x in df_num if x < lower_bound or x > upper_bound]
    
    # Данные без выбросов
    outliers_removed = [x for x in df_num if lower_bound <= x <= upper_bound]
    
    print(f'Идентифицировано нижних выбросов: {len(outliers_lower)}')
    print(f'Идентифицировано верхних выбросов: {len(outliers_higher)}')
    print(f'Общее количество выбросов: {len(outliers_total)}')
    print(f'Количество наблюдений без выбросов: {len(outliers_removed)}')
    print(f'Процент выбросов: {round((len(outliers_total) / len(df_num)) * 100, 4)}%')
    
    return

def fill_mc_values(df, cols):
    for col in cols:
        df[col] = df[col].astype('object').fillna('unknown').astype(str)

def fill_mn_values(df, cols):
    for col in cols:
        df[col] = df[col].fillna(-999)

def plot_cf(df, column):
    tmp = pd.crosstab(df[column], df['isFraud'], normalize='index') * 100
    tmp.reset_index(inplace=True)
    tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)
    
    cat_counts = df[column].value_counts()
    cat_perc = (cat_counts / cat_counts.sum() * 100).sort_index()
    
    df_melted = tmp.melt(id_vars=column, value_vars=['NoFraud', 'Fraud'],
                         var_name='isFraud', value_name='Percentage')
    
    order = sorted(df[column].unique())
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    sns.countplot(
        x=column,
        data=df,
        order=order,
        ax=axes[0],
        color='crimson'
    )
    
    axes[0].set_title(f'{column} Distribution', fontsize=18)
    axes[0].set_xlabel(column, fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    
    sns.barplot(
        x=column,
        y='Percentage',
        hue='isFraud',
        data=df_melted,
        order=order,
        ax=axes[1],
        palette=['powderblue', 'crimson']
    )
    axes[1].set_title(f'Fraud vs Non-Fraud percentage by {column}', fontsize=19)
    axes[1].set_xlabel(column, fontsize=12)
    axes[1].set_ylabel('%', fontsize=12)
    axes[1].set_ylim(0, 100)
    
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.2f', label_type='edge')
    
    sns.boxenplot(
        x=column,
        y='TransactionAmt',
        hue='isFraud',
        data=df[df['TransactionAmt'] <= 2000],
        order=order,
        ax=axes[2],
        palette=['powderblue', 'crimson']
    )
    axes[2].set_title(f'Transaction Amount by {column} and Fraud Status', fontsize=18)
    axes[2].set_xlabel(column, fontsize=12)
    axes[2].set_ylabel('TransactionAmt', fontsize=12)
    
    plt.show()

def plot_cfmv(df, col, lim=2000):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    total = len(df)
    total_amt = df['TransactionAmt'].sum()
    
    plt.figure(figsize=(16,14))    
    plt.suptitle(f'{col} Distributions ', fontsize=24)
    
    plt.subplot(211)
    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values), palette="Blues_d")
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,tmp['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=7) 
        
    g.set_ylim(0,max(sizes)*1.15)
    
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() \
                / df.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()
    perc_amt = perc_amt.fillna(0)
    plt.subplot(212)
    g1 = sns.barplot(x=col, y='TransactionAmt', 
                       data=amt, 
                       order=list(tmp[col].values),
                    palette="Blues_d")
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, 
                        order=list(tmp[col].values),
                       color='black', legend=False, )
    g1t.set_ylim(0,perc_amt['Fraud'].max()*1.1)
    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    g1.set_title(f"{col} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)
    g1.set_xlabel(f"{col} Category Names", fontsize=16)
    g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)
    g1.set_xticklabels(g.get_xticklabels(),rotation=45)    
    
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt*100),
                ha="center",fontsize=7) 
        
    plt.subplots_adjust(hspace=.4, top = 0.9)
    plt.show()

def plot_card_1235(df):
    tmp = pd.crosstab(df['card3'], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index().rename(columns={0: 'NoFraud', 1: 'Fraud'})
    
    tmp2 = pd.crosstab(df['card5'], df['isFraud'], normalize='index') * 100
    tmp2 = tmp2.reset_index().rename(columns={0: 'NoFraud', 1: 'Fraud'})
    
    total = len(df)
    
    plt.figure(figsize=(14, 30))
    
    plt.subplot(611)
    g = sns.kdeplot(df[df['isFraud'] == 1]['card1'], label='Fraud', fill=True)
    g = sns.kdeplot(df[df['isFraud'] == 0]['card1'], label='NoFraud', fill=True)
    g.legend()
    g.set_title("Card 1 Values Distribution by Target", fontsize=16)
    g.set_xlabel("Card 1 Values", fontsize=12)
    g.set_ylabel("Density", fontsize=12)
    
    plt.subplot(612)
    g1 = sns.kdeplot(df[df['isFraud'] == 1]['card2'].dropna(), label='Fraud', fill=True)
    g1 = sns.kdeplot(df[df['isFraud'] == 0]['card2'].dropna(), label='NoFraud', fill=True)
    g1.legend()
    g1.set_title("Card 2 Values Distribution by Target", fontsize=16)
    g1.set_xlabel("Card 2 Values", fontsize=12)
    g1.set_ylabel("Density", fontsize=12)
    
    plt.subplot(613)
    g2 = sns.countplot(x='card3', data=df, order=list(tmp.card3.values))
    g2.set_title("Card 3 Values Distribution", fontsize=16)
    g2.set_xlabel("Card 3 Values", fontsize=12)
    g2.set_ylabel("Count", fontsize=12)
    for p in g2.patches:
        height = p.get_height()
        g2.text(p.get_x() + p.get_width() / 2., height + 2000,
                '{:1.2f}%'.format(height / total * 100), ha="center", fontsize = 7)
    
    plt.subplot(614)
    g3 = sns.barplot(x='card3', y='Fraud', data=tmp, palette="Blues_d")
    g3.set_title("Card 3 Fraud Transaction Percentage", fontsize=16)
    g3.set_xlabel("Values", fontsize=12)
    g3.set_ylabel("%", fontsize=12)
    for p in g3.patches:
        g3.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.1,
                '{:.2f}%'.format(p.get_height()), ha="center", fontsize = 7)
    
    plt.subplot(615)
    g4 = sns.countplot(x='card5', data=df, order=list(tmp2.card5.values))
    g4.set_title("Card 5 Values Distribution", fontsize=16)
    g4.set_xlabel("Values", fontsize=12)
    g4.set_ylabel("Count", fontsize=12)
    g4.set_xticklabels(g4.get_xticklabels(), rotation=90)
    for p in g4.patches:
        height = p.get_height()
        g4.text(p.get_x() + p.get_width() / 2., height + 2000,
                '{:1.2f}%'.format(height / total * 100), ha="center", fontsize = 7)
    
    plt.subplot(616)
    g5 = sns.barplot(x='card5', y='Fraud', data=tmp2, palette="Blues_d")
    g5.set_title("Card 5 Fraud Transaction Percentage", fontsize=16)
    g5.set_xlabel("Values", fontsize=12)
    g5.set_ylabel("%", fontsize=12)
    g5.set_xticklabels(g5.get_xticklabels(), rotation=90)
    for p in g5.patches:
        g5.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.1,
                '{:.2f}%'.format(p.get_height()), ha="center", fontsize = 7)
    
    plt.subplots_adjust(hspace=0.6, top=0.95)
    plt.show()

