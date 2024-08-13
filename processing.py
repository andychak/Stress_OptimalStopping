import os
import sys
import tqdm
import pandas as pd
import numpy as np
import glob
import datetime as dt
from datetime import timedelta
from scipy.optimize import minimize
import datetime
from tqdm import tqdm
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm
import scipy.stats



processfile = 'processing.txt'
def semi_std(series):
    mean = series.mean()
    semi_var = ((series[series < mean] - mean) ** 2).mean()
    return semi_var ** 0.5
def run_stress_analysis(mytickers, topic):  
    
        file = open(processfile, 'w')
        file.write(f"Processing: {mytickers} {topics}")
        file.write("\n")     
        file.close()
        outputdf = pd.DataFrame(columns=['ticker','date','Adj Close',
                'BH_days', 'BH_price', 'BH_profit_pct', 'BH_profit_dollars', 
                'OS_price', 'OS_days', 'OS_profit_pct', 'OS_profit_dollars', 
                'Stress_price', 'Stress_days', 'Stress_profit_pct', 'Stress_profit_dollars', 
                'StressVol_price', 'StressVol_days', 'StressVol_profit_pct', 'StressVol_profit_dollars'])
        for ticker in tqdm(mytickers, desc=f"Processing topic: {topic}"):
                    
                stressdf = pd.read_parquet(f'/data/{ticker}_{topic}_stress_with_stockvol.parquet')
                pricedf = pd.read_csv(f'/data/{ticker}_stock_data.csv')
                pricedf['date'] = pd.to_datetime(pricedf['Date'], format = 'mixed')
                pricedf.rename(columns={'Ticker': 'ticker'}, inplace=True)
                pricedf = pricedf.sort_values(by=['ticker','date'])
                stressdf['date'] = pd.to_datetime(stressdf['date'], format = 'mixed')
                stressdf = stressdf.sort_values(by=['ticker', 'date'])
                
                for idx, row in stressdf.iterrows():
                        Adj_Close = row['Adj Close']
                        volmult = 2
                        vol = row['Vol'] * volmult
                        if pd.isna(Adj_Close) or row['mean']==0 or pd.isna(row['Vol']) or pd.isna(row['mean']) or pd.isna(row['std']):
                                continue  # Skip rows where Adj Close is None
                        riskunit = row['Vol']/Adj_Close   
                        UL_OS = Adj_Close + row['Vol']
                        LL_OS = Adj_Close - row['Vol']
                        UL_Stress = Adj_Close + (row['Vol'])/row['mean']
                        LL_Stress = Adj_Close - (row['Vol'])/row['mean']
                        UL_Stress_Vol = Adj_Close + (row['Vol']/(row['mean']+row['std']))
                        LL_Stress_Vol = Adj_Close - (row['Vol']/(row['mean']+row['std']))
             
                        
                        file = open(processfile, 'a')
                        file.write(f'{ticker} {row["date"]} Adj Close: {Adj_Close} OS:{UL_OS} {LL_OS} Stress:{UL_Stress} {LL_Stress} Stress Vol: {UL_Stress_Vol} {LL_Stress_Vol}')
                        file.write("\n")
                        file.close()
                        start_date = row['date']
                        if idx + 1 < len(stressdf):
                                end_date = stressdf['date'].iloc[idx + 1]
                                BH_price = stressdf['Adj Close'].iloc[idx + 1]
                        else:
                                end_date = row['date'] 
                                BH_price = row['Adj Close']
                                   
                        BH_days = (end_date - start_date).days
                        BH_profit_pct = (BH_price - Adj_Close)/Adj_Close
                        BH_profit_dollars = BH_price - Adj_Close

                        filtered_pricedf = pricedf[(pricedf['date'] > start_date) & (pricedf['date'] < end_date)]
                       
                        # Iterate over each row in the filtered pricedf
                        
                        os_crossed, stress_crossed, stressvol_crossed = False, False, False
                        for _, price_row in filtered_pricedf.iterrows():    
                                price_price = price_row['Adj Close']
                                if pd.isna(price_price):
                                        continue  # Skip rows where Adj Close is None
                                # Check for OS condition
                               
                                if (price_price > UL_OS or price_price < LL_OS) and not os_crossed:
                                        OS_days = (price_row['date'] - start_date).days
                                        OS_price = price_price
                                        OS_profit_pct = (OS_price - Adj_Close)/Adj_Close
                                        OS_profit_dollars = OS_price - Adj_Close
                                        os_crossed = True
                                        
                                # Check for Stress condition
                                if (price_price > UL_Stress or price_price < LL_Stress) and not stress_crossed:
                                        Stress_days = (price_row['date'] - start_date).days
                                        Stress_price = price_price
                                        Stress_profit_pct = (Stress_price - Adj_Close)/Adj_Close
                                        Stress_profit_dollars = Stress_price - Adj_Close
                                        stress_crossed = True
                                        
                                # Check for Stress Vol condition
                                if (price_price > UL_Stress_Vol or price_price < LL_Stress_Vol) and not stressvol_crossed:
                                        StressVol_days = (price_row['date'] - start_date).days
                                        StressVol_price = price_price
                                        StressVol_profit_pct = (StressVol_price - Adj_Close)/Adj_Close
                                        StressVol_profit_dollars = StressVol_price - Adj_Close
                                        stressvol_crossed = True
                                if os_crossed and stress_crossed and stressvol_crossed:
                                        break 
                        if not stress_crossed:
                                Stress_days = BH_days
                                Stress_price = BH_price
                                Stress_profit_pct = BH_profit_pct
                                Stress_profit_dollars = BH_profit_dollars
                        if not stressvol_crossed:
                                StressVol_days = BH_days
                                StressVol_price = BH_price
                                StressVol_profit_pct = BH_profit_pct
                                StressVol_profit_dollars = BH_profit_dollars 
                        if not os_crossed:
                                OS_days = BH_days
                                OS_price = BH_price
                                OS_profit_pct = BH_profit_pct
                                OS_profit_dollars = BH_profit_dollars
                                      
                        row_data = {'ticker': row['ticker'],'date': row['date'],'Adj Close': Adj_Close,'BH_days': BH_days,'BH_price': BH_price,'BH_profit_pct': BH_profit_pct,'BH_profit_dollars': BH_profit_dollars,
                        'OS_price': OS_price,
                        'OS_days': OS_days,
                        'OS_profit_pct': OS_profit_pct,
                        'OS_profit_dollars': OS_profit_dollars,
                        'Stress_price': Stress_price,
                        'Stress_days': Stress_days,
                        'Stress_profit_pct': Stress_profit_pct,
                        'Stress_profit_dollars': Stress_profit_dollars,
                        'StressVol_price': StressVol_price,
                        'StressVol_days': StressVol_days,
                        'StressVol_profit_pct': StressVol_profit_pct,
                        'StressVol_profit_dollars': StressVol_profit_dollars
                        }
                        row_df = pd.DataFrame([row_data])
                        outputdf = pd.concat([outputdf, row_df], ignore_index=True)
        outputdf = outputdf.dropna(subset=['Adj Close'])
        # Sort the DataFrame by ticker and date
        outputdf = outputdf.sort_values(by=['ticker', 'date'])

        # Drop the bottom row of each ticker
        outputdf = outputdf.groupby('ticker').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
               
        return outputdf

newdf2 = run_stress_analysis(sample,'any_topic')

newdf2.to_csv('no_topic_output.csv', index=False)
# Group by ticker and calculate the required statistics
summary_by_ticker = newdf2.groupby('ticker').agg(
    trades=('date', 'nunique'),
    #BH_days=('BH_days', 'mean'),
    BH_profit_pct=('BH_profit_pct', 'mean'),BH_profit_std=('BH_profit_pct', 'std'),
    #OS_days=('OS_days', 'mean'),
    OS_profit_pct=('OS_profit_pct', 'mean'),OS_profit_std=('OS_profit_pct', 'std'),
    #Stress_days=('Stress_days', 'mean'),
    Stress_profit_pct=('Stress_profit_pct', 'mean'),Stress_profit_std=('Stress_profit_pct', 'std'),
   
    #StressVol_days=('StressVol_days', 'mean'),
    StressVol_profit_pct=('StressVol_profit_pct', 'mean'), StressVol_profit_std=('StressVol_profit_pct', 'std'),
    BH_semi_std=('BH_profit_pct', semi_std),
    OS_semi_std=('OS_profit_pct', semi_std),
    Stress_semi_std=('Stress_profit_pct', semi_std),
    StressVol_semi_std=('StressVol_profit_pct', semi_std)
).reset_index()
summary_by_ticker['BH_sortino_ratio'] = summary_by_ticker['BH_profit_pct'] / summary_by_ticker['BH_semi_std']
summary_by_ticker['OS_sortino_ratio'] = summary_by_ticker['OS_profit_pct'] / summary_by_ticker['OS_semi_std']
summary_by_ticker['Stress_sortino_ratio'] = summary_by_ticker['Stress_profit_pct'] / summary_by_ticker['Stress_semi_std']
summary_by_ticker['StressVol_sortino_ratio'] = summary_by_ticker['StressVol_profit_pct'] / summary_by_ticker['StressVol_semi_std']

summary_by_ticker['BH_sharpe_ratio'] = summary_by_ticker['BH_profit_pct'] / summary_by_ticker['BH_profit_std']
summary_by_ticker['OS_sharpe_ratio'] = summary_by_ticker['OS_profit_pct'] / summary_by_ticker['OS_profit_std']
summary_by_ticker['Stress_sharpe_ratio'] = summary_by_ticker['Stress_profit_pct'] / summary_by_ticker['Stress_profit_std']
summary_by_ticker['StressVol_sharpe_ratio'] = summary_by_ticker['StressVol_profit_pct'] / summary_by_ticker['StressVol_profit_std']

# Calculate the overall average for each column
overall_average = summary_by_ticker.mean(numeric_only=True)
overall_average['ticker'] = 'Overall Average'

# Convert the overall average to a DataFrame
overall_average_df = pd.DataFrame([overall_average])

# Concatenate the summary table with the overall average
summary_by_ticker = pd.concat([summary_by_ticker, overall_average_df], ignore_index=True)

# Display the summary table
import pandas as pd

# Data
data = {
    "Metric": ["Return Mean", "Return Std Dev", "Sharpe Ratio", "Sortino Ratio"],
    "BH": [
        f"{overall_average_profit_pct_mean_BH * 100:.2f}%",
        f"{overall_average_profit_pct_std_BH * 100:.2f}%",
        f"{overall_average_sharpe_ratio_BH:.2f}",
        f"{overall_average_sortino_ratio_BH:.2f}"
    ],
    "OS": [
        f"{overall_average_profit_pct_mean_OS * 100:.2f}%",
        f"{overall_average_profit_pct_std_OS * 100:.2f}%",
        f"{overall_average_sharpe_ratio_OS:.2f}",
        f"{overall_average_sortino_ratio_OS:.2f}"
    ],
    "Stress": [
        f"{overall_average_profit_pct_mean_Stress * 100:.2f}%",
        f"{overall_average_profit_pct_std_Stress * 100:.2f}%",
        f"{overall_average_sharpe_ratio_Stress:.2f}",
        f"{overall_average_sortino_ratio_Stress:.2f}"
    ],
    "StressVol": [
        f"{overall_average_profit_pct_mean_StressVol * 100:.2f}%",
        f"{overall_average_profit_pct_std_StressVol * 100:.2f}%",
        f"{overall_average_sharpe_ratio_StressVol:.2f}",
        f"{overall_average_sortino_ratio_StressVol:.2f}"
    ]
}

