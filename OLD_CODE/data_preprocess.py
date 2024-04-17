# -*- coding: utf-8 -*-
"""
Pre-process GEFCom 2014 data

@author: a.stratigakos
"""

import pickle
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)


# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5,2) # Height can be changed
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

#%%
folder_path = 'C:\\Users\\akyla\\feature-deletion-robust\\data\\wind-GEFCom2014\\'

df_list = []

for zone_num in range(1,11):
    
    file_path = folder_path+f'final_W_Zone{zone_num}.csv'
    
    wind_df = pd.read_csv(file_path, index_col=1, parse_dates=True)
    wind_df = wind_df.interpolate('linear')
    wind_df = wind_df[:'2013-12-30']
    wind_df = wind_df.rename(columns = {'TARGETVAR':'POWER'})

    del wind_df['ZONEID']
            
    wind_df['wspeed10'] = np.sqrt(wind_df['U10'] ** 2 + wind_df['V10'] ** 2)
    wind_df['wspeed10_sq'] = np.power(wind_df['wspeed10'], 2)
    wind_df['wspeed10_cb'] = np.power(wind_df['wspeed10'], 3)
    wind_df['wdir10'] = (180/np.pi)*(np.arctan2(wind_df['U10'], wind_df['V10']))
    
    wind_df['wspeed100'] = np.sqrt(wind_df['U100'] ** 2 + wind_df['V100'] ** 2)
    wind_df['wdir100'] = (180/np.pi)*(np.arctan2(wind_df['U100'], wind_df['V100']))
    wind_df['wspeed100_sq'] = np.power(wind_df['wspeed100'], 2)
    wind_df['wspeed100_cb'] = np.power(wind_df['wspeed100'], 3)
    
    # direction to rads
    wind_df['wdir10_rad'] = np.sin(np.deg2rad(wind_df['wdir10']))
    wind_df['wdir100_rad'] = np.sin(np.deg2rad(wind_df['wdir100']))
    
    
    header = pd.MultiIndex.from_product([[f'Z{zone_num}'], wind_df.columns])

    wind_df.columns = header

    df_list.append(wind_df)

aggr_wind_df = pd.concat(df_list, axis = 1)
#%%
print(aggr_wind_df.xs('wdir10', axis=1, level=1).corr())

# Target = Zone1, Pred = Z2, Z4, Z8, Z9

# Add diurnal patters
aggr_wind_df['diurnal_1'] = np.sin(2*np.pi*(aggr_wind_df.index.hour+1)/24)
aggr_wind_df['diurnal_2'] = np.cos(2*np.pi*(aggr_wind_df.index.hour+1)/24)
aggr_wind_df['diurnal_3'] = np.sin(4*np.pi*(aggr_wind_df.index.hour+1)/24)
aggr_wind_df['diurnal_4'] = np.cos(4*np.pi*(aggr_wind_df.index.hour+1)/24)

aggr_wind_df.to_csv(cd+'\\data\\GEFCom2014-processed.csv')
