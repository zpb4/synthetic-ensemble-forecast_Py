# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:20:34 2026

@author: zpb4
"""

import os
import sys
sys.path.insert(0, os.path.abspath('./src'))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import matplotlib.pyplot as plt
from util import split_return
from hecdss import HecDss

max_leads = 16

loc = 'HHD'
sites = ['AUBW1','HHDW1']
"""
hefs_dir = '/data/projects/Hydro/CHPS/hindcasts/projects/firo/howardhanson/fcst/cw3e/flows/hefs/'
esp_dir = '/data/projects/Hydro/CHPS/hindcasts/projects/firo/howardhanson/fcst/cw3e/flows/esp/'
obs_dir = '/data/projects/Hydro/CHPS/hindcasts/projects/firo/howardhanson/obs/nwrfc/flows/'
"""
hefs_dir = './raw_data/HEFS'
esp_dir = './raw_data/ESP'
obs_dir = './raw_data'

hefs_out_dir = './raw_data/HEFS' 
esp_out_dir = './raw_data/ESP' 

out_dir = './data/%s' %(loc)

os.makedirs(hefs_out_dir,exist_ok=True)
os.makedirs(esp_out_dir,exist_ok=True)
os.makedirs(out_dir,exist_ok=True)
#---------------------- Read and process obs data ------------------------------
#NWRFC obs
obs_file = '/NWRFC_HHDW1_Historical_Inflow_WRES.csv'
obs_raw = pd.read_csv(fr'{obs_dir}{obs_file}',header=0,index_col=0) 
obs_dtg = pd.to_datetime(obs_raw.index.values)
obs_flow = obs_raw['value'].values
obs = pd.DataFrame(obs_flow,index=obs_dtg,columns=[sites[0]])

obs_outfile = '/observed_inflows_nwrfc.csv'
obs.to_csv(out_dir + obs_outfile,index=True)

"""
#USACE DSS file for hourly obs
dss_file = './raw_data/HHDhourlyInflow.dss' 
dss_data_file = '//HAH/Flow-In/14Mar1991-10Jul2025/1Hour/NWSRADIO-COMPUTED-REV/'
dss = HecDss(dss_file)
data = dss.get(dss_data_file)
obs_dtg = pd.to_datetime(data.times)
flow_cfs = data.values
obs_dtg_shift = obs_dtg + pd.Timedelta(hours=12)
flow_cfs_pd = pd.DataFrame(flow_cfs,index=obs_dtg_shift)
flow_cfs_daily = flow_cfs_pd.resample("D").mean()
flow_cfs_daily.index = flow_cfs_daily.index + pd.Timedelta(hours=12)

obs = flow_cfs_daily
obs_outfile = '/observed_inflows_dss.csv'
obs.to_csv(out_dir + obs_outfile,index=True)
obs_dtg = obs.index 
"""
#USACE obs
#Howard Hanson Dam inflows (HHDW1)
hhd_obs_file = '/HHDhourlyInflow_usace_nws.csv'
hhd_obs_raw = pd.read_csv(fr'{obs_dir}{hhd_obs_file}',header=0,index_col=0) 
hhd_obs_dtg = pd.to_datetime(hhd_obs_raw.index.values)
hhd_obs_flow = hhd_obs_raw['Flow(CFS)'].values.copy()
hhd_obs_flow[hhd_obs_flow<0] = np.nan
hhd_obs = pd.DataFrame(hhd_obs_flow,index=hhd_obs_dtg,columns=[sites[1]])
hhd_obs_out = hhd_obs.ffill()
hhd_obs_dly = hhd_obs_out.resample("24h", offset="12h", label='right', closed='right').mean()
hhd_obs_6hrly = hhd_obs_out.resample("6h", offset="12h", label='right', closed='right').mean()

#Auburn Local flows (AUBW1)
aub_obs_file = '/AuburnLocals3HourSmoothed.csv'
aub_obs_raw = pd.read_csv(fr'{obs_dir}{aub_obs_file}',header=0,index_col=0) 
aub_obs_dtg = pd.to_datetime(aub_obs_raw.index.values)
aub_obs_flow = aub_obs_raw['values'].values.copy()
aub_obs_flow[aub_obs_flow<0] = np.nan
aub_obs = pd.DataFrame(aub_obs_flow,index=aub_obs_dtg,columns=[sites[0]])
aub_obs_out = aub_obs.ffill()
aub_obs_dly = aub_obs_out.resample("24h", offset="12h", label='right', closed='right').mean()
aub_obs_6hrly = aub_obs_out.resample("6h", offset="12h", label='right', closed='right').mean()

#combined daily dataframe
dly_st = max(hhd_obs_dly.index[0],aub_obs_dly.index[0])
dly_en = min(hhd_obs_dly.index[-1],aub_obs_dly.index[-1])

dly_out = aub_obs_dly[dly_st:dly_en]
dly_out[sites[1]] = hhd_obs_dly[dly_st:dly_en].values

dly_outfile = '/observed_inflows_daily.csv'
dly_out.to_csv(fr'{out_dir}{dly_outfile}',index=True)

#combined 6-hourly dataframe
st_6hrly = max(hhd_obs_6hrly.index[0],aub_obs_6hrly.index[0])
en_6hrly = min(hhd_obs_6hrly.index[-1],aub_obs_6hrly.index[-1])
if st_6hrly.hour < 12:
    shift_st = st_6hrly.replace(hour=12)
elif st_6hrly.hour > 12:
    shift_st = st_6hrly.replace(day=st_6hrly.day + 1,hour=12)
    
if en_6hrly.hour < 12:
    shift_en = en_6hrly.replace(day=en_6hrly.day - 1,hour=12)
elif en_6hrly.hour > 12:
    shift_en = en_6hrly.replace(hour=12)

out_6hrly = aub_obs_6hrly[shift_st:shift_en]
out_6hrly[sites[1]] = hhd_obs_6hrly[shift_st:shift_en].values

outfile_6hrly = '/observed_inflows_6-hourly.csv'
out_6hrly.to_csv(fr'{out_dir}{outfile_6hrly}',index=True)

#---------------------- calculate obs_forward array - daily ------------------------------
obs_dtg_dly = dly_out.index.to_numpy(dtype="datetime64[us]")
n_obs, n_sites = len(obs_dtg_dly), len(sites)

n_time_forward = n_obs - max_leads      #we have to drop 'leads' observations
obs_forward_dly = np.full((n_sites, n_time_forward, max_leads+1), np.nan, dtype=float)

#IMPORTANT - because 'lead1' in hefs_forward is actually lead0 (i.e., it aligns with
#            the target day being predicted), we need to make sure 'lead1' in obs_forward
#            also aligns with the current date of interest
               
for j in range(n_sites):                  # 0..n_sites-1
    for i in range(n_time_forward):       # 0..(n_obs - leads - 1)
        # rows i+1 .. i+1+leads (remember, exclusive of upper index)
        obs_forward_dly[j, i, :] = dly_out.iloc[i:(i+max_leads+1), j].to_numpy()    #used to be obs_flows.iloc[(i+1):(i+1+leads), j].to_numpy(), but I think this was a mistake that caused misalignment with hefs_forward

# Dates associated with "lead0", i.e., the first lead entry in obs_forward:
# on date t, this is the sequence of obs flows over the NEXT `leads` days
obs_fwd_dtg_dly = obs_dtg_dly[:n_time_forward]

print(obs_dtg_dly[0],obs_dtg_dly[-1])
print(obs_fwd_dtg_dly[0],obs_fwd_dtg_dly[-1])

cols = [str(i)+'d_lead' for i in np.arange(max_leads+1)]
aub_obs_fwd = pd.DataFrame(obs_forward_dly[0,:,:],index=obs_fwd_dtg_dly,columns=cols)
aub_dly_obs_fwd_file = '/%s_obs-fwd[perfect-forc]_daily.csv' %(sites[0])
aub_obs_fwd.to_csv(fr'{out_dir}{aub_dly_obs_fwd_file}',index=True)

hhd_obs_fwd = pd.DataFrame(obs_forward_dly[1,:,:],index=obs_fwd_dtg_dly,columns=cols)
hhd_dly_obs_fwd_file = '/%s_obs-fwd[perfect-forc]_daily.csv' %(sites[1])
hhd_obs_fwd.to_csv(fr'{out_dir}{hhd_dly_obs_fwd_file}',index=True)

#---------------------- calculate obs_forward array - 6hourly ------------------------------
out_6hrly_init_idx = np.where(out_6hrly.index.hour == 12)
obs_dtg_6hrly = out_6hrly.index[out_6hrly_init_idx][:-1].to_numpy(dtype="datetime64[us]")
n_obs, n_sites = len(obs_dtg_6hrly), len(sites)

n_time_forward = n_obs - max_leads     #we have to drop 'leads'observations for 6-hourly
obs_forward_6hrly = np.full((n_sites, n_time_forward, (max_leads*4)+1), np.nan, dtype=float)

#IMPORTANT - because 'lead1' in hefs_forward is actually lead0 (i.e., it aligns with
#            the target day being predicted), we need to make sure 'lead1' in obs_forward
#            also aligns with the current date of interest
               
for j in range(n_sites):                  # 0..n_sites-1
    for i in range(n_time_forward):       # 0..(n_obs - leads - 1)
        # rows i+1 .. i+1+leads (remember, exclusive of upper index)
        st_idx = out_6hrly_init_idx[0][i]
        obs_forward_6hrly[j, i, :] = out_6hrly.iloc[st_idx:(st_idx+(max_leads*4)+1), j].to_numpy()    #used to be obs_flows.iloc[(i+1):(i+1+leads), j].to_numpy(), but I think this was a mistake that caused misalignment with hefs_forward

# Dates associated with "lead0", i.e., the first lead entry in obs_forward:
# on date t, this is the sequence of obs flows over the NEXT `leads` days
obs_fwd_dtg_6hrly = obs_dtg_6hrly[:n_time_forward]

print(obs_dtg_6hrly[0],obs_dtg_6hrly[-1])
print(obs_fwd_dtg_6hrly[0],obs_fwd_dtg_6hrly[-1])

cols = [str(i)+'hr_lead' for i in (6*np.arange((max_leads*4)+1))]
aub_obs_fwd = pd.DataFrame(obs_forward_6hrly[0,:,:],index=obs_fwd_dtg_6hrly,columns=cols)
aub_6hrly_obs_fwd_file = '/%s_obs-fwd[perfect-forc]_6-hourly.csv' %(sites[0])
aub_obs_fwd.to_csv(fr'{out_dir}{aub_6hrly_obs_fwd_file}',index=True)

hhd_obs_fwd = pd.DataFrame(obs_forward_6hrly[1,:,:],index=obs_fwd_dtg_6hrly,columns=cols)
hhd_6hrly_obs_fwd_file = '/%s_obs-fwd[perfect-forc]_6-hourly.csv' %(sites[1])
hhd_obs_fwd.to_csv(fr'{out_dir}{hhd_6hrly_obs_fwd_file}',index=True)


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#HEFS and ESP processing

#---------------------- process ESP 1989-2019 ------------------------------
file_names = os.listdir(esp_dir+'/%s' %(sites[0]))
st = '%s-%s-%s' %(file_names[0][:4],file_names[0][4:6],file_names[0][6:8])
en = '%s-%s-%s' %(file_names[-1][:4],file_names[-1][4:6],file_names[-1][6:8])

esp_template = pd.read_csv(esp_dir + '/%s/%s' %(sites[0],file_names[0]),header=0, skiprows=0, index_col=0)
ens_names = np.unique(esp_template['ensemblemember_id'].values)

ixx_out_esp = pd.date_range(st,en,freq='D')+pd.Timedelta(value='12h')

out_arr_esp = np.zeros((len(sites),len(ixx_out_esp),max_leads,len(ens_names)))
out_arr_esp_6hrly = np.zeros((len(sites),len(ixx_out_esp),max_leads*4,len(ens_names)))

def process_hourly_forecasts(k):
    file_names = os.listdir(esp_dir+'/%s' %(sites[k]))
    esp_template = pd.read_csv(esp_dir + '/%s/%s' %(sites[k],file_names[0]),header=0, skiprows=0, index_col=0)
    ens_names = np.unique(esp_template['ensemblemember_id'].values)

    fname_temp = file_names[0]
    
    new_raw_dir = '/%s_ESP_DailyInflow' %(sites[k])
    new_raw_dir_6hrly = '/%s_ESP_6HourlyInflow' %(sites[k])
    os.makedirs(esp_out_dir + new_raw_dir,exist_ok=True)
    os.makedirs(esp_out_dir + new_raw_dir_6hrly,exist_ok=True)
    site_arr = np.zeros((len(ixx_out_esp),max_leads,np.shape(out_arr_esp)[3]))
    site_arr_6hrly = np.zeros((len(ixx_out_esp),max_leads*4,np.shape(out_arr_esp)[3]))
    missing_dates = []
    for i in range(len(ixx_out_esp)):
        d_idx = ixx_out_esp[i].strftime('%Y%m%d')
        fname_out = d_idx + fname_temp[8:]
        try:
            esp_path = '/%s/%s' %(sites[k],fname_out)
            esp_in_hourly = pd.read_csv(esp_dir + esp_path,header=0, skiprows=0, index_col=0)
        except FileNotFoundError: 
            #if no file, record it as a missing date and replace it with a dummy file from a different year (1992 leap year to include all possible dates)
            try:
                missing_dates.append(ixx_out_esp[i])
                d_idx_mday = ixx_out_esp[i].strftime('%m%d')
                new_d_idx = '1992'+d_idx_mday
                fname_out = new_d_idx + fname_temp[8:]
            
                esp_path = '/%s/%s' %(sites[k],fname_out)
                esp_in_hourly = pd.read_csv(esp_dir + esp_path,header=0, skiprows=0, index_col=0)
            #if 1992 is not avail, try 1996
            except FileNotFoundError:
                missing_dates.append(ixx_out_esp[i])
                d_idx_mday = ixx_out_esp[i].strftime('%m%d')
                fname_out = new_d_idx + fname_temp[8:]
            
                esp_path = '/%s/%s' %(sites[k],fname_out)
                esp_in_hourly = pd.read_csv(esp_dir + esp_path,header=0, skiprows=0, index_col=0)
        
        ens_names = np.unique(esp_in_hourly['ensemblemember_id'].values)
        forc_dates = np.unique(esp_in_hourly['value_date'].values)
        forc_values = esp_in_hourly['value'].values
        forc_mat = np.reshape(forc_values,(len(forc_dates),len(ens_names)))

        site_in_6hourly = pd.DataFrame(forc_mat,index=pd.to_datetime(forc_dates),columns=ens_names)
        site_in_6hourly[site_in_6hourly<0] = np.nan
        site_in_6hourly_out = site_in_6hourly.ffill()
        outfile = new_raw_dir_6hrly + '/%s12_%s_esp_6hourly.csv' %(d_idx,sites[k])
        site_in_6hourly_out.to_csv(esp_out_dir + outfile,index=True)
        site_arr_6hrly[i,:,:] = site_in_6hourly_out
        
        
        site_daily = site_in_6hourly_out.resample("24h", offset="12h", label='right', closed='right').mean()
        
        site_daily_out = site_daily.iloc[:max_leads,:]
        
        site_arr[i,:,:] = site_daily_out
        
        outfile = new_raw_dir + '/%s12_%s_esp_daily.csv' %(d_idx,sites[k])
        site_daily_out.to_csv(esp_out_dir + outfile,index=True)

    return site_arr,missing_dates,site_arr_6hrly

par_out_esp = Parallel(n_jobs=len(sites))(delayed(process_hourly_forecasts)(i) for i in range(len(sites)))

missing_dates = []
    
for k in range(len(sites)):
    site_arr = par_out_esp[k][0]
    missing_dates.append(par_out_esp[k][1])
    out_arr_esp[k,:,:,:] = site_arr
    site_arr_esp_6hrly = par_out_esp[k][2]
    out_arr_esp_6hrly[k,:,:,:] = site_arr_esp_6hrly

#---------------------- process GEFSv12 1989-2023 ------------------------------
file_names = os.listdir(hefs_dir+'/%s' %(sites[0]))
st = '%s-%s-%s' %(file_names[0][:4],file_names[0][4:6],file_names[0][6:8])
en = '%s-%s-%s' %(file_names[-1][:4],file_names[-1][4:6],file_names[-1][6:8])

hefs_template = pd.read_csv(hefs_dir + '/%s/%s' %(sites[0],file_names[0]),header=0, skiprows=0, index_col=0)
ens_names = np.unique(hefs_template['ensemblemember_id'].values)

ixx_out = pd.date_range(st,en,freq='D')+pd.Timedelta(value='12h')

out_arr = np.zeros((len(sites),len(ixx_out),max_leads,len(ens_names)))
out_arr_6hrly = np.zeros((len(sites),len(ixx_out),max_leads*4,len(ens_names)))

fname_temp = file_names[0]

def process_hourly_forecasts(k):
    file_names = os.listdir(hefs_dir+'/%s' %(sites[k]))
    hefs_template = pd.read_csv(hefs_dir + '/%s/%s' %(sites[k],file_names[0]),header=0, skiprows=0, index_col=0)
    ens_names = np.unique(hefs_template['ensemblemember_id'].values)
    
    fname_temp = file_names[0]
    
    new_raw_dir = './%s_GEFSv12_HEFS_DailyInflow' %(sites[k])
    new_raw_dir_6hrly = './%s_GEFSv12_HEFS_6HourlyInflow' %(sites[k])
    os.makedirs(hefs_out_dir + new_raw_dir,exist_ok=True)
    os.makedirs(hefs_out_dir + new_raw_dir_6hrly,exist_ok=True)
    site_arr = np.zeros((len(ixx_out),max_leads,np.shape(out_arr)[3]))
    site_arr_6hrly = np.zeros((len(ixx_out),max_leads*4,np.shape(out_arr)[3]))
    missing_dates = []
    for i in range(len(ixx_out)):
        d_idx = ixx_out[i].strftime('%Y%m%d')
        fname_out = d_idx + fname_temp[8:]
        try:
            hefs_path = '/%s/%s' %(sites[k],fname_out)
            hefs_in_hourly = pd.read_csv(hefs_dir + hefs_path,header=0, skiprows=0, index_col=0)
        except FileNotFoundError: 
            #if no file, record it as a missing date and replace it with a dummy file from a different year (1992 leap year to include all possible dates)
            try:
                missing_dates.append(ixx_out[i])
                d_idx_mday = ixx_out[i].strftime('%m%d')
                new_d_idx = '1992'+d_idx_mday
                fname_out = new_d_idx + fname_temp[8:]
            
                hefs_path = '/%s/%s' %(sites[k],fname_out)
                hefs_in_hourly = pd.read_csv(hefs_dir + hefs_path,header=0, skiprows=0, index_col=0)
            #if 1992 is not avail, try 1996
            except FileNotFoundError:
                missing_dates.append(ixx_out[i])
                d_idx_mday = ixx_out[i].strftime('%m%d')
                fname_out = new_d_idx + fname_temp[8:]
            
                hefs_path = '/%s/%s' %(sites[k],fname_out)
                hefs_in_hourly = pd.read_csv(hefs_dir + hefs_path,header=0, skiprows=0, index_col=0)
        
        ens_names = np.unique(hefs_in_hourly['ensemblemember_id'].values)
        forc_dates = np.unique(hefs_in_hourly['value_date'].values)
        forc_values = hefs_in_hourly['value'].values
        forc_mat = np.reshape(forc_values,(len(forc_dates),len(ens_names)))

        site_in_6hourly = pd.DataFrame(forc_mat,index=pd.to_datetime(forc_dates),columns=ens_names)
        if site_in_6hourly.index[0].hour != 18:
            shift_st = site_in_6hourly.index[0].replace(hour=18)
            site_in_6hourly = site_in_6hourly[shift_st:site_in_6hourly.index[-1]]           #some sites have erroneous 12z time as first forecast
        
        site_in_6hourly[site_in_6hourly<0] = np.nan
        site_in_6hourly_out = site_in_6hourly.ffill()
        outfile = new_raw_dir_6hrly + '/%s12_%s_hefs_gefs_6hourly.csv' %(d_idx,sites[k])
        site_in_6hourly_out.to_csv(hefs_out_dir + outfile,index=True)
        site_arr_6hrly[i,:,:] = site_in_6hourly_out
        
        site_daily = site_in_6hourly_out.resample("24h", offset="12h", label='right', closed='right').mean()
        site_daily_out = site_daily.iloc[:max_leads,:]
        
        site_arr[i,:,:] = site_daily_out
        
        outfile = new_raw_dir + '/%s12_%s_hefs_gefs_daily.csv' %(d_idx,sites[k])
        site_daily_out.to_csv(hefs_out_dir + outfile,index=True)

    return site_arr,missing_dates,site_arr_6hrly

par_out = Parallel(n_jobs=len(sites))(delayed(process_hourly_forecasts)(i) for i in range(len(sites)))

missing_dates = []
    
for k in range(len(sites)):
    site_arr = par_out[k][0]
    missing_dates.append(par_out[k][1])
    out_arr[k,:,:,:] = site_arr
    site_arr_6hrly = par_out[k][2]
    out_arr_6hrly[k,:,:,:] = site_arr_6hrly

missing_dates = np.unique(missing_dates)

outfile_npz = '/%s_hefs_gefs_daily.npz' %(loc)
np.savez(out_dir + outfile_npz,obs_fwd=obs_forward_dly,hefs=out_arr,esp=out_arr_esp,sites=sites,obs_fwd_dtg=obs_fwd_dtg_dly,hefs_dtg=ixx_out.to_numpy(dtype="datetime64[us]"),esp_dtg=ixx_out_esp.to_numpy(dtype="datetime64[us]"),missing_dates=missing_dates)

outfile_npz_6hrly = '/%s_hefs_gefs_6hourly.npz' %(loc)
np.savez(out_dir + outfile_npz_6hrly,obs_fwd=obs_forward_6hrly,hefs=out_arr_6hrly,esp=out_arr_esp_6hrly,sites=sites,obs_fwd_dtg=obs_fwd_dtg_6hrly,hefs_dtg=ixx_out.to_numpy(dtype="datetime64[us]"),esp_dtg=ixx_out_esp.to_numpy(dtype="datetime64[us]"),missing_dates=missing_dates)

#//////////////////////////////////////////////////////////////////////////////////////////////
#Simple plot verification of synchronization
#//////////////////////////////////////////////////////////////////////////////////////////////
data = np.load(out_dir + outfile_npz, allow_pickle=True)
#hefs array [n_sites x n_obs x n_leads x n_ens]
hefs = data['hefs']
#esp array [n_sites x n_obs x n_leads x n_ens]
esp = data['esp']
#obs forward (perfect forecast array) [n_sites x n_obs x n_leads]  **note: n_leads is 1 longer than hefs_array because col 0 is day t observations in obs fwd
obs_fwd = data['obs_fwd']
#site index  **note: uses 'LAMC1F' and 'HOPC1L' in place of 'LAMC1' and 'HOPC1'
sites = data['sites']
#date/time vector
obs_fwd_dtg = data['obs_fwd_dtg']
hefs_dtg = data['hefs_dtg']
esp_dtg = data['esp_dtg']
#bad forecast days
bad_forcs = data['missing_dates']

print(np.shape(hefs))
print(np.shape(hefs))
print(np.shape(obs_fwd))
print(sites)
print(obs_fwd_dtg[0],obs_fwd_dtg[-1])
print(hefs_dtg[0],hefs_dtg[-1])
print(esp_dtg[0],esp_dtg[-1])
print(bad_forcs)

###############################################################
#plot timeseries comparison
#start and end dates likely to be in all runs
site = 'AUBW1'
forc_type = 'hefs'                          # 'hefs' or 'esp'
site_idx = np.where((sites == site))[0][0]

st_cmn = max(obs_fwd_dtg[0],hefs_dtg[0])
en_cmn = min(obs_fwd_dtg[-1],hefs_dtg[-1])

dtg_cmn = pd.date_range(st_cmn,en_cmn,freq='D').to_numpy(dtype="datetime64[us]")
 
sset_idx_hefs = np.isin(hefs_dtg,dtg_cmn)
sset_idx_obs_fwd = np.isin(obs_fwd_dtg,dtg_cmn)

obs_fwd_ver = obs_fwd[site_idx,sset_idx_obs_fwd,:]
hefs_ver = hefs[site_idx,sset_idx_hefs,:,:]
esp_ver = esp[site_idx,sset_idx_hefs,:,:]

#get indices of largest events
ext_idx = np.argsort(obs_fwd_ver[:,0])[::-1]

#plot the top 12 sorted events at a 3-d lead
fig = plt.figure(layout='constrained',figsize=(10,8))
gs0 = fig.add_gridspec(4,3)

ld = 3
evt_indices = ext_idx[:12] - ld

#plot HEFS
for i in range(len(evt_indices)):
    ax1 = fig.add_subplot(gs0[i])
    ax1.plot(np.arange(max_leads+1),obs_fwd_ver[evt_indices[i],:],linewidth=2,c='black')
    ax1.axvline(ld,c='gray',linewidth=0.5,linestyle='--',alpha=0.5)
    ylm = max(obs_fwd_ver[evt_indices[0],:])*1.1
    print(ylm)
    ax1.set_ylim([0,ylm])
    ax1.text(4,0.9*ylm,str(dtg_cmn[evt_indices[i]+ld])[:10])
    for k in range(np.shape(hefs_ver)[2]):
        ens_out = obs_fwd_ver[evt_indices[i],:].copy()
        if forc_type == 'esp':
            ens_out[1:] = esp_ver[evt_indices[i],:,k].copy()
        if forc_type == 'hefs':
            ens_out[1:] = hefs_ver[evt_indices[i],:,k].copy()
        ax1.plot(np.arange(max_leads+1),ens_out,c='gray',linewidth=1,alpha=0.5)

plt.show()


###############################################################END###########################################################################