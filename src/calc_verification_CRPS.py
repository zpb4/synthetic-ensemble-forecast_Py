import os
import sys
sys.path.insert(0, os.path.abspath('./src'))
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from numba import njit

from util import declust_evts_extract,fig_title,compute_fcst_crps,water_day,calc_climo_ensemble

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#key user defined specifications
#location specifics
loc             = 'HHD'
keysite_label   = 'HHDW1'   #keysite for synthetic algorithm optimization and sampling

#basic algorithmic settings to extract correct optimized parameters file
opt_pct         = 0.99      #percentile of data to optimize to (e.g., 0.99 = optimize to top 1% of events by flow magnitude)
fixed_kk        = True      #use fixed k value for knn sampling?
fixed_knn_pwr   = True      #use a fixed knn_pwr value for knn sampling?
fix_kk          = 30        #if fixed_kk = True, what value to use (default: 20)
fix_knn_pwr     = -0.5      #if fixed_knn_pwr = True, what value to use (default: -0.5)

#NOTE: the optimization parameter .pkl output file always includes a value for 'fix_kk' and 'fix_knn_pwr', even if 'fixed_kk' and 'fixed_knn_pwr' are set to False
#If using the fixed_kk and fixed_knn_pwr set to 'False' in optimization, the 'fix_kk' and 'fix_knn_pwr' values are meaningless, but should be set to defaults for consistency

fit_gen_strategy = 'default'  #set to 'default' to fit all available paired hindcast/obs_fwd and gen all available obs_fwd; set to 'specify' to set yourself
n_samples = 10
workers = 10        #number of cores to utilize in parallel; 50 works on Hopper w/ADO test case; reduce as needed to not overload memory

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
data_dir = './data/%s' %(loc)
out_dir = './out/%s/keysite=%s_optpct=%s' %(loc,keysite_label,opt_pct)

# --------------------- Read in key inputs ----------------------------
#load basic data
outfile_npz = '/%s_hefs_gefs_daily.npz' %(loc)
data = np.load(data_dir + outfile_npz, allow_pickle=True) 
sites = data['sites']
esp   = data['esp']
hcst_dtg = data['hefs_dtg']

#load basic syn-forecasting data
data_outfile = '/syn-forecast-data_keysite=%s_optpct=%s_fixed-kk=%s_kk=%s_fixed-knn-pwr=%s_knn-pwr=%s.npz' %(keysite_label,opt_pct,fixed_kk,fix_kk,fixed_knn_pwr,fix_knn_pwr)
basic_data      = np.load(out_dir + data_outfile,allow_pickle=True)
hcst_fit        = basic_data['hcst_fit']
obs_fwd_fit     = basic_data['obs_fwd_fit']
obs_fwd_gen     = basic_data['obs_fwd_gen']
ixx_fit         = basic_data['ixx_fit']
ixx_gen         = basic_data['ixx_gen']

#load aggregated syn-forecast data
agg_outfile = '/syn-forecast-aggregated_keysite=%s_optpct=%s_fixed-kk=%s_kk=%s_fixed-knn-pwr=%s_knn-pwr=%s_samps=%s.npz' %(keysite_label,opt_pct,fixed_kk,fix_kk,fixed_knn_pwr,fix_knn_pwr,n_samples)
syn_fcst_data       = np.load(out_dir + agg_outfile,allow_pickle=True)
syn_fcst_arr        = syn_fcst_data['syn_fcst_arr']
resamp_date_arr     = syn_fcst_data['resamp_date_arr']
hcst_scale_arr      = syn_fcst_data['hcst_scale_arr']


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#calculations
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#user defined inputs
nep    = 0.9                #what nonexceedance probability to look at

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
site_idx = np.where((sites == keysite_label))[0][0]     #use keysite as large event sampling index

#subset size for the fit and gen data
n_eval_fit = round((1-nep) * len(ixx_fit))
n_eval_gen = round((1-nep) * len(ixx_gen))
    
#extract the fit and gen indices
eval_fit_idx = np.argsort(obs_fwd_fit[site_idx,:,0])[::-1][:n_eval_fit]
eval_gen_idx = np.argsort(obs_fwd_gen[site_idx,:,0])[::-1][:n_eval_gen]


#-----------------------------------------------------------------
#evaluate hindcasts in fit period
obs_fwd_eval_fit = obs_fwd_fit[:,eval_fit_idx,:]
hcst_eval_fit = hcst_fit[:,eval_fit_idx,:,:]

crps_array_hcst_fit = compute_fcst_crps(forecasts=hcst_eval_fit,obs_fwd=obs_fwd_eval_fit)

print(np.min(crps_array_hcst_fit),np.max(crps_array_hcst_fit))

#-----------------------------------------------------------------
#evaluate ESP hindcasts in fit period
hcst_fit_idx = np.isin(hcst_dtg,ixx_fit)
esp_fit = esp[:,hcst_fit_idx,:,:]
esp_eval_fit = esp_fit[:,eval_fit_idx,:,:]

crps_array_esp_fit = compute_fcst_crps(forecasts=esp_eval_fit,obs_fwd=obs_fwd_eval_fit)

print(np.min(crps_array_esp_fit),np.max(crps_array_esp_fit))

#-----------------------------------------------------------------
#calculate a climatological forecast for the generation set
max_lds = np.shape(hcst_fit)[2]
ixx_gen_climo = pd.date_range(ixx_gen[0],ixx_gen[-1] + pd.Timedelta(days=max_lds),freq='D')
dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in ixx_gen_climo])

obs = obs_fwd_gen[:,:,0]
n_ens = np.shape(hcst_fit)[3]

climo_array_gen = calc_climo_ensemble(obs=obs,n_ens=n_ens,n_leads=max_lds,dowy=dowy)
fit_idx = np.isin(ixx_gen,ixx_fit)
climo_array_fit = climo_array_gen[:,fit_idx,:,:]

climo_eval_fit = climo_array_fit[:,eval_fit_idx,:,:]
climo_eval_gen = climo_array_gen[:,eval_gen_idx,:,:]

crps_array_climo_fit = compute_fcst_crps(forecasts=climo_eval_fit,obs_fwd=obs_fwd_fit)
crps_array_climo_gen = compute_fcst_crps(forecasts=climo_eval_gen,obs_fwd=obs_fwd_gen)

#----------------------------------------------------------------
#evaluate syn_forecasts in fit period
syn_fit_idx = np.isin(ixx_gen,ixx_fit)
syn_fcst_fit = syn_fcst_arr[:,:,syn_fit_idx,:,:]
syn_fcst_eval_fit = syn_fcst_fit[:,:,eval_fit_idx,:,:]

crps_dims = np.shape(crps_array_hcst_fit)
crps_array_syn_fcst_fit = np.full((n_samples,crps_dims[0],crps_dims[1],crps_dims[2]),np.nan)

for i in range(n_samples):
    crps_array_syn_fcst_fit[i,:,:,:] = compute_fcst_crps(forecasts=syn_fcst_eval_fit[i,:,:,:,:],obs_fwd=obs_fwd_eval_fit)

print(np.min(crps_array_syn_fcst_fit),np.max(crps_array_syn_fcst_fit))


#---------------------------------------------------------------
#evaluate syn_forecasts in gen period
syn_fcst_eval_gen = syn_fcst_arr[:,:,eval_gen_idx,:,:]
obs_fwd_eval_gen = obs_fwd_gen[:,eval_gen_idx,:]

crps_dims = np.shape(syn_fcst_eval_gen)[:4]
crps_array_syn_fcst_gen = np.full((crps_dims),np.nan)

for i in range(n_samples):
    crps_array_syn_fcst_gen[i,:,:,:] = compute_fcst_crps(forecasts=syn_fcst_eval_gen[i,:,:,:,:],obs_fwd=obs_fwd_eval_gen)

print(np.min(crps_array_syn_fcst_gen),np.max(crps_array_syn_fcst_gen))

#--------------------------------------------------------------
#CRPS skill scores
#hindcasts
#eval against ESP forecasts
hcst_crpss_fit_esp = 1 - (crps_array_hcst_fit / crps_array_esp_fit)
print(np.min(hcst_crpss_fit_esp),np.max(hcst_crpss_fit_esp))
#eval against climatological forecasts
hcst_crpss_fit_climo = 1 - (crps_array_hcst_fit / crps_array_climo_fit)
print(np.min(hcst_crpss_fit_climo),np.max(hcst_crpss_fit_climo))

#synthetic forecasts
#Fit period
#eval against ESP forecasts
syn_fcst_crpss_fit_esp = np.full(np.shape(crps_array_syn_fcst_fit),np.nan)

for i in range(n_samples):
    syn_fcst_crpss_fit_esp[i,:,:,:] = 1 - (crps_array_syn_fcst_fit[i,:,:,:] / crps_array_esp_fit)
print(np.min(syn_fcst_crpss_fit_esp),np.max(syn_fcst_crpss_fit_esp))

#eval against climo forecasts
syn_fcst_crpss_fit_climo = np.full(np.shape(crps_array_syn_fcst_fit),np.nan)

for i in range(n_samples):
    syn_fcst_crpss_fit_climo[i,:,:,:] = 1 - (crps_array_syn_fcst_fit[i,:,:,:] / crps_array_climo_fit)
print(np.min(syn_fcst_crpss_fit_climo),np.max(syn_fcst_crpss_fit_climo))

#Gen period
#eval against climo forecasts (cannot eval against ESP for the entire period)
syn_fcst_crpss_gen_climo = np.full(np.shape(crps_array_syn_fcst_gen),np.nan)

for i in range(n_samples):
    syn_fcst_crpss_gen_climo[i,:,:,:] = 1 - (crps_array_syn_fcst_gen[i,:,:,:] / crps_array_climo_gen)
print(np.min(syn_fcst_crpss_gen_climo),np.max(syn_fcst_crpss_gen_climo))

outfile_npz = fr'/%s_ensemble-skill_CRPS_NEP={nep}.npz' 
np.savez(out_dir + outfile_npz,crps_array_hcst_fit=crps_array_hcst_fit,crps_array_esp_fit=crps_array_esp_fit,crps_array_climo_fit=crps_array_climo_fit,
         crps_array_climo_gen=crps_array_climo_gen,crps_array_syn_fcst_fit=crps_array_syn_fcst_fit,crps_array_syn_fcst_gen=crps_array_syn_fcst_gen,
         hcst_crpss_fit_esp=hcst_crpss_fit_esp,hcst_crpss_fit_climo=hcst_crpss_fit_climo,syn_fcst_crpss_fit_esp=syn_fcst_crpss_fit_esp,
         syn_fcst_crpss_fit_climo=syn_fcst_crpss_fit_climo,syn_fcst_crpss_gen_climo=syn_fcst_crpss_gen_climo)



    
####################################################################END#######################################################################