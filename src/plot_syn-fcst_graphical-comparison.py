import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import properscoring as ps
from util import declust_evts_extract,fig_title

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
### ------------- Time series plots of HEFS and synthetic forecasts for selected dates ------------- ###
plt_site = 'HHDW1'
site_idx = np.where((sites == plt_site))[0][0]
max_lds = np.shape(hcst_fit)[2]

#plot top X events comparing hindcast and syn-forecasts (3 samples)
n_evts      = 10
n_syn_plots = 3
lds         = [1,3,5,7]
ylm_scale   = 1.25


title_locs  = 0.91/(n_syn_plots+1) * np.flip(np.arange((n_syn_plots+1))) + 1.25/(n_syn_plots+1)/2

sset_idx_syn_fcst = np.isin(ixx_gen,ixx_fit)
syn_samps = np.random.choice(np.arange(n_samples),size=n_syn_plots,replace=False)

syn_fcst_plt = syn_fcst_arr[:,site_idx,sset_idx_syn_fcst,:,:] 
syn_fcst_plt = syn_fcst_plt[syn_samps,:,:,:] / 1000             #kcfs
hcst_plt = hcst_fit[site_idx,:,:,:] / 1000                      #kcfs
obs_fwd_plt = obs_fwd_fit[site_idx,:,:] / 1000                  #kcfs

#calculate top X events indices
ext_idx = declust_evts_extract(obs_fwd_plt[:,0],n_evts=n_evts,sep=15)

for i in range(n_evts):
    #plot setup
    fig = plt.figure(layout='constrained',figsize=(10,8))
    gs0 = fig.add_gridspec((n_syn_plots + 1),len(lds))
    
    for s in range(n_syn_plots + 1):
        for j in range(len(lds)):
            ax1 = fig.add_subplot(gs0[s,j])
            ylm = max(obs_fwd_plt[ext_idx[0],:])*ylm_scale
            ax1.set_ylim([0,ylm])
            ax1.text(lds[j],0.9*ylm,str(ixx_fit[ext_idx[i]])[:10])
            if s == 0:
                ax1.set_title(plt_site)
                for k in range(np.shape(hcst_plt)[2]):
                    ax1.axvline(lds[j],c='gray',linewidth=0.5,linestyle='--',alpha=0.5)
                    ens_out = obs_fwd_plt[(ext_idx[i]-lds[j]),:].copy()
                    ens_out[1:] = hcst_plt[(ext_idx[i]-lds[j]),:,k].copy()
                    ax1.plot(np.arange(max_lds+1),ens_out,c='gray',linewidth=1,alpha=0.25)
            else:
                for k in range(np.shape(hcst_plt)[2]):
                    ax1.axvline(lds[j],c='gray',linewidth=0.5,linestyle='--',alpha=0.5)
                    ens_out = obs_fwd_plt[(ext_idx[i]-lds[j]),:].copy()
                    ens_out[1:] = syn_fcst_plt[(s-1),(ext_idx[i]-lds[j]),:,k].copy()
                    ax1.plot(np.arange(max_lds+1),ens_out,c='salmon',linewidth=1,alpha=0.25)
            ax1.plot(np.arange(max_lds+1),obs_fwd_plt[(ext_idx[i]-lds[j]),:],linewidth=2,c='black')
            
            if j == 0 and s == 0:
                fig_title(fig,r'Hindcast',loc=(-0.02,title_locs[s]),fontsize='large',rotation=90,ha='center',va='center')
            elif j == 0 and s != 0:
                fig_title(fig,fr'SynForecast#{s}',loc=(-0.02,title_locs[s]),fontsize='large',rotation=90,ha='center',va='center')
            
            if j == 0:
                ax1.set_ylabel('Flow (kcfs)')
            else:
                ax1.yaxis.set_ticklabels([])
                
            if s == (n_syn_plots):
                ax1.set_xlabel('Lead Time (days)')
            else:
                ax1.xaxis.set_ticklabels([])

    plt.show()

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
### ------------- Plot across sites for same lead time ------------- ###
ld = 3
ylm_scale   = 1.5
site_rearrange_idx = [0,1]        #want to plot FOLC1 first

syn_fcst_plt = syn_fcst_arr[:,:,sset_idx_syn_fcst,:,:] 
syn_fcst_plt = syn_fcst_plt[syn_samps,:,:,:] / 1000             #kcfs
hcst_plt = hcst_fit[:,:,:,:] / 1000                      #kcfs
obs_fwd_plt = obs_fwd_fit[:,:,:] / 1000                  #kcfs

for i in range(n_evts):
    #plot setup
    fig = plt.figure(layout='constrained',figsize=(10,8))
    gs0 = fig.add_gridspec((n_syn_plots + 1),len(sites))
    
    for s in range(n_syn_plots + 1):
        for j in range(len(sites)):
            ax1 = fig.add_subplot(gs0[s,j])
            ylm = max(obs_fwd_plt[site_rearrange_idx[j],ext_idx[i]-ld,:])*ylm_scale
            ax1.set_ylim([0,ylm])
            ax1.text(ld,0.9*ylm,str(ixx_fit[ext_idx[i]])[:10])
            if s == 0:
                ax1.set_title(sites[site_rearrange_idx[j]])
                for k in range(np.shape(hcst_plt)[2]):
                    ax1.axvline(ld,c='gray',linewidth=0.5,linestyle='--',alpha=0.5)
                    ens_out = obs_fwd_plt[site_rearrange_idx[j],(ext_idx[i]-ld),:].copy()
                    ens_out[1:] = hcst_plt[site_rearrange_idx[j],(ext_idx[i]-ld),:,k].copy()
                    ax1.plot(np.arange(max_lds+1),ens_out,c='gray',linewidth=1,alpha=0.25)
            else:
                for k in range(np.shape(hcst_plt)[2]):
                    ax1.axvline(ld,c='gray',linewidth=0.5,linestyle='--',alpha=0.5)
                    ens_out = obs_fwd_plt[site_rearrange_idx[j],(ext_idx[i]-ld),:].copy()
                    ens_out[1:] = syn_fcst_plt[(s-1),site_rearrange_idx[j],(ext_idx[i]-ld),:,k].copy()
                    ax1.plot(np.arange(max_lds+1),ens_out,c='salmon',linewidth=1,alpha=0.25)
            ax1.plot(np.arange(max_lds+1),obs_fwd_plt[site_rearrange_idx[j],(ext_idx[i]-ld),:],linewidth=2,c='black')
            
            if j == 0 and s == 0:
                fig_title(fig,r'Hindcast',loc=(-0.02,title_locs[s]),fontsize='large',rotation=90,ha='center',va='center')
            elif j == 0 and s != 0:
                fig_title(fig,fr'SynForecast#{s}',loc=(-0.02,title_locs[s]),fontsize='large',rotation=90,ha='center',va='center')
            
            if j == 0:
                ax1.set_ylabel('Flow (kcfs)')
                
            if s == (n_syn_plots):
                ax1.set_xlabel('Lead Time (days)')
            else:
                ax1.xaxis.set_ticklabels([])

    plt.show()
    
    
    
####################################################################END#######################################################################