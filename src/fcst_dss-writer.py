import os
from datetime import datetime
import numpy as np
import pandas as pd
from hecdss import HecDss
from hecdss import RegularTimeSeries
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#key user defined specifications
#location specifics
loc             = 'HHD'
keysite_label   = 'HHDW1'   #keysite for synthetic algorithm optimization and sampling
sites           = ['AUBW1','HHDW1']

#basic algorithmic settings to extract correct optimized parameters file
max_lds         = 15        #number of daily lead times to optimize to (default is total number of leads in hindcast dataset)
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

# site & dates to optimize on:
keysite_label = "HHDW1" #keysite for synthetic algorithm

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
data_dir = './data/%s' %(loc)
out_dir = './out/%s/keysite=%s_optpct=%s' %(loc,keysite_label,opt_pct)

# --------------------- Read in key inputs ----------------------------
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


### ------------- Convert to DSS files ------------- ###
mlvl: int = 0                       #quiet all outputs
HecDss.set_global_debug_level(mlvl)

#obs dss files
obs = obs_fwd_gen[:,:,0]
pt_a = ''
pt_b = [fr'{sites[i]}_ObsFlow' for i in range(len(sites))]
pt_c = 'Flow'
pt_d = fr'{pd.to_datetime(ixx_gen[0]).strftime("%d%b%Y")} - {pd.to_datetime(ixx_gen[-1]).strftime("%d%b%Y")}'
pt_e = '1Day'
pt_f = fr'C:000001|{loc}-USACE_Observed_Flows'

dss_obs_outfile = fr'./out/DSS/{sites}_ObservedFlow_fullPOR.dss'

with HecDss(dss_obs_outfile) as outDssObs:
    for i in range(len(sites)):
        #output each daily-aggregated synthetic observation to a DSS file
        obs_outpath = fr'/{pt_a}/{pt_b[i]}/{pt_c}//{pt_d}/{pt_f}/' 
        times = pd.to_datetime(ixx_gen)
        obs_values = obs[i,:]       
        obsValuesAsList = obs_values.tolist()
        outTimeSeriesForThisTrace = RegularTimeSeries.create(obsValuesAsList, data_type='PER-AVE',times=times, start_date=times[0], interval='1Day', units="cfs", path=obs_outpath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 
        outDssObs.put(outTimeSeriesForThisTrace)
outDssObs.close()

#HEFS hindcasts
n_ens = np.shape(hcst_fit)[3]
n_leads = np.shape(hcst_fit)[2]

pt_a = ''
pt_b = [fr'{sites[i]}_FcstFlow' for i in range(len(sites))]
pt_c = 'Flow'
pt_d = fr'{pd.to_datetime(ixx_fit[0]).strftime("%d%b%Y")} - {pd.to_datetime(ixx_fit[-1]).strftime("%d%b%Y")}'
pt_e = '1Day'
pt_f = r'HEFSHindcast'

fcst_issue_dates = pd.to_datetime(ixx_fit)

dss_hcst_outfile = fr'./out/DSS/{sites}_HEFSHindcastFlow.dss'
n_ens = np.shape(hcst_fit)[3]
with HecDss(dss_hcst_outfile) as outDss:
    for i in range(len(sites)):
        for j in range(len(fcst_issue_dates)):
            #format F_part datetime groups
            fcstIssueDate_short = fcst_issue_dates[j].strftime("%Y%m%d-%H%M")
            fcstIssueDate = fcst_issue_dates[j].strftime("%Y%m%d-%H%M%S")
            #time vector for the RegularTimeSeries file
            times = pd.date_range(fcst_issue_dates[j],fcst_issue_dates[j] + pd.Timedelta(days=n_leads+1),freq='D')
            for k in range(n_ens):
                #format the required elements of the output record name for DSS
                ensembleMemberID = k+1
                # Do the following for _each_ ensemble member in _each_ forecast issued.   Example pathname we are looking to create from TSEnsemble library "//Kanektok.BCAC1/flow/01Nov2013/1Hour/C:000007|T:20131103-1200|V:20131103-120000|/"
                a = pt_a
                b = pt_b[i]
                c = pt_c
                d = pt_d
                e = pt_e # for daily timestep forecast, "6Hour" for 6 hour steps, "1Hour", for hourly, "15Min" for 15Min data... there's a list but this must match.
                f = "C:%06d|T:%s|V:%s|%s" % (
                    ensembleMemberID,       # ensemble member number
                    fcstIssueDate_short, 
                    fcstIssueDate,          # for a study, the T/V values are identical.
                    pt_f             # F part label from json file
                    )
                dssOutPath = "/".join(["",a,b,c,d,e,f,""])          #combine all part labels for the record name
                traceValues = hcst_fit[i,j,:,k]
                traceValuesAsList_fcst = traceValues.tolist()                    #RegularTimeSeries requires the flow values as a list
                traceValuesAsList = [float(obs_fwd_fit[i,j,0])] + traceValuesAsList_fcst
                outTimeSeriesForThisTrace = RegularTimeSeries.create(traceValuesAsList, data_type='PER-AVE',times=times, start_date=fcst_issue_dates[j], interval=e, units="cfs", path=dssOutPath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 
                outDss.put(outTimeSeriesForThisTrace)
outDss.close()

#Perfect forecast (hcst fit period)
n_leads = np.shape(obs_fwd_fit)[2]

pt_a = ''
pt_b = [fr'{sites[i]}_PerfFcstFlow' for i in range(len(sites))]
pt_c = 'Flow'
pt_d = fr'{pd.to_datetime(ixx_fit[0]).strftime("%d%b%Y")} - {pd.to_datetime(ixx_fit[-1]).strftime("%d%b%Y")}'
pt_e = '1Day'
pt_f = r'PerfHindcast'

fcst_issue_dates = pd.to_datetime(ixx_fit)

dss_perf_hcst_outfile = fr'./out/DSS/{sites}_USACEPerfHindcastFlow.dss'
n_ens = np.shape(hcst_fit)[3]
with HecDss(dss_perf_hcst_outfile) as outDss:
    for i in range(len(sites)):
        for j in range(len(fcst_issue_dates)):
            #format F_part datetime groups
            fcstIssueDate_short = fcst_issue_dates[j].strftime("%Y%m%d-%H%M")
            fcstIssueDate = fcst_issue_dates[j].strftime("%Y%m%d-%H%M%S")
            #time vector for the RegularTimeSeries file
            times = pd.date_range(fcst_issue_dates[j],fcst_issue_dates[j] + pd.Timedelta(days=n_leads+1),freq='D')
            # Do the following for _each_ ensemble member in _each_ forecast issued.   Example pathname we are looking to create from TSEnsemble library "//Kanektok.BCAC1/flow/01Nov2013/1Hour/C:000007|T:20131103-1200|V:20131103-120000|/"
            a = pt_a
            b = pt_b[i]
            c = pt_c
            d = pt_d
            e = pt_e # for daily timestep forecast, "6Hour" for 6 hour steps, "1Hour", for hourly, "15Min" for 15Min data... there's a list but this must match.
            f = "C:%06d|T:%s|V:%s|%s" % (
                1,       # ensemble member number
                fcstIssueDate_short, 
                fcstIssueDate,          # for a study, the T/V values are identical.
                pt_f             # F part label from json file
                )
            dssOutPath = "/".join(["",a,b,c,d,e,f,""])          #combine all part labels for the record name
            traceValues = obs_fwd_fit[i,j,:]
            traceValuesAsList = traceValues.tolist()                    #RegularTimeSeries requires the flow values as a list
            outTimeSeriesForThisTrace = RegularTimeSeries.create(traceValuesAsList, data_type='PER-AVE',times=times, start_date=fcst_issue_dates[j], interval=e, units="cfs", path=dssOutPath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 
            outDss.put(outTimeSeriesForThisTrace)
outDss.close()

#Perfect forecast (full obs POR)
pt_a = ''
pt_b = [fr'{sites[i]}_PerfFcstFlow' for i in range(len(sites))]
pt_c = 'Flow'
pt_d = fr'{pd.to_datetime(ixx_gen[0]).strftime("%d%b%Y")} - {pd.to_datetime(ixx_gen[-1]).strftime("%d%b%Y")}'
pt_e = '1Day'
pt_f = r'PerfHindcast_fullPOR'

fcst_issue_dates = pd.to_datetime(ixx_gen)

dss_perf_hcst_outfile = fr'./out/DSS/{sites}_USACEPerfHindcastFlow_fullPOR.dss'
with HecDss(dss_perf_hcst_outfile) as outDss:
    for i in range(len(sites)):
        for j in range(len(fcst_issue_dates)):
            #format F_part datetime groups
            fcstIssueDate_short = fcst_issue_dates[j].strftime("%Y%m%d-%H%M")
            fcstIssueDate = fcst_issue_dates[j].strftime("%Y%m%d-%H%M%S")
            #time vector for the RegularTimeSeries file
            times = pd.date_range(fcst_issue_dates[j],fcst_issue_dates[j] + pd.Timedelta(days=n_leads+1),freq='D')
            # Do the following for _each_ ensemble member in _each_ forecast issued.   Example pathname we are looking to create from TSEnsemble library "//Kanektok.BCAC1/flow/01Nov2013/1Hour/C:000007|T:20131103-1200|V:20131103-120000|/"
            a = pt_a
            b = pt_b[i]
            c = pt_c
            d = pt_d
            e = pt_e # for daily timestep forecast, "6Hour" for 6 hour steps, "1Hour", for hourly, "15Min" for 15Min data... there's a list but this must match.
            f = "C:%06d|T:%s|V:%s|%s" % (
                1,       # ensemble member number
                fcstIssueDate_short, 
                fcstIssueDate,          # for a study, the T/V values are identical.
                pt_f             # F part label from json file
                )
            dssOutPath = "/".join(["",a,b,c,d,e,f,""])          #combine all part labels for the record name
            traceValues = obs_fwd_gen[i,j,:]
            traceValuesAsList = traceValues.tolist()                    #RegularTimeSeries requires the flow values as a list
            outTimeSeriesForThisTrace = RegularTimeSeries.create(traceValuesAsList, data_type='PER-AVE',times=times, start_date=fcst_issue_dates[j], interval=e, units="cfs", path=dssOutPath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 
            outDss.put(outTimeSeriesForThisTrace)
outDss.close()

#Synthetic Forecasts
n_ens = np.shape(syn_fcst_arr)[4]
n_leads = np.shape(syn_fcst_arr)[2]

pt_a = ''
pt_b = [fr'{sites[i]}_FcstFlow' for i in range(len(sites))]
pt_c = 'Flow'
pt_d = fr'{pd.to_datetime(ixx_gen).strftime("%d%b%Y")} - {pd.to_datetime(ixx_gen[-1]).strftime("%d%b%Y")}'
pt_e = '1Day'
pt_f = r'SynFcst_fullPOR'

fcst_issue_dates = pd.to_datetime(ixx_gen)

#def syn_fcst_dsswrite(s):
for s in range(n_samples):
    mlvl: int = 0                       #quiet all outputs
    HecDss.set_global_debug_level(mlvl)
    dss_hcst_outfile = fr'./out/DSS/SynForecast/{sites}_SynHEFHindcastFlow_samp={s+1}.dss'
    with HecDss(dss_hcst_outfile) as outDss:
        for i in range(len(sites)):
            for j in range(len(fcst_issue_dates)):
                #format F_part datetime groups
                fcstIssueDate_short = fcst_issue_dates[j].strftime("%Y%m%d-%H%M")
                fcstIssueDate = fcst_issue_dates[j].strftime("%Y%m%d-%H%M%S")
                #time vector for the RegularTimeSeries file
                times = pd.date_range(fcst_issue_dates[j],fcst_issue_dates[j] + pd.Timedelta(days=n_leads+1),freq='D')
                for k in range(n_ens):
                    #format the required elements of the output record name for DSS
                    ensembleMemberID = k+1
                    # Do the following for _each_ ensemble member in _each_ forecast issued.   Example pathname we are looking to create from TSEnsemble library "//Kanektok.BCAC1/flow/01Nov2013/1Hour/C:000007|T:20131103-1200|V:20131103-120000|/"
                    a = pt_a
                    b = pt_b[i]
                    c = pt_c
                    d = pt_d
                    e = pt_e # for daily timestep forecast, "6Hour" for 6 hour steps, "1Hour", for hourly, "15Min" for 15Min data... there's a list but this must match.
                    f = "C:%06d|T:%s|V:%s|%s" % (
                        ensembleMemberID,       # ensemble member number
                        fcstIssueDate_short, 
                        fcstIssueDate,          # for a study, the T/V values are identical.
                        pt_f             # F part label from json file
                        )
                    dssOutPath = "/".join(["",a,b,c,d,e,f,""])          #combine all part labels for the record name
                    traceValues = syn_fcst_arr[s,i,j,:,k]
                    traceValuesAsList_fcst = traceValues.tolist()                    #RegularTimeSeries requires the flow values as a list
                    traceValuesAsList = [float(obs_fwd_gen[i,j,0])] + traceValuesAsList_fcst
                    outTimeSeriesForThisTrace = RegularTimeSeries.create(traceValuesAsList, data_type='PER-AVE',times=times, start_date=fcst_issue_dates[j], interval=e, units="cfs", path=dssOutPath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 
                    outDss.put(outTimeSeriesForThisTrace)
    outDss.close()

#Parallelize the output of each synthetic observation and synthetic forecast file
"""
with joblib_progress("Writing SynForecast DSS files...", total=n_samples):
    par_out = Parallel(n_jobs=workers)(
    delayed(syn_fcst_dsswrite)(i) for i in range(n_samples))
"""


####################################################################END#######################################################################