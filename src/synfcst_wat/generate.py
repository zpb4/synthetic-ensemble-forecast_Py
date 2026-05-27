# general
import os
import sys
from pathlib import Path

# config files
import json
from synfcst.model_params import ModelParams
#from synfcst.io import DssHydrographs # TODO!

# others
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
from hecdss import HecDss
from hecdss import RegularTimeSeries
from util import water_day
import json
import random
import calendar

import time

#numba compatible function import 
from syn_gen_hec_wat_fra import syn_gen_hec_wat_fra,obs_fwd_fun

# reduce logging
HecDss.set_global_debug_level(2) 


class WatCompute:
    """ class to wrap config file to get properties

    this mostly wraps the .json file, but gives us a tiny bit of abstraction
    """
    def __init__(self, filename:str):
        """ constructor to read WAT model configuration for this compute
        """
        data = json.load(open(filename, 'r'))

        # general WAT settings
        self.watershed = data['Outputs']['Watershed Directory']
        self.outFPart = data['Outputs']['F Part'] 
        self.runDirectory = data['Outputs']['Run Directory']  # lifecycle folder
        self.outDirectory = data['Outputs']['Out Directory']  # where to write data, may be the same.
        self.simName = data['Outputs']['Simulation Name']
        self.simfile = str(Path(self.outDirectory, "%s.dss" % self.simName)) # lifecycle dss
        # set FRM settings
        #retrieve key json config elements
        self.realization = data['Indices']['Realization Number']
        self.lifecycle = data['Indices']['Lifecycle Number']
        self.event = data['Indices']['Event Number']
        self.nEventsPerLifecycle = data['Indices']["Events Per Lifecycle"]

        self.realization_seed = data['Randoms']['Realization Random']
        self.lifecycle_seed = data['Randoms']['Lifecycle Random']
        self.event_seed = data['Randoms']['Event Random']

        self.lifecycle_compute = False  # vs false if we want to do per-event
        
        # model locations
        # TODO: handle more than one location!
        self.location = data["Locations"][0]

class GeneratorSettings:
    # DEFAULT_MAX_WORKERS = 10

    def __init__(self, max_workers):
        self.workers = min(10, max_workers) # use number of cores limited by events

class SynFcstGenerator:
    """ main class for WAT synthetic forecasts
    """
    def __init__(self, compute_options:WatCompute, model_parameters:ModelParams):
        """ loads up the generator but does not compute
        """
        self.compute_options = compute_options
        self.model_parameters = model_parameters
        # can't use more than n workers
        self._genconfig = GeneratorSettings(self.compute_options.nEventsPerLifecycle)




    def _prepare_files(self):
        # THIS SHOULD BE OVERLOADED FOR EACH COMPUTE TYPE OR PROVIDED?
        # this function ensures the input and output directories exist
        try:
            watershed_dir = Path(self.compute_options.watershed)
            self.opt_dir = watershed_dir / "synfcst"  # set this to the watershed's synfcst model directory
            self.data_dir = watershed_dir / "synfcst" # this could be wat's shared instead
            self.out_dir = Path(self.compute_options.outDirectory)
            os.makedirs(out_dir, exist_ok=True)
        except:
            print("ERROR: File system issue.")
        return True





    def _load_generator_data(self):
        # read in model fitting data and hindcast/obs_fwd data
        mp = self.model_parameters
        ## Loading data
        obs_data = np.load(self.data_dir / self.model_parameters.files["obs_file"], allow_pickle=True)
        
        # unpack obsdata
        #hefs array [n_sites x n_obs x n_leads x n_ens]
        hcst = obs_data['hefs']
        #obs forward (perfect forecast array) [n_sites x n_obs x n_leads]  **note: n_leads is 1 longer than hefs_array because col 0 is day t observations in obs fwd
        obs_fwd = obs_data['obs_fwd']
        #site index  
        sites = obs_data['sites']
        
        #date/time vectors
        obs_fwd_dtg = obs_data['obs_fwd_dtg']
        hcst_dtg = obs_data['hefs_dtg']
        #bad forecast days
        bad_forcs = obs_data['missing_dates']
        
        
        ## fit file
        self.opt_pars = pickle.load(open(self.opt_dir / mp.files["fit_file"], 'rb'), encoding='latin1')
        #print(opt_pars)

        keysite_idx = np.where(sites==mp.keysite_label)[0][0]      #set keysite index for syn_gen code (indexing site to preserve spatial correlations)
        site_idx = np.where(sites==mp.gen_site)[0][0]              #set site index for syn_gen code    (site where synthetic forecasts are being generated)

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 4. Set datetime indices and clean data inputs for synthetic generation

        #set datetime array for fit and gen periods
        #ensure fit dataset includes all dates in both hefs and obs_fwd date/time groups
        fit_start = max(hcst_dtg[0],obs_fwd_dtg[0])
        fit_end = min(hcst_dtg[-1],obs_fwd_dtg[-1])
        
        # convert to avoid issues!
        obs_fwd_dtg = pd.to_datetime(obs_fwd_dtg,utc=True).to_numpy(dtype="datetime64[us]")
        hcst_dtg = pd.to_datetime(hcst_dtg,utc=True).to_numpy(dtype="datetime64[us]")
        bad_forcs = pd.to_datetime(bad_forcs,utc=True).to_numpy(dtype="datetime64[us]")

        ixx_fit = pd.date_range(fit_start, fit_end, freq="D").to_numpy(dtype="datetime64[us]")

        print("full range available for hindcast: %s to %s" % (fit_start, fit_end))

        fit_dates = pd.date_range(mp.st_fit, mp.en_fit, freq="D").to_numpy(dtype="datetime64[us]")
        if mp.fit_gen_strategy == 'specify':
            if fit_dates[0] < ixx_fit[0] or fit_dates[-1] > ixx_fit[-1]:
                raise ValueError("Specified fit dates outside available hindcast/obs_fwd dataset")
            ixx_fit = fit_dates
        print("\t using hindcasts from %s to %s" % (fit_start, fit_end))


        #index synthetic generation arrays
        #arrays for calibration (fit) dataset observation and hindcast pairs
        obs_fwd_fit = obs_fwd[site_idx,np.isin(obs_fwd_dtg,ixx_fit),:]
        hcst_fit = hcst[site_idx,np.isin(hcst_dtg,ixx_fit),:,:]

        #remove any bad forecast days from the calibration (fit) datasets
        if mp.remove_bad_forecasts:
            rmv_idx = np.isin(ixx_fit,bad_forcs)
            obs_fwd_fit = np.delete(obs_fwd_fit,rmv_idx,axis=0)
            hcst_fit = np.delete(hcst_fit,rmv_idx,axis=0)

        #to prevent instabilities in generation (e.g. divide by zero), set all obs and obs_fwd zeros to min non-zero value
        min_nzero_vec = obs_fwd[obs_fwd>0.0]
        min_nzero = min(min_nzero_vec)
        obs_fwd_fit[obs_fwd_fit==0.0] = min_nzero

        #extract the ensemble number value from the hindcast dataset
        self.n_ens = np.shape(hcst)[3]

        ## These are more metadata than anything
        self.fit_sites = sites
        self.fit_dates = fit_dates
        self.key_site = mp.keysite_label
        self.min_nzero = min_nzero
        ## TODO: wrap most of these in a data model class?
        self.ixx_fit = ixx_fit
        self.obs_fwd_dtg = obs_fwd_dtg
        self.obs_fwd_fit = obs_fwd_fit
        self.hcst_dtg = hcst_dtg
        self.hcst_fit = hcst_fit
        self.obs_fwd = obs_fwd
        return True
        
        
        
    def _create_daily_means(self):
        # optionally, create daily means to extend hydrograph

        #calculate a daily mean across obs_data (used to concatenate additional days to the synthetic events in section 5 below)
        obs_dtg = pd.to_datetime(self.obs_fwd_dtg)
        dowy_obs = np.array([water_day(d,calendar.isleap(d.year)) for d in obs_dtg])
        dly_mean = np.zeros(np.max(dowy_obs))
        site_idx = np.where(self.fit_sites==self.model_parameters.gen_site)[0][0]
        for i in range(len(dly_mean)):
            dowy_idx = np.where(dowy_obs == i)
            dly_mean[i] = np.mean(self.obs_fwd[site_idx,dowy_idx,0])
        # Screen out missing values
        dly_mean[dly_mean < 0] = 0
        self.dly_mean = dly_mean
        
        return True
        
        
        
   
    def _load_input_hydrographs(self, hydrograph_source):
        # TODO - I don't think we need to do this separately?
        
        # read in the hydrographs for generation, prepare them
        # hydrograph source is an object wrapping the DSS file or other input
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 5. Read the HEC-WAT hydrologic sample DSS output file, extract 'n_events' synthetic hydrologic event, and format for synthetic forecast generation

        #read in data
        """
        #current dss file path and data for testing; these data should come from the json file or be provided as a script input from the 'callSynForecast' function
        """
        dss_file = Path(self.compute_options.outDirectory, "%s.dss" % self.compute_options.simName)
        #load the DSS file of synthetic events for first event to extract key info
        dss = HecDss(str(dss_file))
        
        dss_pathname = self.compute_options.location["dss_pathname"]

        #extract information elements from the DSS file (note: much of this is redundant with the procedure in the loop below, but left for now for demonstration purposes)
        # TODO: remove this block - still needed to create `flow_daily` variable
        # TODO: move daily aggregation to WAT script to handle daily conversion prior to passing into this script
        #print(dss_pathname)
        parts = dss_pathname.split('/')[1:]
        alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)][:len(parts)]
        part_dict = {alphabet[i]:parts[i] for i in range(len(parts))}
        # deal with DSS v6 pathnames coming in - likely WAT model-linking/scripting bug
        part_dict["e"] = part_dict["e"].replace("MIN", "Minute")
        part_dict["e"] = part_dict["e"].replace("MON", "Month")
        part_dict["e"] = part_dict["e"].replace("HR", "Hour")
        dss_pathname = "/".join(["", part_dict["a"], part_dict["b"], part_dict["c"], "", part_dict["e"], part_dict["f"],""])

        data = dss.get(dss_pathname)
        #print(data.start_date,data.units,data.data_type)
        dtg = pd.to_datetime(data.times)
        flow = data.values

        self.flow_daily = pd.Series(flow, index=dtg)
        return True
    
    def _load_event_hydrographs(self, hydrograph_source):
        # this reads the events from the hydrograph source and creates obs_fwd_gen_mat matrix
        # READ DSS
        max_lds = self.model_parameters.max_lds
        
        dss_file = Path(self.compute_options.outDirectory, "%s.dss" % self.compute_options.simName)
        #load the DSS file of synthetic events for first event to extract key info
        dss = HecDss(str(dss_file))
        
        dss_pathname = self.compute_options.location["dss_pathname"]
        parts = dss_pathname.split('/')[1:]
        alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)][:len(parts)]
        part_dict = {alphabet[i]:parts[i] for i in range(len(parts))}
        # deal with DSS v6 pathnames coming in - likely WAT model-linking/scripting bug
        part_dict["e"] = part_dict["e"].replace("MIN", "Minute")
        part_dict["e"] = part_dict["e"].replace("MON", "Month")
        part_dict["e"] = part_dict["e"].replace("HR", "Hour")
        dss_pathname = "/".join(["", part_dict["a"], part_dict["b"], part_dict["c"], "", part_dict["e"], part_dict["f"],""])
        self.part_dict = part_dict
        self.dss_pathname = dss_pathname

        #extract each of the synthetic events from the DSS file and save to an 'obs_fwd' configured array
        #array to store obs_fwd arrays
        # TODO: this assumes that all events are the same length?
        n_events = self.compute_options.nEventsPerLifecycle
        obs_fwd_gen_mat = np.full((n_events,len(self.flow_daily),max_lds+1),np.nan,dtype=np.float64)
        #print(obs_fwd_gen_mat.shape)
        #dictionary to store the date/time sequence for each synthetic event
        fcst_issue_dates = {}
        # TODO: this should be getting list of events from compute_options instead of assuming it starts at 1.
        for i in range(n_events):
            i_idx = i+1
            evt_num = f'{i_idx:06}'
            event_f_part = "C:%s|%s" % (evt_num, part_dict['f'].split("|")[-1])
            event_dss_pathname = "/".join(["", part_dict['a'], part_dict['b'], part_dict['c'], "", part_dict['e'], event_f_part, ""])
            print(event_dss_pathname)
            data = dss.get(event_dss_pathname)
            dtg = pd.to_datetime(data.times)
            flow = data.values  #/ 1000
            print(("event %d: dtg: %d" % (i, len(dtg))))

            # TODO: we shouldn't need to do this any more with WAT 1DAY inputs
            #configure data to daily timeseries aggregated across 12 - 12 UTC
            #dtg_shift = dtg + pd.Timedelta(hours=12)
            #flow_series = pd.Series(flow, index=dtg_shift)
            #flow_daily = flow_series.resample("D").mean()
            #reindex = self.flow_daily.index #- pd.Timedelta(hours=12)
            flow_daily = pd.Series(flow, index=dtg)
            print(("event %d: flow_daily: %d" % (i, len(flow_daily))))

            # TODO: move this concat step into before the generation
            #add 'max_lds' number of days of daily mean to each synthetic obs sequence to support 'obs_fwd' generation across the entire synthetic sequence
            ext_dates = pd.date_range(pd.to_datetime(flow_daily.index[-1]) + pd.Timedelta(hours=12),pd.to_datetime(flow_daily.index[-1]) + pd.Timedelta(hours=(max_lds-1)*24+12),freq='D')
            dowy_concat = np.array([water_day(d,calendar.isleap(d.year)) for d in ext_dates])
            concat_flows = self.dly_mean[dowy_concat]
            flow_daily_concat = np.concat((flow_daily.values,concat_flows))
            #print(("event %d: flow_daily_concat: %d" % (i, len(flow_daily_concat))))     
            #print(flow_daily_concat)
            fcst_issue_dates[evt_num] = flow_daily.index
            obs_fwd_gen_mat[i,:,:] = obs_fwd_fun(flow_daily_concat,max_lds)

        dss.close()
        self.fcst_issue_dates = fcst_issue_dates
        self.obs_fwd_gen_mat = obs_fwd_gen_mat
        return True
        
    def _generate(self):
        #generate in memory, don't write outputs to disk yet
        fwd_gen_mat = self.obs_fwd_gen_mat
        
        # set zeros to non-zero to avoid instabilities in generation
        fwd_gen_mat[fwd_gen_mat==0.0] = self.min_nzero

        ## TODO: How much of this is also in model_params?  Not all of it.  Maybe those are overrides?
        kk          = self.opt_pars['kk']       
        knn_pwr     = self.opt_pars['knn_pwr'] 
        scale_pwr   = self.opt_pars['scale_pwr'] 
        hi          = self.opt_pars['hi'] 
        lo          = self.opt_pars['lo']       # 1.4 
        sig_a       = self.opt_pars['sig_a'] 
        sig_b       = self.opt_pars['sig_b'] 

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 6. Setup the synthetic forecast generation function and run in parallel to generate synthetic forecasts for each HEC-WAT synthetic event
        #function to generate synthetic forecast samples and save an output .npz file for each sample
        def syn_gen_par(i):
            out = syn_gen_hec_wat_fra(
                seed=int(self.compute_options.event_seed*1000),                    
                kk=kk,                                  
                knn_pwr=knn_pwr,                              
                scale_pwr=scale_pwr,                            
                hi=hi,                                     
                lo=lo,                                     
                sig_a=sig_a,                                  
                sig_b=sig_b,                                 
                ixx_fit=self.ixx_fit,
                obs_fwd_fit=self.obs_fwd_fit,
                obs_fwd_gen=fwd_gen_mat[i,:,:],
                hcst_fit=self.hcst_fit
            )
            
            lc = self.compute_options.lifecycle
            #outfile = './syn-forecast_site=%s_lifecycle=%s_event=%s.npz' %(self.model_parameters.gen_site,lc,i+1)
            #np.savez(out_dir/outfile,syn_fcst=out)

            return out

        #for i in range(n_events):
        #    obs_mat_shape = fwd_gen_mat[i,:,:].shape
        #    print(("event %d: obs_mat_shape: %s" % (i, str(obs_mat_shape))))

        #run synthetic generation code in parallel; par_out is a list of length 'n_events' where each list element is a n_obs x n_leads+1 x n_ens array
        #Note: the array is configured to have index 0 in dimension 2 as the day t observation for compatibility with HEC-WAT; this is why dimension 2 is n_leads+1
        
        workers = self._genconfig.workers
        n_events = self.compute_options.nEventsPerLifecycle
        
        compute_start_time = time.time()
        par_out = Parallel(n_jobs=workers)(delayed(syn_gen_par)(np.array(i)) for i in range(n_events))
        self.gen_out = par_out
        
        # record times
        self.finished_gen_out_time = time.ctime()
        self.gen_out_compute_time = time.time() - compute_start_time
        print("compute finished at %s, took %0.2f seconds" % (self.finished_gen_out_time, self.gen_out_compute_time))
        return True

    def _output_hydrographs(self, outputs):
        # TODO: for output_sink in outputs: outputs.write(...)
        # write to DSS, CSV, NPZ, depending on output writers provided
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 7. Output both the aggregated daily obs file and the synthetic forecast sequence to a DSS file for each HEC-WAT synthetic event
        max_lds = self.model_parameters.max_lds
        n_events = self.compute_options.nEventsPerLifecycle
        
        #define a function for Parallel processing 
        def dss_out_par(i):
            i_idx = i+1
            evt_num = f'{i_idx:06}' #0-padded event number
            #outfiles for the synthetic forecast and daily synthetic obs
            #dss_outfile = "%s-%s-event_%s" % (self.model_parameters.gen_site, "fcst", evt_num)
            dss_outfile = "syn-fcst-event_%s" % evt_num
            outdss = Path(self.compute_options.outDirectory, dss_outfile)
            
            fPartSuffix = self.compute_options.outFPart.split("|")[-1]
            fPartSuffix = "" # TODO Fix this

            part_dict = self.part_dict
            #output each daily-aggregated synthetic observation to a DSS file
            obs_outpath = "/".join(["", part_dict["a"], part_dict["b"], part_dict["c"] + "-synobs", "", "1Day", "C:%s|%s" % (evt_num, fPartSuffix), ""])
            times = self.fcst_issue_dates[evt_num]
            obs_values = self.obs_fwd_gen_mat[i,:,0] #* 1000      #reset to cfs
            obsValuesAsList = obs_values.tolist()
            outTimeSeriesForThisTrace = RegularTimeSeries.create(obsValuesAsList, 
                data_type='PER-AVER',times=times, start_date=times[0], interval='1Day', units="cfs", path=obs_outpath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 

            #output the synthetic forecast sequence for each synthetic event to a DSS file for HEC-WAT ingest
            with HecDss(str(outdss)) as outDss:
                outDss.put(outTimeSeriesForThisTrace)
                for j in range(len(self.fcst_issue_dates[evt_num])):
                    #format F_part datetime groups
                    fcstIssueDate_short = self.fcst_issue_dates[evt_num][j].strftime("%Y%m%d-%H%M")
                    fcstIssueDate = self.fcst_issue_dates[evt_num][j].strftime("%Y%m%d-%H%M%S")
                    #time vector for the RegularTimeSeries file
                    times = pd.date_range(self.fcst_issue_dates[evt_num][j],self.fcst_issue_dates[evt_num][j] + pd.Timedelta(days=max_lds),freq='D')
                    for k in range(self.n_ens):
                        #format the required elements of the output record name for DSS
                        ensembleMemberID = k+1
                        # Do the following for _each_ ensemble member in _each_ forecast issued.   Example pathname we are looking to create from TSEnsemble library "//Kanektok.BCAC1/flow/01Nov2013/1Hour/C:000007|T:20131103-1200|V:20131103-120000|/"
                        a = part_dict['a']
                        b = part_dict['b']
                        c = part_dict['c']
                        d = self.fcst_issue_dates[evt_num][0].strftime("%d%b%Y")
                        ePart = "1Day" # for daily timestep forecast, "6Hour" for 6 hour steps, "1Hour", for hourly, "15Min" for 15Min data... there's a list but this must match.
                        fPart = "C:%06d|T:%s|V:%s|%s" % (
                            ensembleMemberID,       # ensemble member number
                            fcstIssueDate_short, 
                            fcstIssueDate,          # for a study, the T/V values are identical other than _seconds_
                            fPartSuffix                   # F part label from json file
                            )
                        dssOutPath = "/".join(["",a,b,c,d,ePart,fPart,""])          #combine all part labels for the record name
                        traceValues = self.gen_out[i][j,:,k] #* 1000                      #reset values to kcfs
                        traceValuesAsList = traceValues.tolist()                    #RegularTimeSeries requires the flow values as a list
                        outTimeSeriesForThisTrace = RegularTimeSeries.create(traceValuesAsList, 
                            data_type='PER-AVER',times=times, start_date=self.fcst_issue_dates[evt_num][j], interval=ePart, units="cfs", path=dssOutPath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 
                        outDss.put(outTimeSeriesForThisTrace)
            outDss.close()

        #Parallelize the output of each synthetic observation and synthetic forecast file
        # TODO - no need to parallelize this
        Parallel(n_jobs=1)(delayed(dss_out_par)(i) for i in range(n_events))
 
        return True


    def compute(self):
        # This is a high level method intended to work through the required steps

        # create output locations
        self._prepare_files()
        
        # read in the data required for generation, this could be done _once_
        self._load_generator_data()
        
        # if self.model_parameters.hydrograph_extension:
        self._create_daily_means()
        
        # this reads the data on which we are going to generate forecasts
        dss_source = None #TODO: DssHydrographs()
        self._load_input_hydrographs(dss_source)
        self._load_event_hydrographs(dss_source)
        
        # this does the generation
        self._generate()
        
        # writes the data back out
        npz_source = None #TODO: NpzHydrographs()
        self._output_hydrographs(outputs=[dss_source, npz_source])
        
        # if compute succeeds:
        return True
