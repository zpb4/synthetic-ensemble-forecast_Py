import numpy as np
import pandas as pd 
import scipy.stats as sstat
import matplotlib.pyplot as plt
import matplotlib
from numba import njit

cfs_to_afd = 2.29568411*10**-5 * 86400
afd_to_cfs = 1 / cfs_to_afd

def water_day(d, is_leap_year):
    # Convert the date to day of the year
    day_of_year = d.timetuple().tm_yday
    
    # For leap years, adjust the day_of_year for dates after Feb 28
    if is_leap_year and day_of_year > 59:
        day_of_year -= 1  # Correcting the logic by subtracting 1 instead of adding
    
    # Calculate water day
    if day_of_year >= 274:
        # Dates on or after October 1
        dowy = day_of_year - 274
    else:
        # Dates before October 1
        dowy = day_of_year + 91  # Adjusting to ensure correct offset
    
    return dowy

def split_return(x,match):
    spl_tex = x.split('.')
    out = False
    if spl_tex[0] == match:
        out = True
    return out

def declust_evts_extract(Q,n_evts,sep):
    rnk_data = sstat.rankdata(-Q)
    srt_rnks_idx = np.argsort(rnk_data)[0:n_evts*10]

    vec = np.zeros(len(srt_rnks_idx))
    for i in range(len(srt_rnks_idx)):
        vec[i]=np.where(rnk_data==np.min(rnk_data[max((srt_rnks_idx[i]-sep),0):min(len(Q),(srt_rnks_idx[i]+sep))]))[0][0]
    declust_evts=np.unique(vec)
    sel_idx=np.argsort(-Q[np.int64(declust_evts)])
    evt_idx = np.int64(declust_evts[sel_idx])[0:n_evts]
    
    return evt_idx

def rtn_per_peakflow_fun(rtns_interp,pk_flows_interp,rtns):
    out = np.interp(np.log(rtns),np.log(rtns_interp),np.log(pk_flows_interp))
    return np.exp(out)

def fig_title(
    fig: matplotlib.figure.Figure, txt: str, loc=(0.5,0.98), fontdict=None, **kwargs
):
    """Alternative to fig.suptitle that behaves like ax.set_title.
    DO NOT use with suptitle.

    See also:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html
    https://stackoverflow.com/a/77063164/8954109
    """
    if fontdict is not None:
        kwargs = {**fontdict, **kwargs}
    if "fontsize" not in kwargs and "size" not in kwargs:
        kwargs["fontsize"] = plt.rcParams["axes.titlesize"]

    if "fontweight" not in kwargs and "weight" not in kwargs:
        kwargs["fontweight"] = plt.rcParams["figure.titleweight"]

    if "verticalalignment" not in kwargs:
        kwargs["verticalalignment"] = "top"
    if "horizontalalignment" not in kwargs:
        kwargs["horizontalalignment"] = "center"

    # Tell the layout engine that our text is using space at the top of the figure
    # so that tight_layout does not break.
    # Is there a more direct way to do this?
    fig.suptitle(" ")
    text = fig.text(loc[0], loc[1], txt, transform=fig.transFigure, in_layout=True, **kwargs)

    return text

#------------------------------------------------------------------
# numba compatible random arr sampling function
#-----------------------------------------------------------------
@njit
def numba_choice(vec, size=1, replace=True):
    """
    Numba-compatible replacement for numpy.random.choice(range(n), size, replace)
    
    Parameters:
        n (int): Population size (0 to n-1 will be sampled)
        size (int): Number of samples to draw
        replace (bool): Sample with or without replacement
    
    Returns:
        np.ndarray: Array of sampled integers
    """
    n = len(vec)
    
    if n <= 0:
        raise ValueError("n must be positive")
    if size < 0:
        raise ValueError("size must be non-negative")
    if not replace and size > n:
        raise ValueError("Cannot take a larger sample than population when 'replace=False'")
    
    result = np.empty(size, dtype=np.float32)
    
    if replace:
        # Simple random integers with replacement
        for i in range(size):
            result[i] = vec[np.random.randint(0, n)]
        
    else:
        # Fisher–Yates shuffle for sampling without replacement
        arr = np.arange(n)
        for i in range(size):
            j = i + np.random.randint(0, n - i)
            arr[i], arr[j] = arr[j], arr[i]
            result[i] = vec[arr[i]]
    
    return result

#------------------------------------------------------------------
# climatological ensemble function)
#-----------------------------------------------------------------
@njit
def calc_climo_ensemble(obs,
                        n_ens,
                        n_leads,
                        dowy,
                        ):
    """
    Compute mean CRPS by lead time for forecasts at a single site
    Requires pre-indexed optimization obs and forecast index
    Forecasts need to be for the same slice period as the pre-indexed obs
    The n_opt_obs is the subset of observations (generally upper quantile) being used for optimization
    
    Inputs
    ------
    obs : a n_sites x n_obs vector of streamflow observations
    n_ens : number of ensemble members needed
    n_leads : number of lead times needed
    dowy : a day of water year vector matching the obs input

    Returns
    -------
    climo_array : an n_sites x n_obs x n_leads array of climatological forecasts
    """
    n_sites,n_obs = np.shape(obs)
    
    climo_array = np.full((n_sites,n_obs,n_leads,n_ens),np.nan)
    
    for s in range(n_sites):
        for o in range((n_obs-n_leads)):
            fcst_vec = dowy[(o+1):(o+n_leads+1)]
            for i in range(len(fcst_vec)):
                ens_idx = np.where(dowy == fcst_vec[i])[0]
                ens_samps = obs[s,ens_idx]
                
                if len(ens_samps) >= n_ens:
                    out_samps = numba_choice(ens_samps,size=n_ens,replace=False)
                else:
                    out_samps = numba_choice(ens_samps,size=n_ens,replace=True)
                
                climo_array[s,o,i,:] = out_samps

    return climo_array


#------------------------------------------------------------------
# Numba compatible ensemble crps calculation function (not compatible with properscoring package)
#-----------------------------------------------------------------

@njit
def ensemble_crps(ensemble,tgt):
    """
    Return the ensemble CRPS value given an ensemble prediction and a target predictand
    """
    ne = len(ensemble)
    term1 = (1/ne) * np.sum(np.abs(ensemble - tgt))
    
    #for loop to calculate second ecrps term per Wilks 2019
    term2 = np.zeros((ne,ne))
    for i in range(ne-1):
        for j in range(i+1,ne):
            term2[i,j] = np.abs(ensemble[i]-ensemble[j])
    
    term2_result = (1 / (ne * (ne-1))) * np.sum(term2)
    out = term1 - term2_result
    
    return out

#------------------------------------------------------------------
# Ensemble ranking helper function
#-----------------------------------------------------------------
@njit
def ensemble_rank(obs, ens):
    """
    Return the rank of obs relative to ensemble ens.
    Rank = number of ensemble members <= obs, in [0, K].
    """
    ens = np.asarray(ens)
    if not np.isfinite(obs):
        return np.nan
    ens = ens[np.isfinite(ens)]
    if ens.size == 0:
        return np.nan
    return np.sum(ens <= obs)

#------------------------------------------------------------------
# Function to compute mean crps across lead times for a single forecast sample
#-----------------------------------------------------------------
@njit
def compute_fcst_crps(
    forecasts,
    obs_fwd,
    ):
    """
    Compute mean CRPS by lead time for forecasts at a single site
    Requires pre-indexed optimization obs and forecast index
    Forecasts need to be for the same slice period as the pre-indexed obs
    The n_opt_obs is the subset of observations (generally upper quantile) being used for optimization
    
    Inputs
    ------
    forecasts : an n_sites x n_obs x n_leads x n_ens array
    obs_fwd : a n_sites x n_obs x (n_leads + 1) fwd-looking obs array (perfect forecast)
    forc_idx : an array n_leads x n_opt_obs (only used if forecast is not pre subsetted)
    sset_forecast : if True, forecast is already subsetted to a flattened vector of forecast indices for the n_opt_obs subset (i.e. syn_gen_opt function)
                    if False, forecast is not subsetted (i.e. parent hindcast dataset)

    Returns
    -------
    crps_array : an n_sites x n_obs x n_leads array of CRPS values
    """
    n_sites,n_obs,n_leads,n_ens = np.shape(forecasts)
        
    crps_array = np.full((n_sites,n_obs,n_leads),np.nan)
    
    for s in range(n_sites):
        for l in range(n_leads):
            for o in range(n_obs):
                crps_array[s,o,l] = ensemble_crps(forecasts[s,o,l,:],obs_fwd[s,o,(l+1)])

    return crps_array

########################################################END################################################