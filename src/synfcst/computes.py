# general
import os
import sys
from pathlib import Path

# config files
import json

# others
from pathlib import Path
from datetime import datetime

class StandaloneCompute:
    """ class to wrap a config file, does not expect to run in WAT
    TODO: implement sources and sinks for hydrograph data as part of these computes

    """
    def __init__(self, filename:str):
        data = json.load(open(filename, 'r'))
    
    def get_temp_dir():
        return ""
    
    def get_output_dir():
        return ""
    
    def get_input_hydrographs():
        return ""
    
    def get_output_sink():
        return None
    
    

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

        self.lifecycle_compute = data['LifecycleCompute'] # vs false if we want to do per-event
        self.n_events = 1 # default
        if self.lifecycle_compute:
            self.n_events = self.nEventsPerLifecycle

        # model locations
        # TODO: handle more than one location!
        self.location = data["Locations"][0]
