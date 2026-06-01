# for general tools needed
from pathlib import Path
from datetime import datetime

# for numpy and pandas data
import numpy as np
import pandas as pd

# for DSS data
from hecdss import HecDss
from hecdss import RegularTimeSeries

# for CSV data

class GenericHydrographIO:
    def __init__(self, filename):
        pass

# Concept: this would provide a class that deals with full ensemble sets once
# class GenericForecastIO:
#    def __init__(self, filename):
#        pass

class DssPathname():
    def __init__(self, pathString):
        self._path = pathString
        self._path_parts()

    def __str__(self):
        self._update()
        return self._path

    def _path_parts(self):
        parts = dss_pathname.split('/')[1:]        
        self._path_dict = {'abcdef'[i]:parts[i] for i in range(len(parts)-1)}

    def _join_path_parts(self)
        self._path = "/".join(["", part_dict["a"], part_dict["b"], part_dict["c"], "", part_dict["e"], part_dict["f"],""])

    def _update(self):
        """
        we always call this after changing something.
        """
        self._path = self._join_path_parts()

    def set_part(self, part, string):
        ## DSS v6 vs DSS 7 pathnames
        # part_dict["e"] = part_dict["e"].replace("MIN", "Minute")
        # part_dict["e"] = part_dict["e"].replace("MON", "Month")
        # part_dict["e"] = part_dict["e"].replace("HR", "Hour")
        self._path_dict[part.lower()] = string
        self._update()
    
    def get_part(self, part):
        return self._path_dict[part.lower()]



class DssEnsemblePathname(DssPathname):
    def __init__(self):
        self.ensemble_id = "C:%06d" % 0
        self._issue_date = datetime64()
        self._version_date = datetime64()
        self.f_part_suffix = ""

    def _update(self):
        """
        we always call this after changing something.
        """
        f_part = self._new_fpart()
        super.set_part("F", f_part)
        super._update()

    def _new_fpart(self)
        return "|".join(self.ensemble_id, self.issue_date_string(), self.version_date_string(), self.f_part_suffix)

    def issue_date_string(self):
        return "T:%s" % self._issue_date.strftime("%Y%m%d-%H%M")
        
    def version_date_string(self):
        return "V:%s" % self._issue_date.strftime("%Y%m%d-%H%M%S")

    def set_issue_date(self, issue_date):
        self._issue_date = issue_date
        self._update()

    def set_version_date(self, version_date):
        self._version_date = version_date
        self._update()

    def set_ensemble_id(self, ensembleMemberID):
        self.ensemble_id = "C:%06d" % ensembleMemberID
        self._update()

# this class 
class DssHydrographIO(GenericHydrographIO):
    def __init__(self, dssFilename):
        self.filename = dssFilename

    def read(self, pathnames):
        try:
            dss = HecDss(str(dss_file))
            for path in pathnames:
                dss.get(pathname)
                # not sure if this is required:
                dtg = pd.to_datetime(data.times)
                flow = data.values
                yield pd.Series(flow, index=dtg)
        finally:
            dss.done()

    def write(self, pathname, times, values):
        ## TODO: Make this more efficent for many timeseries, one reason to create the forecast IO class?
        try:
            dss = HecDss(str(dss_file))
            out = RegularTimeSeries.create(values, times=times, start_date=times[0], 
            path=pathname, data_type='PER-AVER', interval=ePart, units="cfs")   
            dss.put(out)
        finally:
            dss.done()

# class DssForecastIO(GenericForecastIO):
#     def __init__(self, dssFilename):
#         pass
#     def read(self, pathname, ensembles=[]):
#         for ens_id in ensembles:
#             super.read(...)
#     def write(self, pathname):
#         pass

class NpzHydrographIO(GenericHydrographIO):
    def __init__(self, dssFilename):
        pass
    def read(self, pathname):
        with()
        pass
    def write(self, pathname):
        pass

# class NpzForecastIO(GenericForecastIO):
#     def __init__(self, dssFilename):
#         pass
#     def read(self, pathname):
#         pass
#     def write(self, pathname):
#         pass 

class CsvHydrographIO(GenericHydrographIO):
    def __init__(self, dssFilename):
        pass
    def read(self, pathname):
        pass
    def write(self, pathname):
        pass 

# class CsvForecastIO(GenericForecastIO)    
#     def __init__(self, dssFilename):
#         pass
#     def read(self, pathname):
#         pass
#     def write(self, pathname):
#         pass 