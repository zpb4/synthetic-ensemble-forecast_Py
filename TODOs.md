# WAT 
- refactor into WAT generator as a module (in progress, more to do)
- connect to WAT EFP plugin (done)
- update WAT jython script to call Python not R (done)
- WAT jython script: write per-aver data for forecasts to DSS (done, SynFcstPreProcessor.py)
- WAT jython script: pass event DSS file into script (done in synfcst)
- WAT jython script:  identify datalocations for script (done, synfcst only handles one location right now)
- WAT jython script: compute-per-lifecycle vs compute-per-event, pass via config (instead going to have SynFcstPreProcessor run for all events at start of lifecycle)
- separate data, synfcst code, python directory (in progress)
    - data goes into WAT watershed as the "model alternative"
    - synfcst code and python directory go into users %appdata% as a "wat plugin"

# Longer term WAT
- add deterministic and stochastic POR capability
- add synfcst refitting per realization to add forecast skill aleatory uncertainty

## DONE
- DSS: limit DSS messages through `zset msglvl` call.
- writes to "test watershed" for WAT compute to use .json file
- put WAT scripting alt into repo (DONE) - but needs updating!

# overall
- modularize the synthetic generation - started
- add tests for WAT behavior to streamline development/refactoring  - in progress
