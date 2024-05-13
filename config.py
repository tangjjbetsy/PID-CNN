############## Please REFER TO THE PAPER FOR UNDERSTANDING FEATURES USED HERE #####################
FEATURES_LIST = [
                 'pitch', 
                 'onset_time', 
                 'offset_time',
                 'velocity',
                 'duration',
                 'ioi', #Inter onset interval
                 'otd', #Offset time duration
                 # <----- Deviation Features ----->
                 'onset_time_dev',
                 'offset_time_dev',
                 'velocity_dev',
                 'duration_dev',
                 'ioi_dev',
                 'otd_dev'
                 ]


############### PARAMETERS FOR QUANTIZATION ###################
DEFAULT_LOADING_PROGRAMS = range(128)
MIN_VELOCITY = 10