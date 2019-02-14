# wifi-fingerprint
Predict location from UJIIndoorLoc dataset

# Dataset observations
* Most locations can see ~16 WAPs

# Issues in dataset
   
* 97.5% of data is 100s  
  - Occurs where phone perceived no signal from WAP
  - Should 100s be treated as NaNs? or substitute -110 dBm to simulate WAP being far away?
  - 76 rows contain only 100s (mostly phone 1, user 8)

* 486 observations of unusually high outlier signals above -30 dBm: 
  - 450 (88%) from user 6, phone 19 (floors 3 & 4?)
  - Quick fix? drop values above -30 dBm

* A corner of floor 4

* Do we mix training & validation for final model? 


# Files
* by_rank.py: organize WAP signals by rank
* descriptive_viz.py: Preliminary descriptive visualization of dataset

