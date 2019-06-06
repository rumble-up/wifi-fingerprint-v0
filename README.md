# TL;DR
Although WAP signal strengths provide very messy data, we can still use it to locate a person to within 10 meters inside a building.

# Introduction
Although GPS is a reliable positioning technnology, it does not function well indoors because buildings block the signals from satellites.  For example, a user in a mall might want to find a way to their favorite store, but they cannot use GPS technology to map a route their because the satellite signals are very weak or non-existent indoors.

For indoor positioning an alternative approach is to utilize the signal strength from WiFi Wireless Access Points (WAPs) within the building.  For example, a user could install an app on their cellphone that senses the signal strength from different WAPs and then calculates their indoor position. This approach is called WiFi fingerprinting - the idea being that each indoor position will have a unique sets of signals from different WAPs in the building.

Although WiFi fingerprinting can be implemented without special sensors beyond those found in a normal mobile phone, the data from WAPs is quite messy.  The signal measurements when a user stands in the same place can be drastically different depending on many factors including:
* Mobile phone model
* Version of Android or iOS
* How the phone is held, the height of the user
* Number of people using the network
* Number of other people and obstacles physically present in the space 

The point of this project was to accurately locate a person despite the messiness in WAP signal data.


# The Dataset
[UJIIndoorLoc](https://archive.ics.uci.edu/ml/datasets/ujiindoorloc) is a publically available dataset from 3 large campus buildings with 4 or more floors. Each record consists of a location somewhere within the buildings and the WAPs detected at that location measured by Received Signal Strength Intensity (RSSI). The data also has the following characteristics:

* Recorded by 20 users and 25 different Android phones
* 108,703 m2 total covered indoor area
* Signals from 520 WAPs
* 19,937 training records and 1,111 validating/test records.

## Dataset challenges
The Training dataset and the Validation dataset were recorded with two different methods. 

__Training - users stand in specific locations__ <br>
933 unique and specific locations were marked on the floor within the 3 buildings and volunteers were asked to stand on these locations and use an app installed on their phone to record WAP signal strength.

__Validation - users stand in random locations throughout buildings__ <br>
Data was recorded 4 months after the Training dataset. Volunteers chose their own location, then used a different app to record the WAP signals. The app guessed their location, then the users corrected the app if the guess was incorrect.

__Testing - unknown collection method__ <br>
A final dataset was collected to use in the 2015 EvAAL-ETRI Competition. International teams had 6 weeks to develop models and then competed to locate positions using WAP signals with the lowest margin of error.  They were provided a test set of new WAP signals, but the target variables, the building, floor, latitude, and longitude were hidden from them. 





