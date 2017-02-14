# innoPositioning

Collection of libraries written in Python for path mining of people's trajectories based on series of WiFi RSSI samples

mapGen - library for coverage map generation for a single AP, given environment configuration. It includes:
  - algorithm for generating coverage map based on an environment configuration
  - sample model of virtual environment configuration
  - sample model of real environment configuration
  - script for collecting RSSI measurements in a real environment along a predefined path
  - test for evaluating generated RSSI with measured RSSI along a predefined path
  
locTrack - library for location tracking of a network user. It includes:
  - algorithm for location tracking of a network user based on a series of collected RSSI measurements
  - sample of collected RSSI measurements along a predefined path in a virtual environment
  - test for evaluating positioning accuracy in a virtual environment
  - sample of collected RSSI measurements along a predefined path in a real environment
  - test for evaluating positioning accuracy in a virtual environment
 
pathMiner - library that implements algorithms for path mining
  - a collection of algorithms for path mining
  - sample trajectories in a virtual environment with the distinct presence of characteristic points for different sets of paths
  - a collection of tests that evaluates the ability of implemented algorithms to identify the presence of characteristic points 
