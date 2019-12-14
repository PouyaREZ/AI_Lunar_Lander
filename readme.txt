###########################################

CS221 Final Project

Authors:

Kongphop Wongpattananukul (kongw@stanford.edu)

Pouya Rezazadeh Kalehbasti (pouyar@stanford.edu)

Dong Hee Song (dhsong@stanford.edu)

###########################################

# Description
This package include the following files:
- environment.yml
	- environment.yml is a conda environment containing all the requirements to run the python files.
- heuristic.py
	- Implementation of section 4.1 from the report
- oracle.py
	- Implementation of section 4.2 from the report
- linear.py
	- Implementation of section 4.3.1 from the report
- linear_based_on_heuristic.py
	- Implementation of section 4.3.1 from the report
- deepQlearn.py
	- Implementation of section 4.3.2 from the report
- pkgDelivery.py
	- Implementation of section 4.3.3 from the report
- takeoff.py
	- Implementation of section 4.3.4 from the report
	
# Notes
* heuristic.py, oracle.py, linear.py, linear_based_on_heuristic.py, deepQlearn.py 
work on implementation of each algorithm on original lunar lander problem (Note
that Q-learning algorithm is partially based on Assignment 4 (Blackjack) from CS221).

* pkgDelivery.py, takeoff.py are an attempt to change environment to tackle other
aspects of this problem (e.g. different lander, different route) derived from 
original LunarLander module in OpenAI's gym package.

# Usage
The following command line argument should work for any python function listed above
and the hyperparameter could be changed inside each python file. The model was developed in Python 3.7.1.
$ python heuristic.py
