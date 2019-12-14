###########################################

CS221 Final Project

Authors:

Kongphop Wongpattananukul (kongw@stanford.edu)

Pouya Rezazadeh Kalehbasti (pouyar@stanford.edu)

Dong Hee Song (dhsong@stanford.edu)

###########################################

# Description
	This package includes the following files:
	- environment.yml
		- A conda environment containing all the requirements to run
		  the contained python files (Python ver. 3.7.4)
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
	implementations are based on the original lunar lander problem from openAI's gym.
	Note that the Q-learning algorithm used in the files is partially based on Assignment
	4 (Blackjack) from CS221.

	* pkgDelivery.py, takeoff.py are attempts to modify the original environment
	to tackle other formats of this problem (e.g. different lander types, different
	objectives), and they are derived from the original LunarLander module from
	OpenAI's gym package.

# Usage
	The following sample command line argument should run any python function
	listed above. The hyperparameters for the included algorithms can be
	changed inside each python file.
	$ python heuristic.py
