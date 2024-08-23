#!/bin/bash

# Datasets
wget -p models/ https://nextcloud.in.tum.de/index.php/s/twGzepiaqoq8kqm/download/datasets_initial_submission.zip
unzip models/datasets_initial_submission.zip

# Q-value functions
wget -p models/ https://nextcloud.in.tum.de/index.php/s/8RyNzX79aBrjWzT/download/value_fns_initial_submission.zip
unzip models/value_fns_initial_submission.zip

# Policies
wget -p models/ https://nextcloud.in.tum.de/index.php/s/xXtATYPWnyG5awE/download/policy_irl_initial_submission.zip
unzip models/policy_irl_initial_submission.zip

# Dynamics
wget -p models/ https://nextcloud.in.tum.de/index.php/s/gqQbd4KPwK9ogjR/download/dynamics_irl_initial_submission.zip
unzip models/dynamics_irl_initial_submission.zip