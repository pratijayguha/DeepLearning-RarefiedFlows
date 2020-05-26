# Investigating Rarefied Flows using Deep Learning

This is a project that investigates an efficient learning methods for deep learning to obtain real time results within acceptable limits.

**IndVal:** This is the model that is involved in optimising the predictions for Normalised Mach Numbers using one vector [X-coordinate,L/D Ratio] at a time.

**MaxVal:** This is the model that handles the maximum values of Mach Number in each L/D Ratio.

**SNwithMult** is a model being used for optimizer evaluation with a Multiplication Layer added in for training.

**SNwithoutMult** is a model without the Multiplication layer included being used for optimizer evaluation. We aim to see which of these are a more effective fit for our problem statement.

For Results please look at the plots in /PostProc/SN*