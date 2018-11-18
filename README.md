# Dynamic_PCA
Statistical procedure for visualizing the animated movies of patterns recognized by principal component analysis (PCA) method.
This method is applied in a stochastic trubulent flow field in a vrotex, simulated exprimentally in WindEEE Dome facility at the Western University.

# Data and Setup Description
A floor panel with several pressure taps was used to measure the surface static pressure deficit. The center floor panel of the simulator with 413 static pressure taps distributed on concentric circles (with a maximum diameter of 56 cm) around the simulator centerline. Each tap was connected to a pressure scanner port using PVC tubing and each scanner can accommodate 16 pressure taps. However, not all ports on each scanner are used for measurements. 
The Cp_tap matrix contains the time history (60sec x 700Hz=42000 measurements) of data for pressure 512 ports (32 scanners x 16 ports each). The data is stored in a 42000 rows by 512 columns matrix. Every 16 column corresponds to one scanner, starting with scanner number 1 (S1). 
