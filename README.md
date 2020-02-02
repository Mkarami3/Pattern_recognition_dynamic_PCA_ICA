# Pattern recognition in a stochastic flow using dynamicPCA and dynamicICA
The classical method for recognition of patterns in data is principal component analysis (PCA). However, this technique, which belongs to unsupervised machine learning algorithms, provides modes that are static in time. Dynamic-PCA, as a novel technique, can overcome this constraint and provide time evolutions of PCA modes. Here, as part of my PhD, I applied this technique to a stochastic surface pressure field of a simulated tornado vortex. Result shows an interesting phenomenon which cannot be easily captured by common visualization techniques such as smoke injection.

# Data and Setup Description
A floor panel with several pressure taps was used to measure the surface static pressure deficit. The center floor panel of the simulator with 413 static pressure taps distributed on concentric circles (with a maximum diameter of 56 cm) around the simulator centerline. Each tap was connected to a pressure scanner port using PVC tubing and each scanner can accommodate 16 pressure taps. However, not all ports on each scanner are used for measurements. 
The Cp_tap matrix contains the time history (60sec x 700Hz=42000 measurements) of data for pressure 512 ports (32 scanners x 16 ports each). The data is stored in a 42000 rows by 512 columns matrix. Every 16 column corresponds to one scanner, starting with scanner number 1 (S1). 
A slice of experimental data, named Cp_tap, is uploaded

Thanks to Dr. Luigi Carassale in University of Genova for his great helps.

# Visualization Results 
# PCA Modes
![](Img/PCA_Mode%201.png)



