# Helpers
Some scripts for folks (NiiStat-related)

extractROIsFromMatFiles.m can help you extract some values from our subject-level .mat files. 

nii_stat_svm.m is edited to return all feature loadings above the initial arbitrary threshold of 1. To use this, replace original nii_stat_svm.m script with this one. Use at your own peril! In most cases, focusing on the top 20 features as the original script did is a much better idea!
