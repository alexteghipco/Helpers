# Helpers
Some scripts for folks (NiiStat-related)

extractROIsFromMatFiles.m can help you extract some values from our subject-level .mat files. 

nii_stat_svm.m is edited to return all feature loadings above the initial arbitrary threshold of 1. To use this, replace original nii_stat_svm.m script with this one. Use at your own peril! In most cases, focusing on the top 20 features as the original script did is a much better idea!

nii_stat_svm_core_stacked.m edits nii_stat_svm_core.m to take in data from two different modalities and build 4 different kinds of models on the same holdout subsamples so they can be compared directly: i) SVR model trained only on data from modality 1, ii) SVR model trained only on data from modality 2, iii) SVR model trained on pooled data (both modalities), and iv) a logistic regression model that blends predictions from the SVR models trained on the individual modalities. The "stacked" logistic regression model is still tested on the holdout samples. Note, logistic regression is used to weight the probabilities of class predictions, not the class predictions themselves.
