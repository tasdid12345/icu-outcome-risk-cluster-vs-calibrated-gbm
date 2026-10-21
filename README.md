# Mortality Risk Ranking: k-Means Cluster Rates vs Calibrated Gradient Boosting

Cluster analysis is a common way to stratify ICU patients by risk, and the paper "Risk of pneumonia and associated outcomes in intensive care unit: an integrated approach of visual and cluster analysis" (2019) uses it to group patients and read off outcome rates. A cluster-level rate is coarse: every patient in a cluster receives the same risk, so ranking within clusters is lost. This experiment asks whether a class-balanced, isotonic-calibrated gradient-boosting model improves ranking and calibration over a k-means cluster-rate baseline.

**Question:** Can a class-balanced, isotonic-calibrated gradient-boosting model give better-ranked and better-calibrated patient mortality risk than a k-means cluster-rate baseline on a public ICU-style clinical dataset?

**Dataset:** UCI Heart Failure Clinical Records (binary death-during-follow-up outcome, clinical features), obtained from the UCI archive (dataset id 519). It is a stand-in for ICU outcome prediction. It is not the paper's own dataset, because the paper's pneumonia ICU data is not publicly available and MIMIC requires credentials. A stratified 75/25 train/test split with seed 42 is used.

**Baseline:** k-means clustering on standardized features. The number of clusters is chosen by silhouette score on the training data. Each test patient receives the training death rate of their assigned cluster.

**Improved model:** class-balanced LightGBM with hyperparameters chosen by 5-fold cross-validation on the training set. Its probabilities are calibrated with isotonic regression fit on out-of-fold training predictions.

**How to run**
```
pip install -r requirements.txt
python analysis.py
```

**Results:** written to `results.json`. Charts are saved to `figures/`.

## Reference

This project is motivated by the following paper:

> G Favara, PM Riela, A Maugeri (2019). *Risk of pneumonia and associated outcomes in intensive care unit: an integrated approach of visual and cluster analysis*. 2019 IEEE World. https://ieeexplore.ieee.org/abstract/document/8817271/

Paper author: Prof. Giovanni Gallo, University of Catania (gallo@dmi.unict.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
