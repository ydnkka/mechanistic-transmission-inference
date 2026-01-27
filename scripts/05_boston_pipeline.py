"""
5) Boston application (case study)

Purpose: demonstrate usefulness on real data; don’t oversell “ground truth”.

What to include:
	•	brief pre-processing summary (sampling period, QC, distance computation TN93→SNP)
	•	resulting SMTCs at selected \gamma values and how they align with known outbreaks:
	•	conference cluster
	•	SNF cluster
	•	BHCHP cluster
	•	show robustness across resolution (do the key clusters persist?)
	•	emphasise what you can infer (structure, density, temporal concentration), not exact who-infected-whom.

Code implication:
	•	05_boston_pipeline.py (compute pairwise distances + run epilink inference + build network + Leiden)
	•	boston_figures.py (cluster summaries and persistence plots)
"""