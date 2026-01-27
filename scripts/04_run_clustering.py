"""
4) Clustering performance (core validation)

Purpose: demonstrate the threshold-free network + Leiden recovers outbreak structure.

What to include:
	•	BCubed Precision / Recall / F1 across \gamma (resolution scan)
	•	stability diagnostics: ARI between consecutive \gamma
	•	sensitivity of clustering to surveillance quality / clock assumptions
	•	show one or two “representative network/cluster” visualisations, not too many

Code implication:
	•	04_run_clustering.py (build networks + Leiden partitions across \gamma)
	•	evaluate_clustering.py (BCubed + ARI + summary)
	•	optionally plot_clustering_results.py to centralise figures
"""