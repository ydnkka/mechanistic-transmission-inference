"""
3) Binary classification evaluation of edges

Purpose: show edges are meaningful as probabilities/scores, without pretending this is the end goal.

What to include (and why):
	•	PR AUC (most informative under heavy class imbalance)
	•	ROC AUC (secondary)
	•	Brier score / log loss for calibration (only for probabilistic outputs)
	•	reliability diagram (optional but persuasive): mechanistic vs logistic

Crucial framing sentence:

“Although the framework is intended for clustering, pairwise scoring provides a useful check that inferred edge weights correspond to recent linkage.”

Code implication:
	•	03_evaluate_edges.py reads the saved scenario datasets, computes edge scores, then writes:
	•	edge_eval_metrics.csv
	•	optional per-scenario curves / calibration plots
"""