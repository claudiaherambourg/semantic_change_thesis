"""
# 07c - CamemBERT contextual regressions

The regression dataset for the CamemBERT contextual analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from google.colab import drive

drive.mount("/content/drive")
PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

MIN_OCCURRENCES = 25
EPSILON = 1e-9

camembert_displacement = pd.read_csv(TABLES_DIR / "semantic_displacement_camembert_centroids.csv")
occurrence_counts = pd.read_csv(TABLES_DIR / "camembert_contextual_occurrence_counts.csv")
contextual_dispersion = pd.read_csv(TABLES_DIR / "camembert_contextual_dispersion.csv")

print(f"Displacement rows: {len(camembert_displacement)}")
print(f"Occurrence rows: {len(occurrence_counts)}")
print(f"Dispersion rows: {len(contextual_dispersion)}")


current_counts = occurrence_counts.rename(columns={"n_occurrences": "frequency_t"})
next_counts = occurrence_counts.rename(columns={"decade": "next_decade", "n_occurrences": "frequency_next"})

analysis_data = camembert_displacement.merge(current_counts[["word", "decade", "frequency_t"]],
                                             on=["word", "decade"], how="left")

analysis_data = analysis_data.merge(next_counts[["word", "next_decade", "frequency_next"]],
                                    on=["word", "next_decade"], how="left")

analysis_data = analysis_data.merge(contextual_dispersion[["word", "decade", "mean_contextual_dispersion", "median_contextual_dispersion"]],
                                    on=["word", "decade"], how="left")

analysis_data = analysis_data.dropna(subset=["semantic_displacement", "frequency_t", "frequency_next", "mean_contextual_dispersion"]).copy()

analysis_data = analysis_data[(analysis_data["frequency_t"] >= MIN_OCCURRENCES) &
                              (analysis_data["frequency_next"] >= MIN_OCCURRENCES)].copy()

analysis_data["log_frequency"] = np.log(analysis_data["frequency_t"] + EPSILON)
analysis_data["log_semantic_displacement"] = np.log(analysis_data["semantic_displacement"] + EPSILON)

analysis_data["norm_log_displacement"] = (analysis_data["log_semantic_displacement"] - analysis_data.groupby("decade")["log_semantic_displacement"].transform("median"))

analysis_data["contextual_dispersion_centered"] = (analysis_data["mean_contextual_dispersion"] - analysis_data.groupby("decade")["mean_contextual_dispersion"].transform("median"))

analysis_data.to_csv(TABLES_DIR / "camembert_contextual_regression_dataset.csv", index=False, encoding="utf-8")

print(f"Regression rows: {len(analysis_data)}")
print(f"Unique words: {analysis_data['word'].nunique()}")

conformity_result = smf.ols("norm_log_displacement ~ log_frequency + C(decade)",
                            data=analysis_data).fit(cov_type="cluster", cov_kwds={"groups": analysis_data["word"]})
print(conformity_result.summary())

with open(TABLES_DIR / "camembert_conformity_results.txt", "w", encoding="utf-8") as f:
    f.write(str(conformity_result.summary()))

innovation_result = smf.ols("norm_log_displacement ~ log_frequency + contextual_dispersion_centered + C(decade)",
                            data=analysis_data).fit(cov_type="cluster", cov_kwds={"groups": analysis_data["word"]})

print(innovation_result.summary())

with open(TABLES_DIR / "camembert_contextual_innovation_results.txt", "w", encoding="utf-8") as f:
    f.write(str(innovation_result.summary()))

def regression_summary(result, model_name):
    table = pd.DataFrame({"term": result.params.index,
                          "coef": result.params.values,
                          "std_error": result.bse.values,
                          "t_or_z": result.tvalues.values,
                          "p_value": result.pvalues.values})

    table["model"] = model_name
    return table


regression_results = pd.concat([regression_summary(conformity_result, "camembert_conformity_ols_clustered"),
                                regression_summary(innovation_result, "camembert_contextual_innovation_ols_clustered")],
                               ignore_index=True)

regression_results.to_csv(TABLES_DIR / "camembert_contextual_regression_results.csv", index=False, encoding="utf-8")
