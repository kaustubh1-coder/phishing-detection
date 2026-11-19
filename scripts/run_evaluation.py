import os
import sys

sys.path.insert(0, os.path.abspath("."))

import dill
import pandas as pd

from networksecurity.utils.ml_utils.evaluation import evaluate_model_robustness

def main():
    model_path = os.path.join("final_model", "model.pkl")
    prep_path = os.path.join("final_model", "preprocessor.pkl")
    test_path = os.path.join("Artifacts", "10_03_2026_22_16_37", "data_ingestion", "ingested", "test.csv")
    
    if not os.path.exists(test_path):
        test_path = os.path.join("Network_Data", "phisingData.csv")

    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = dill.load(f)

    print(f"Loading preprocessor from {prep_path}...")
    with open(prep_path, "rb") as f:
        preprocessor = dill.load(f)

    print(f"Loading test dataset from {test_path}...")
    test_df = pd.read_csv(test_path)

    print("Evaluating model robustness across 4 missing-feature scenarios...")
    report = evaluate_model_robustness(
        model=model,
        preprocessor=preprocessor,
        test_df=test_df,
        output_dir="final_model"
    )

    print("\nRobustness Evaluation Summary:")
    print("=" * 70)
    for sc in report["scenarios"]:
        print(f"Scenario: {sc['name']}")
        print(f"  Missing Features: {sc['missing_features_count']}/30")
        print(f"  Accuracy:  {sc['accuracy'] * 100:.2f}%")
        print(f"  Precision: {sc['precision'] * 100:.2f}%")
        print(f"  Recall:    {sc['recall'] * 100:.2f}%")
        print(f"  F1-Score:  {sc['f1_score'] * 100:.2f}%")
        print(f"  FPR:       {sc['false_positive_rate'] * 100:.2f}%")
        print("-" * 70)

    print("\nArtifacts saved:")
    print("  - final_model/robustness_evaluation.json")
    print("  - final_model/robustness_evaluation.csv")

if __name__ == "__main__":
    main()
