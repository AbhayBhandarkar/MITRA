#!/usr/bin/env python3
import nest_asyncio
nest_asyncio.apply()

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from pipeline import pipeline  # Ensure pipeline.py and zero_shot_filter.py are in the same directory
import os

# Function to evaluate a single prompt asynchronously
async def evaluate_prompt(prompt, expected):
    result = await pipeline(prompt)
    status = result.get("status", "error")
    
    # Extract individual model confidences from the checks for analysis.
    safety_checks = result.get("checks", [])
    details = {check["check_name"]: check["confidence"] for check in safety_checks}
    
    success = 1 if (expected == status) else 0
    return {
        "prompt": prompt,
        "expected": expected,
        "status": status,
        "success": success,
        "error": result.get("error", ""),
        "response": result.get("response", ""),
        **details  # Include confidence scores for each check.
    }

def main():
    # Load the JailbreakBench JBB-Behaviors dataset with the "behaviors" configuration.
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    
    # Prepare the list of prompts with expected outcomes:
    # "harmful" prompts should be blocked, "benign" should be allowed.
    prompts = []
    for split_name in ["harmful", "benign"]:
        for entry in dataset[split_name]:
            # The dataset stores the prompt text under the key "Goal"
            prompt_text = entry["Goal"]
            expected = "blocked" if split_name == "harmful" else "allowed"
            prompts.append({"prompt": prompt_text, "expected": expected})
    
    # Evaluate each prompt through MITRA's pipeline
    results = []
    for item in prompts:
        print(f"Processing prompt: {item['prompt'][:50]}...")  # Preview first 50 characters
        result = asyncio.run(evaluate_prompt(item["prompt"], item["expected"]))
        results.append(result)
    
    # Convert results to a DataFrame for analysis
    df = pd.DataFrame(results)
    overall_success_rate = df["success"].mean()
    print(f"\nOverall success rate: {overall_success_rate * 100:.2f}%")
    
    # Compute False Positive Rate (FPR) and False Negative Rate (FNR)
    false_positives = df[(df["expected"] == "allowed") & (df["status"] == "blocked")].shape[0]
    false_negatives = df[(df["expected"] == "blocked") & (df["status"] == "allowed")].shape[0]
    total_harmful = df[df["expected"] == "blocked"].shape[0]
    total_benign = df[df["expected"] == "allowed"].shape[0]
    
    false_positive_rate = (false_positives / total_benign) * 100 if total_benign else 0
    false_negative_rate = (false_negatives / total_harmful) * 100 if total_harmful else 0
    
    print(f"False Positive Rate (FPR): {false_positive_rate:.2f}%")
    print(f"False Negative Rate (FNR): {false_negative_rate:.2f}%")
    
    # Save results to CSV file in the local directory.
    output_file = "benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Benchmark results saved to {os.path.abspath(output_file)}")
    
    # Plot 1: Success vs. Failure Bar Chart
    plt.figure(figsize=(8, 4))
    df["success"].value_counts().sort_index().plot(kind="bar", color=["red", "green"])
    plt.title("MITRA Guardrail: Defense Success vs. Failure")
    plt.xlabel("Outcome (0 = Failure, 1 = Success)")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=["Failure", "Success"], rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Confusion Matrix Heatmap
    # Build confusion matrix: rows are actual, columns are predicted.
    confusion_matrix = pd.DataFrame(
        [
            [total_harmful - false_negatives, false_negatives],
            [false_positives, total_benign - false_positives]
        ],
        index=["Actual Harmful", "Actual Benign"],
        columns=["Predicted Harmful", "Predicted Benign"]
    )
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Confusion Matrix - MITRA Guardrail")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
