import pandas as pd


def validation_analysis(eval_results: pd.DataFrame, output_dir: str):
    eval_results.to_csv(f"{output_dir}/validation_results.csv", index=False)

    predicted_labels = eval_results['predicted_label']
    true_labels = eval_results['true_label']
    # Calculate metrics
    accuracy = (predicted_labels == true_labels).mean()
    mae = (predicted_labels - true_labels).abs().mean()
    score = 1 - (mae / 4)
    
    distance_per_label = (predicted_labels - true_labels).abs().groupby(true_labels)
    accuracy_per_label = (predicted_labels == true_labels).groupby(true_labels)

    big_mistakes = eval_results[(predicted_labels - true_labels).abs() >= 2]
    big_mistakes.to_csv(f"{output_dir}/big_mistakes.csv", index=False)
    # Save metrics
    with open(f"{output_dir}/validation_metrics.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Overall Score: {score:.4f}\n\n")
        f.write("MAE per True Label:\n")
        for label, group in distance_per_label:
            f.write(f"Label {label}: MAE = {group.mean():.4f}\n")
        f.write("\nAccuracy per True Label:\n")
        for label, group in accuracy_per_label:
            f.write(f"Label {label}: Accuracy = {group.mean():.4f}\n")



