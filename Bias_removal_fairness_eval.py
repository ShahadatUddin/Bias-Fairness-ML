
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

output_dir = "D52"
os.makedirs(output_dir, exist_ok=True)  # Create if not exists


def calculate_probability_distribution(data, column):
    """
    Converts a categorical column to a probability distribution.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column (str): The name of the categorical column to convert.

    Returns:
        pd.Series: A probability distribution with index as categories and values as probabilities.
    """
    # Count occurrences of each category
    counts = data[column].value_counts(normalize=True)  # Normalize=True returns probabilities

    return counts

def calculate_categorical_emd(dataframe, target_column, protected_attribute):
    """
    Calculates EMD for categorical data by using probability distributions of categories.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing the data.
        target_column (str): The column name representing the class to be predicted.
        protected_attribute (str): The column name representing the protected attribute.

    Returns:
        dict: A dictionary containing the EMD values for each group compared to the total dataset.
    """
    # Calculate the overall distribution of the target column
    overall_distribution = calculate_probability_distribution(dataframe, target_column)
    all_categories = overall_distribution.index.tolist()

    # Get unique protected attribute values
    protected_values = dataframe[protected_attribute].unique()

    emd_results = {}

    for value in protected_values:
        group_data = dataframe[dataframe[protected_attribute] == value]
        group_distribution = calculate_probability_distribution(group_data, target_column)

        # Ensure both distributions have the same categories
        overall_probs = overall_distribution.reindex(all_categories, fill_value=0).values
        group_probs = group_distribution.reindex(all_categories, fill_value=0).values

        # Calculate EMD
        emd_value = wasserstein_distance(range(len(all_categories)), range(len(all_categories)), u_weights=group_probs, v_weights=overall_probs)
        emd_results[value] = emd_value

    return emd_results

def permutation_test_by_shuffling_labels(df, target_column, protected_attribute, group_value, n_perm=10000):
    """Estimate an EMD p-value by shuffling the protected attribute."""
    observed_emd = calculate_categorical_emd(df, target_column, protected_attribute)[group_value]
    null_distribution = []

    for _ in range(n_perm):
        shuffled_df = df.copy()

        # Shuffle the protected attribute (e.g., 'Sex')
        shuffled_values = df[protected_attribute].values.copy()
        np.random.shuffle(shuffled_values)
        shuffled_df[protected_attribute] = shuffled_values

        # Recalculate EMD
        try:
            emd = calculate_categorical_emd(shuffled_df, target_column, protected_attribute)[group_value]
        except KeyError:
            emd = 0
        null_distribution.append(emd)

    p_value = sum(abs(x) >= abs(observed_emd) for x in null_distribution) / n_perm
    return observed_emd, p_value, null_distribution


df = pd.read_csv("D52.csv")
target = "Heart Attack"
protected_attribute = "Gender"
female_val = "Female"
male_val = "Male"


# Force binary encoding of the target column
if df[target].dtype == object or not set(df[target].unique()).issubset({0, 1}):
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    print("Encoded target labels:", dict(zip(le.classes_, le.transform(le.classes_))))


# Get gender counts
gender_counts = df[protected_attribute].value_counts()
remove_gender_value = gender_counts.idxmax()  # Gender with more samples
other_gender_value = gender_counts.idxmin()   # Gender with fewer samples

# Get outcome values (e.g., 0 and 1)
outcome_values = df[target].unique()

# Calculate conditional probabilities
cond_probs = {}
for outcome in outcome_values:
    p_outcome_given_remove_gender = (
        (df[(df[protected_attribute] == remove_gender_value) & (df[target] == outcome)].shape[0]) /
        df[df[protected_attribute] == remove_gender_value].shape[0]
    )
    p_outcome_given_other_gender = (
        (df[(df[protected_attribute] == other_gender_value) & (df[target] == outcome)].shape[0]) /
        df[df[protected_attribute] == other_gender_value].shape[0]
    )
    cond_probs[outcome] = {
        'remove_gender': p_outcome_given_remove_gender,
        'other_gender': p_outcome_given_other_gender
    }

# Decide which outcome value to remove
# Remove the one that has higher probability in remove_gender than in other gender
remove_outcome_value = max(cond_probs, key=lambda o: cond_probs[o]['remove_gender'] - cond_probs[o]['other_gender'])

# Get P(remove_outcome_value | other_gender)
p_outcome_other_gender = cond_probs[remove_outcome_value]['other_gender']

# Count values needed for formula
total_remove_gender = df[df[protected_attribute] == remove_gender_value].shape[0]
current_count = df[(df[protected_attribute] == remove_gender_value) & (df[target] == remove_outcome_value)].shape[0]

# Calculate n_remove
denominator = p_outcome_other_gender - 1
if denominator == 0:
    n_remove = float('inf')  # undefined
else:
    n_remove = (p_outcome_other_gender * total_remove_gender - current_count) / denominator
    n_remove = int(round(n_remove)) if n_remove >= 0 else 0

print(
    f"Removal focus: {remove_gender_value} / outcome {remove_outcome_value}; "
    f"maximum removable samples: {n_remove}."
)

result = calculate_categorical_emd(df, target_column=target, protected_attribute=protected_attribute)
print("EMD Results:")
for group, emd_value in result.items():
    print(f"EMD for {group} vs. total population: {emd_value}")

obs, p_val, null_dist = permutation_test_by_shuffling_labels(
    df, target_column=target, protected_attribute=protected_attribute, group_value=female_val, n_perm=1000
)

print(f"Observed EMD (Female): {obs:.4f}")
print(f"P-value: {p_val:.4f}")


sns.kdeplot(null_dist, label='Permuted EMDs')
plt.axvline(obs, color='red', label='Observed EMD')
plt.legend()
plt.title('Distribution of Permuted EMDs vs. Observed')
plt.savefig(os.path.join(output_dir, 'D52_permEMDs.png'))

def evaluate_model_on_gender_subset(model, X, y, n_splits=10):
    """Evaluate a model on a gender-specific subset using Stratified K-Fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tprs, fprs = [], []
    fns, fps, ratios = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0

        tprs.append(tpr)
        fprs.append(fpr)
        fns.append(fn)
        fps.append(fp)
        ratios.append(fn / fp if fp > 0 else np.nan)

    valid_ratios = [r for r in ratios if not np.isnan(r)]
    mean_ratio = np.mean(valid_ratios) if valid_ratios else np.nan

    return {
        "mean_tpr": np.mean(tprs),
        "mean_fpr": np.mean(fprs),
        "fold_fns": fns,
        "fold_fps": fps,
        "fold_tprs": tprs,
        "fold_fprs": fprs,
        "mean_fn": np.mean(fns),
        "mean_fp": np.mean(fps),
        "fold_ratios": ratios,
        "mean_ratio": mean_ratio,
    }

def safe_ttest(a, b):
    """Return Welch's t-test results when both samples have enough observations."""
    if len(a) > 1 and len(b) > 1:
        return ttest_ind(a, b, equal_var=False)
    return (np.nan, np.nan)

# Define models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

fairness_threshold = 0.05

# Base setup
X_full = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
y_full = df[target]
sex = df[protected_attribute]


# Identify removable samples
removable_indices = df[(df[protected_attribute] == remove_gender_value) & (df[target] == remove_outcome_value)].index.tolist()

print(
    f"Identified {len(removable_indices)} removable samples: "
    f"{remove_gender_value} individuals with outcome {remove_outcome_value}."
)


# Initialize tracking
tracking_results = []
satisfied_models = set()


# Iteratively remove biased samples while logging fairness metrics
for removal_step in range(n_remove):
    print(f"Step {removal_step}...")
    step_record = {"Samples_Removed": removal_step}

    emd_results = calculate_categorical_emd(df, target_column=target, protected_attribute=protected_attribute)
    step_record['Female_EMD'] = emd_results.get(female_val, np.nan)
    step_record['Male_EMD'] = emd_results.get(male_val, np.nan)

    obs, p_val, null_dist = permutation_test_by_shuffling_labels(
        df,
        target_column=target,
        protected_attribute=protected_attribute,
        group_value=female_val,
        n_perm=1000
    )
    step_record['perm_pval'] = p_val

    X_full = df.drop(columns=[target])
    y_full = df[target]
    sex = df[protected_attribute]

    X_full = pd.get_dummies(X_full, drop_first=True)

    for model_name, model in models.items():
        if model_name in satisfied_models:
            print(f"Skipping {model_name}, already satisfied fairness.")
            continue

        mask_m = sex == male_val
        X_m = X_full[mask_m].values
        y_m = y_full[mask_m]

        mask_f = sex == female_val
        X_f = X_full[mask_f].values
        y_f = y_full[mask_f]

        male_metrics = evaluate_model_on_gender_subset(model, X_m, y_m)
        female_metrics = evaluate_model_on_gender_subset(model, X_f, y_f)

        step_record.update({
            f"{model_name}_Male_TPR": male_metrics["mean_tpr"],
            f"{model_name}_Female_TPR": female_metrics["mean_tpr"],
            f"{model_name}_Male_FPR": male_metrics["mean_fpr"],
            f"{model_name}_Female_FPR": female_metrics["mean_fpr"],
            f"{model_name}_Male_FN/FP": male_metrics["mean_ratio"],
            f"{model_name}_Female_FN/FP": female_metrics["mean_ratio"]
        })

        _, p_value_tpr = safe_ttest(male_metrics["fold_tprs"], female_metrics["fold_tprs"])
        _, p_value_fpr = safe_ttest(male_metrics["fold_fprs"], female_metrics["fold_fprs"])
        _, p_value_fn = safe_ttest(male_metrics["fold_fns"], female_metrics["fold_fns"])
        _, p_value_fp = safe_ttest(male_metrics["fold_fps"], female_metrics["fold_fps"])
        _, p_value_ratio = safe_ttest(
            [r for r in male_metrics["fold_ratios"] if not np.isnan(r)],
            [r for r in female_metrics["fold_ratios"] if not np.isnan(r)]
        )

        # Save the p-values
        step_record.update({
            f"{model_name}_TPR_pvalue": p_value_tpr,
            f"{model_name}_FPR_pvalue": p_value_fpr,
            f"{model_name}_FN_pvalue": p_value_fn,
            f"{model_name}_FP_pvalue": p_value_fp,
            f"{model_name}_FN/FP_pvalue": p_value_ratio
        })

        if all(p >= fairness_threshold for p in [p_value_tpr, p_value_fpr, p_value_ratio]):
            satisfied_models.add(model_name)
            print(f"{model_name} has satisfied all fairness criteria and will be skipped in future steps.")


    tracking_results.append(step_record)

    if removal_step < n_remove - 1:
        to_remove = random.choice(removable_indices)
        df = df.drop(index=to_remove)

        removable_indices = df[(df[protected_attribute] == remove_gender_value) & (df[target] == remove_outcome_value)].index.tolist()
    if len(satisfied_models) == len(models):
        print(f"All models have satisfied fairness by step {removal_step}.")
        break

tracking_df = pd.DataFrame(tracking_results)
tracking_df.to_excel(os.path.join(output_dir, "D52_fairness_metrics_tracking.xlsx"), index=False)

# Visualise headline EMD trajectory and model-level trends
plt.figure(figsize=(10, 6))
plt.plot(tracking_df['Samples_Removed'], tracking_df['Female_EMD'], label='Female EMD', color='purple')
plt.plot(tracking_df['Samples_Removed'], tracking_df['Male_EMD'], label='Male EMD', color='orange')
plt.xlabel('Number of Male Positive Samples Removed')
plt.ylabel('EMD')
plt.title('Male vs Female EMD Over Sample Removals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "D52_EMD_vs_samples_removed.png"))


def plot_all_models_grid_uniform(tracking_df):
    """Plot TPR, FPR, and FN/FP ratios for each model over sample removals."""
    models = ["KNN", "Logistic Regression", "SVM", "ANN", "Decision Tree", "Random Forest"]

    metric_colors = {
        "TPR": "blue",
        "FPR": "red",
        "FN/FP": "green"
    }

    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.flatten()

    y_min, y_max = 0, 2.0

    for idx, model_name in enumerate(models):
        ax = axs[idx]
        samples = tracking_df['Samples_Removed']

        # Plot each metric
        ax.plot(samples, tracking_df[f"{model_name}_Male_TPR"], label="Male TPR", color=metric_colors["TPR"], linestyle='-')
        ax.plot(samples, tracking_df[f"{model_name}_Female_TPR"], label="Female TPR", color=metric_colors["TPR"], linestyle='--')

        ax.plot(samples, tracking_df[f"{model_name}_Male_FPR"], label="Male FPR", color=metric_colors["FPR"], linestyle='-')
        ax.plot(samples, tracking_df[f"{model_name}_Female_FPR"], label="Female FPR", color=metric_colors["FPR"], linestyle='--')

        ax.plot(samples, tracking_df[f"{model_name}_Male_FN/FP"], label="Male FN/FP Ratio", color=metric_colors["FN/FP"], linestyle='-')
        ax.plot(samples, tracking_df[f"{model_name}_Female_FN/FP"], label="Female FN/FP Ratio", color=metric_colors["FN/FP"], linestyle='--')

        ax.set_title(f"{model_name}", fontsize=14)
        ax.set_xlabel("Samples Removed", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_ylim(y_min, y_max)  # keep consistent y-axis across plots
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=10)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=12, frameon=False)

    plt.subplots_adjust(top=0.8)

    plt.suptitle("Fairness Metrics vs Samples Removed (Male vs Female)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "D52_metrics_vs_samples_removed.png"))



plot_all_models_grid_uniform(tracking_df)



def plot_pvalues_grid(tracking_df):
    """Visualise p-value trends for each model and metric."""
    models = ["KNN", "Logistic Regression", "SVM", "ANN", "Decision Tree", "Random Forest"]

    metrics = ["TPR", "FPR", "FN/FP"]

    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.flatten()

    for idx, model_name in enumerate(models):
        ax = axs[idx]
        samples = tracking_df['Samples_Removed']

        for metric in metrics:
            pval_column = f"{model_name}_{metric}_pvalue"
            if pval_column in tracking_df.columns:
                ax.plot(samples, tracking_df[pval_column], label=f"{metric} p-value")

        # Fairness threshold line at 0.05
        ax.axhline(y=0.05, color='red', linestyle='--', label="Fairness Threshold (p=0.05)")

        ax.set_title(f"{model_name}", fontsize=14)
        ax.set_xlabel("Samples Removed", fontsize=12)
        ax.set_ylabel("p-value", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', labelsize=10)

    plt.suptitle("p-value Trends vs Samples Removed (for each Model and Metric)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "D52_metrics_pval_trends"))

plot_pvalues_grid(tracking_df)


original_emd = tracking_df['Female_EMD'].iloc[0]

tracking_df['EMD_Reduction_Percent'] = 100 * (original_emd - tracking_df['Female_EMD']) / original_emd


tracking_df.to_csv(os.path.join(output_dir, "D52_fairness_metrics_tracking.csv"), index=False)
