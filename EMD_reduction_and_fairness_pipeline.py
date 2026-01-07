from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import random
from scipy.stats import wasserstein_distance
import os
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import zlib


def MODEL_FACTORIES(seed: int):
    return {
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=seed),
        "KNN":          lambda: KNeighborsClassifier(n_neighbors=5),
        "LogReg":       lambda: LogisticRegression(solver="lbfgs", max_iter=1000),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "SVM":          lambda: SVC(kernel="rbf", gamma="scale", C=1.0),
    }

# ---------- Helper Functions ----------

def calculate_probability_distribution(data, column):
    counts = data[column].value_counts(normalize=True)
    return counts

def _prep_xy(df, target):
    """One-hot encode features (drop target), return X, y as numpy arrays."""
    data = df.copy()

    # Ensure binary 0/1 target
    if data[target].dtype == object or not set(pd.unique(data[target])).issubset({0, 1}):
        le = LabelEncoder()
        data[target] = le.fit_transform(data[target])

    X = pd.get_dummies(
        data.drop(columns=[target]),  
        drop_first=True
    )
    y = data[target].astype(int)

    # Drop rows with any missing values after encoding to keep X,y aligned
    full = pd.concat([X, y], axis=1).dropna()
    X = full.drop(columns=[target]).values
    y = full[target].values
    return X, y

def _rng_for(base_seed, *keys):
    """
    Build a stable per-call seed from a base seed + keys (dataset, phase, group, mid, etc.).
    Avoids Python's built-in hash randomness.
    """
    key = "|".join(map(str, keys))
    s = (zlib.crc32(key.encode("utf-8")) + int(base_seed)) % (2**32)
    return np.random.default_rng(s)

def calculate_categorical_emd(dataframe, target_column, protected_attribute):
    overall_distribution = calculate_probability_distribution(dataframe, target_column)
    all_categories = overall_distribution.index.tolist()
    protected_values = dataframe[protected_attribute].unique()

    emd_results = {}
    for value in protected_values:
        group_data = dataframe[dataframe[protected_attribute] == value]
        group_distribution = calculate_probability_distribution(group_data, target_column)
        overall_probs = overall_distribution.reindex(all_categories, fill_value=0).values
        group_probs = group_distribution.reindex(all_categories, fill_value=0).values
        emd_value = wasserstein_distance(range(len(all_categories)), range(len(all_categories)),
                                         u_weights=group_probs, v_weights=overall_probs)
        emd_results[value] = emd_value
    return emd_results

def permutation_test_by_shuffling_labels(df, target_column, protected_attribute, group_value, n_perm=10000, rng = None):
    observed_emd = calculate_categorical_emd(df, target_column, protected_attribute)[group_value]
    null_distribution = []

    if rng is None:
        rng = np.random.default_rng(42)  # fallback deterministic seed

    for _ in range(n_perm):
        shuffled_df = df.copy()
        shuffled_values = df[protected_attribute].to_numpy().copy()
        rng.shuffle(shuffled_values)
        shuffled_df[protected_attribute] = shuffled_values

        try:
            emd = calculate_categorical_emd(shuffled_df, target_column, protected_attribute)[group_value]
        except KeyError:
            emd = 0.0
        null_distribution.append(emd)

    p_value = sum(abs(x) >= abs(observed_emd) for x in null_distribution) / n_perm
    return observed_emd, p_value, null_distribution

def get_unbiased_emd_info(csv_path, target, protected_attribute, female_val, male_val,
                           threshold, n_perm=10000, seed=42):
    
    model_factories = MODEL_FACTORIES(seed)
    

    df = pd.read_csv(csv_path)
    dataset_name = os.path.basename(csv_path).replace(".csv", "")
    total_rows = len(df)
    rng = random.Random(seed)

    if df[target].dtype == object or not set(df[target].unique()).issubset({0, 1}):
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])

    gender_counts = df[protected_attribute].value_counts()
    remove_gender_value = gender_counts.idxmax()
    other_gender_value = gender_counts.idxmin()

    
    outcome_values = df[target].unique()
    cond_probs = {}
    for outcome in outcome_values:
        p_remove = (df[(df[protected_attribute] == remove_gender_value) & (df[target] == outcome)].shape[0]) / gender_counts[remove_gender_value]
        p_other = (df[(df[protected_attribute] == other_gender_value) & (df[target] == outcome)].shape[0]) / gender_counts[other_gender_value]
        cond_probs[outcome] = {'remove_gender': p_remove, 'other_gender': p_other}

    remove_outcome_value = max(cond_probs, key=lambda o: cond_probs[o]['remove_gender'] - cond_probs[o]['other_gender'])
    p_outcome_other_gender = cond_probs[remove_outcome_value]['other_gender']
    total_remove_gender = gender_counts[remove_gender_value]
    current_count = df[(df[protected_attribute] == remove_gender_value) & (df[target] == remove_outcome_value)].shape[0]

    denominator = p_outcome_other_gender - 1
    if denominator == 0:
        n_max = float('inf')
    else:
        n_max = int(round((p_outcome_other_gender * total_remove_gender - current_count) / denominator))
        n_max = max(n_max, 0)

    removable_indices = df[(df[protected_attribute] == remove_gender_value) & (df[target] == remove_outcome_value)].index.tolist()
    rng.shuffle(removable_indices)


    # Original EMDs
    emd_orig = calculate_categorical_emd(df, target_column=target, protected_attribute=protected_attribute)
    emd_female_orig = emd_orig.get(female_val, np.nan)
    emd_male_orig = emd_orig.get(male_val, np.nan)

    # NEW: deterministic RNGs for original p-values
    rng_female_orig = _rng_for(seed, dataset_name, "orig", "female")
    rng_male_orig   = _rng_for(seed, dataset_name, "orig", "male")


    # # --- Original EMD p-values (permutation test) ---
    _, p_female_emd_orig, _ = permutation_test_by_shuffling_labels(
        df, target, protected_attribute, female_val, n_perm, rng=rng_female_orig
    )
    _, p_male_emd_orig,  _  = permutation_test_by_shuffling_labels(
        df, target, protected_attribute, male_val,  n_perm, rng=rng_male_orig
    )


    # Binary search for unbiased dataset
    low, high = 0, min(n_max, len(removable_indices))
    best_k = None
    final_emds = {}
    best_pf = None
    best_pm = None

    while low <= high:
        mid = (low + high) // 2
        df_try = df.drop(index=removable_indices[:mid])

        rng_f_mid = _rng_for(seed, dataset_name, "search", "female", mid)
        rng_m_mid = _rng_for(seed, dataset_name, "search", "male",   mid)

        obs_f, p_f, _ = permutation_test_by_shuffling_labels(
            df_try, target, protected_attribute, female_val, n_perm, rng=rng_f_mid
        )
        obs_m, p_m, _ = permutation_test_by_shuffling_labels(
            df_try, target, protected_attribute, male_val,   n_perm, rng=rng_m_mid
        )

        if p_f > threshold and p_m > threshold:
            best_k = mid
            best_pf, best_pm = p_f, p_m 
            final_emds = calculate_categorical_emd(df_try, target_column=target, protected_attribute=protected_attribute)
            high = mid - 1
        else:
            low = mid + 1

    # --- If NO unbiased set found ---
    if best_k is None:
        results = {
            "Dataset name": dataset_name,
            "Total rows in dataset": total_rows,
            "Total Males": (df[protected_attribute] == male_val).sum(),
            "Total Females": (df[protected_attribute] == female_val).sum(),
            "Original EMD female": emd_female_orig,
            "Original EMD_pval_female": p_female_emd_orig,
            "Original EMD male": emd_male_orig,
            "Original EMD_pval_male":  p_male_emd_orig,
            "Unbiased EMD female": np.nan,
            "Unbiased EMD_pval_female": np.nan,
            "Unbiased EMD male": np.nan,
            "Unbiased EMD_pval_male":   np.nan,

            "Number of samples to be deleted": "Not found",
            "Gender of data to be deleted": remove_gender_value if isinstance(remove_gender_value, str) else ("Female" if remove_gender_value==female_val else "Male"),
            "Outcome of data to be deleted": remove_outcome_value,
            "Unbiased dataset found?": "No",
        }

        # Original only (unbiased does not exist)
        for name, fac in model_factories.items():
            print(f"\n>>> Starting model: {name} on ORIGINAL (biased) dataset")

            pvals = evaluate_model_fairness_pvals_split_train(
                fac, df, target, protected_attribute, female_val, male_val, n_splits=20, log_values=False
            )


            results[f"Original {name}_TPR_pval"]   = pvals.get("TPR_pval", np.nan)
            results[f"Original {name}_FPR_pval"]   = pvals.get("FPR_pval", np.nan)
            results[f"Original {name}_FN/FP_pval"] = pvals.get("FN/FP_pval", np.nan)

            results[f"Unbiased {name}_TPR_pval"]   = np.nan
            results[f"Unbiased {name}_FPR_pval"]   = np.nan
            results[f"Unbiased {name}_FN/FP_pval"] = np.nan

            # 80:20 TEST metrics on the biased/original dataset
            orig_metrics = _test_metrics_for_model(fac, df, target, random_state=42)
            results[f"Original {name}_Test_Accuracy"]  = orig_metrics["test_acc"]
            results[f"Original {name}_Test_Precision"] = orig_metrics["test_prec"]
            results[f"Original {name}_Test_Recall"]    = orig_metrics["test_rec"]
            results[f"Original {name}_Test_F1"]        = orig_metrics["test_f1"]

            # Fill Unbiased with NaN since not found
            results[f"Unbiased {name}_Test_Accuracy"]  = np.nan
            results[f"Unbiased {name}_Test_Precision"] = np.nan
            results[f"Unbiased {name}_Test_Recall"]    = np.nan
            results[f"Unbiased {name}_Test_F1"]        = np.nan


            results[f"Original {name}_POS_RATE_pval"] = pvals.get("POS_RATE_pval", np.nan)
            results[f"Original {name}_BER_pval"]      = pvals.get("BER_pval", np.nan)

            results[f"Unbiased {name}_POS_RATE_pval"] = np.nan
            results[f"Unbiased {name}_BER_pval"]      = np.nan
            print(f"<<< Finished model: {name} on ORIGINAL (biased) dataset")
        return results



    unbiased_df = df.drop(index=removable_indices[:best_k])

    # Save to CSV
    output_filename = _unbiased_path(dataset_name, threshold)
    unbiased_df.to_csv(output_filename, index=False)
    print(f"Saved unbiased dataset to {output_filename}")

    p_female_emd_unb, p_male_emd_unb = best_pf, best_pm

    

    results = {
    "Dataset name": dataset_name,
    "Total rows in dataset": total_rows,
    "Total Males": (df[protected_attribute] == male_val).sum(),
    "Total Females": (df[protected_attribute] == female_val).sum(),
    "Original EMD female": emd_female_orig,
    "Original EMD_pval_female": p_female_emd_orig,
    "Original EMD male": emd_male_orig,
    "Original EMD_pval_male":  p_male_emd_orig,
    "Unbiased EMD female": final_emds.get(female_val, np.nan),
    "Unbiased EMD_pval_female": p_female_emd_unb,
    "Unbiased EMD male": final_emds.get(male_val, np.nan),
    "Unbiased EMD_pval_male":   p_male_emd_unb,
    "Number of samples to be deleted": best_k,
    "Gender of data to be deleted": remove_gender_value if isinstance(remove_gender_value, str) else ("Female" if remove_gender_value==female_val else "Male"),
    "Outcome of data to be deleted": remove_outcome_value,
    "Unbiased dataset found?": "Yes",

    
    }

    # Add p-values for original and unbiased data
    # Add TPR, FPR, FN/FP p-values for original and unbiased datasets
    # Original + Unbiased (split‑train evaluation)
    for name, fac in model_factories.items():
        print(f"\n>>> Starting model: {name} on ORIGINAL & UNBIASED datasets")

        pvals_orig = evaluate_model_fairness_pvals_split_train(
            fac, df, target, protected_attribute, female_val, male_val, n_splits=20, log_values=False
        )
        pvals_unb = evaluate_model_fairness_pvals_split_train(
            fac, unbiased_df, target, protected_attribute, female_val, male_val, n_splits=20, log_values=False
        )

        results[f"Original {name}_TPR_pval"]   = pvals_orig.get("TPR_pval", np.nan)
        results[f"Original {name}_FPR_pval"]   = pvals_orig.get("FPR_pval", np.nan)
        results[f"Original {name}_FN/FP_pval"] = pvals_orig.get("FN/FP_pval", np.nan)

        results[f"Unbiased {name}_TPR_pval"]   = pvals_unb.get("TPR_pval", np.nan)
        results[f"Unbiased {name}_FPR_pval"]   = pvals_unb.get("FPR_pval", np.nan)
        results[f"Unbiased {name}_FN/FP_pval"] = pvals_unb.get("FN/FP_pval", np.nan)

        # NEW: 80:20 TRAIN metrics for biased and unbiased datasets
        orig_metrics = _test_metrics_for_model(fac, df, target, random_state=42)
        unb_metrics  = _test_metrics_for_model(fac, unbiased_df, target, random_state=42)

        results[f"Original {name}_Test_Accuracy"]  = orig_metrics["test_acc"]
        results[f"Original {name}_Test_Precision"] = orig_metrics["test_prec"]
        results[f"Original {name}_Test_Recall"]    = orig_metrics["test_rec"]
        results[f"Original {name}_Test_F1"]        = orig_metrics["test_f1"]

        results[f"Unbiased {name}_Test_Accuracy"]  = unb_metrics["test_acc"]
        results[f"Unbiased {name}_Test_Precision"] = unb_metrics["test_prec"]
        results[f"Unbiased {name}_Test_Recall"]    = unb_metrics["test_rec"]
        results[f"Unbiased {name}_Test_F1"]        = unb_metrics["test_f1"]


        results[f"Original {name}_POS_RATE_pval"] = pvals_orig.get("POS_RATE_pval", np.nan)
        results[f"Original {name}_BER_pval"]      = pvals_orig.get("BER_pval", np.nan)

        results[f"Unbiased {name}_POS_RATE_pval"] = pvals_unb.get("POS_RATE_pval", np.nan)
        results[f"Unbiased {name}_BER_pval"]      = pvals_unb.get("BER_pval", np.nan)

        print(f"<<< Finished model: {name} on ORIGINAL & UNBIASED datasets")

    return results

def evaluate_model_fairness_pvals_split_train(model_factory, df, target, protected_attribute, female_val, male_val, n_splits=20, log_values=False):
    """
    Train separate models for each gender and run Wilcoxon rank-sum on TPR, FPR, and FN/FP etc.
    
    Parameters:
        model_factory: callable -> returns a new untrained model instance
        df: pandas DataFrame
        target: str -> target column name
        protected_attribute: str -> gender column name
        female_val, male_val: values in gender column representing female and male
        n_splits: int -> folds for stratified CV
        log_values: bool -> print fold metrics if True
    
    Returns:
        dict with TPR_pval, FPR_pval, FN/FP_pval
    """
    data = df.copy()

    # Encode target to 0/1 if needed
    y_raw = data[target]
    if y_raw.dtype == object or not set(pd.unique(y_raw)).issubset({0, 1}):
        le = LabelEncoder()
        data[target] = le.fit_transform(y_raw)

    # Function to compute metrics for a single gender subset
    def get_metrics_for_group(group_val):
        subset = data[data[protected_attribute] == group_val].copy()
        X = pd.get_dummies(subset.drop(columns=[target, protected_attribute]), drop_first=True)
        y = subset[target].astype(int)

        # Remove missing values
        full = pd.concat([X, y], axis=1).dropna()
        X = full.drop(columns=[target])
        y = full[target]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        tprs, fprs, fnfp_ratios = [], [], []
        pos_rates, bers = [], []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = model_factory()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            epsilon = 1e-6
            fnfp = fn / (fp + epsilon)
            pos_rate = float(np.mean(y_pred == 1))
            fnr = 1.0 - tpr  # (= fn/(tp+fn) when denom>0, consistent with tpr above)
            ber = 0.5 * (fnr + fpr)

            tprs.append(tpr)
            fprs.append(fpr)
            fnfp_ratios.append(fnfp)
            pos_rates.append(pos_rate)
            bers.append(ber)

            if log_values:
                print(f"Fold {fold} ({group_val}) - TPR: {tpr:.4f}, FPR: {fpr:.4f}, POS: {pos_rate:.4f}, BER: {ber:.4f}, FN/FP: {fnfp:.4f}")


        return tprs, fprs, fnfp_ratios, pos_rates, bers

    # Get metrics for each gender
    female_tprs, female_fprs, female_fnfp, female_pos, female_ber = get_metrics_for_group(female_val)
    male_tprs,   male_fprs,   male_fnfp,   male_pos,   male_ber   = get_metrics_for_group(male_val)


    def safe_wilcoxon_ranksum(a, b):
        '''
        Returns:
            p-value from two-sided test
            '''
         # Clean arrays (remove NaN/Inf)
        a_clean = [x for x in a if np.isfinite(x)]
        b_clean = [x for x in b if np.isfinite(x)]
        
        if len(a_clean) > 1 and len(b_clean) > 1:
            # Mann-Whitney U test (two-sided)
            u_stat, p_val = mannwhitneyu(a_clean, b_clean, alternative="two-sided")
            
            
            
            return p_val
        
        return np.nan
    
    p_tpr = safe_wilcoxon_ranksum(female_tprs, male_tprs)
    p_fpr = safe_wilcoxon_ranksum(female_fprs, male_fprs)
    p_fnfp = safe_wilcoxon_ranksum(female_fnfp, male_fnfp)
    p_pos = safe_wilcoxon_ranksum(female_pos, male_pos)
    p_ber = safe_wilcoxon_ranksum(female_ber, male_ber)


    return {
        "TPR_pval": p_tpr,
        
        "FPR_pval": p_fpr,
        
        "FN/FP_pval": p_fnfp,
        
        "POS_RATE_pval": p_pos,
        "BER_pval": p_ber,

    }



# ---------- NEW HELPERS FOR ORCHESTRATION ----------

def _compute_original_emd_and_pvals(df, dataset_name, target, protected_attribute, female_val, male_val, n_perm=10000, seed=42):
    """Return dict with original EMDs and permutation p-values (deterministic)."""
    # Ensure binary target like elsewhere
    data = df.copy()
    if data[target].dtype == object or not set(pd.unique(data[target])).issubset({0, 1}):
        le = LabelEncoder()
        data[target] = le.fit_transform(data[target])

    emd_orig = calculate_categorical_emd(data, target_column=target, protected_attribute=protected_attribute)
    emd_f = emd_orig.get(female_val, np.nan)
    emd_m = emd_orig.get(male_val,   np.nan)

    rng_f = _rng_for(seed, dataset_name, "orig", "female")
    rng_m = _rng_for(seed, dataset_name, "orig", "male")

    _, p_f, _ = permutation_test_by_shuffling_labels(data, target, protected_attribute, female_val, n_perm, rng=rng_f)
    _, p_m, _ = permutation_test_by_shuffling_labels(data, target, protected_attribute, male_val,   n_perm, rng=rng_m)

    return {"Original EMD female": emd_f, "Original EMD male": emd_m,
            "Original EMD_pval_female": p_f, "Original EMD_pval_male": p_m}

def _eval_models_for_df(df, label_prefix, target, protected_attribute, female_val, male_val, seed=42):
    """Run your per-fold fairness Wilcoxon tests + 80:20 TEST metrics for ALL models. Returns a dict of results keyed with prefix."""
    
    model_factories = MODEL_FACTORIES(seed)
    out = {}
    for name, fac in model_factories.items():
        print(f"\n>>> Starting model: {name} on {label_prefix} dataset")
        pvals = evaluate_model_fairness_pvals_split_train(
            fac, df, target, protected_attribute, female_val, male_val, n_splits=20, log_values=False
        )

        # Fairness p-vals
        out[f"{label_prefix} {name}_TPR_pval"]   = pvals.get("TPR_pval", np.nan)
        out[f"{label_prefix} {name}_FPR_pval"]   = pvals.get("FPR_pval", np.nan)
        out[f"{label_prefix} {name}_FN/FP_pval"] = pvals.get("FN/FP_pval", np.nan)
        out[f"{label_prefix} {name}_POS_RATE_pval"] = pvals.get("POS_RATE_pval", np.nan)
        out[f"{label_prefix} {name}_BER_pval"]      = pvals.get("BER_pval", np.nan)

        metrics = _test_metrics_for_model(fac, df, target, random_state=42)

        # Test metrics instead of Train
        out[f"{label_prefix} {name}_Test_Accuracy"]  = metrics["test_acc"]
        out[f"{label_prefix} {name}_Test_Precision"] = metrics["test_prec"]
        out[f"{label_prefix} {name}_Test_Recall"]    = metrics["test_rec"]
        out[f"{label_prefix} {name}_Test_F1"]        = metrics["test_f1"]

        print(f"<<< Finished model: {name} on {label_prefix} dataset")
    return out

def _rename_unbiased_keys(d, tag):
    """Take the dict returned by get_unbiased_emd_info(threshold=...), keep only the 'Unbiased ...' bits and rename to e.g. 'Unbiased05 ...'."""
    out = {}
    prefix = f"Unbiased{tag} "
    for k, v in d.items():
        if k.startswith("Unbiased "):
            out[prefix + k[len("Unbiased "):]] = v
        # Also capture counts/info columns with clearer names
        elif k == "Number of samples to be deleted":
            out[f"Unbiased{tag} Samples to delete"] = v
        elif k == "Gender of data to be deleted":
            out[f"Unbiased{tag} Deleted gender"] = v
        elif k == "Outcome of data to be deleted":
            out[f"Unbiased{tag} Deleted outcome"] = v
        elif k == "Unbiased dataset found?":
            out[f"Unbiased{tag} dataset found?"] = v
    return out

def _unbiased_path(dataset_name, threshold):
    """Build a unique filename per alpha (e.g., unbiased_a05_D51.csv)."""
    tag = f"a{int(round(threshold*100)):02d}"   # 0.05 -> a05, 0.10 -> a10
    return f"unbiased_{tag}_{dataset_name}.csv"

def _test_metrics_for_model(model_factory, df, target, random_state=42, test_size=0.2):
    """
    Do a stratified split, fit on TRAIN, compute metrics on the TEST portion.
    Returns: {'test_acc', 'test_prec', 'test_rec', 'test_f1'}
    """
    X, y = _prep_xy(df, target)
    # Guard: need at least 2 classes and a few rows
    if len(np.unique(y)) < 2 or len(y) < 5:
        return {"test_acc": np.nan, "test_prec": np.nan, "test_rec": np.nan, "test_f1": np.nan}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = model_factory()
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)

    test_acc  = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec  = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1   = f1_score(y_test, y_pred_test, zero_division=0)

    return {
        "test_acc":  test_acc,
        "test_prec": test_prec,
        "test_rec":  test_rec,
        "test_f1":   test_f1,
    }


def _format_tag(threshold: float) -> str:
    """0.05 -> '05', 0.10 -> '10', 0.2 -> '20', 0.30 -> '30'."""
    return f"{int(round(threshold * 100)):02d}"

def process_dataset_pipeline(csv_path, target, protected_attribute, female_val, male_val,
                             n_perm=10000, seed=42, alpha_base=0.05,
                             thresholds=(0.05, 0.10, 0.20, 0.30)):
    """
    Steps:
      1) Compute original EMD + p-vals; if NOT significantly biased at alpha_base, record and stop.
      2) Evaluate fairness for models on ORIGINAL.
      3) For each threshold in `thresholds`, try to build an 'Unbiased@threshold' dataset
         (via get_unbiased_emd_info) and evaluate models if found.
    Returns one combined row (dict) for Excel.
    """
    df = pd.read_csv(csv_path)
    dataset_name = os.path.basename(csv_path).replace(".csv", "")
    total_rows = len(df)

    # --- Original EMDs + p-values ---
    orig = _compute_original_emd_and_pvals(df, dataset_name, target, protected_attribute,
                                           female_val, male_val, n_perm, seed)
    p_f = orig["Original EMD_pval_female"]; p_m = orig["Original EMD_pval_male"]
    print(f"\n=== {dataset_name}: Original EMD p-values: female={p_f:.4g}, male={p_m:.4g} @ alpha={alpha_base}")

    row = {
        "Dataset name": dataset_name,
        "Total rows in dataset": total_rows,
        "Total Males": (df[protected_attribute] == male_val).sum(),
        "Total Females": (df[protected_attribute] == female_val).sum(),
        **orig
    }

    # --- Always evaluate models on ORIGINAL (baseline fairness) ---
    row.update(_eval_models_for_df(df, "Original", target, protected_attribute, female_val, male_val, seed=seed))

    # --- If not significantly biased at alpha_base, stop here ---
    if (p_f > alpha_base) and (p_m > alpha_base):
        print(f"✔ Data NOT significantly biased at alpha={alpha_base}. Skipping unbiasing.")
        row["Unbiasing skipped (alpha)"] = alpha_base
        return row

    print(f"✱ Data IS significantly biased at alpha={alpha_base}. Proceeding to unbiasing runs.")

    # --- Iterate thresholds (e.g., 0.05, 0.10, 0.20, 0.30) ---
    for thr in thresholds:
        tag = _format_tag(thr)  # e.g., '05', '10', '20', '30'
        print(f"\n→ Attempting Unbiased@{thr:.2f}")
        res = get_unbiased_emd_info(
            csv_path, target, protected_attribute, female_val, male_val,
            threshold=thr, n_perm=n_perm, seed=seed
        )

        # Keep only “Unbiased ...” keys, renamed to “Unbiased{tag} ...”
        row.update(_rename_unbiased_keys(res, tag))

        # If dataset found, evaluate models on it
        if row.get(f"Unbiased{tag} dataset found?") == "Yes":
            unb_path = _unbiased_path(dataset_name, thr)  # reuses your existing path formatter
            df_unb = pd.read_csv(unb_path)
            row.update(_eval_models_for_df(df_unb, f"Unbiased{tag}", target,
                                           protected_attribute, female_val, male_val, seed=seed))
        else:
            print(f"Unbiased@{thr:.2f} not found; skipping model eval at this threshold.")

    return row



datasets = [
    ("datasets/new_provided/D73.csv", "hospital_outcome_1alive_0dead", "sex_0male_1female", 1, 0),
]

summary_rows = []
for path, target, prot, f, m in datasets:
    row = process_dataset_pipeline(path, target, prot, f, m, n_perm=10000, seed=42, alpha_base=0.05)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel("D73_biased_datasets_summary.xlsx", index=False)
print("✅ Wrote D73_biased_datasets_summary.xlsx")
