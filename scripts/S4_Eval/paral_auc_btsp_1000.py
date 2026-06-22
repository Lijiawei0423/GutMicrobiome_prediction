import pandas as pd
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(description='Parallel Bootstrap AUC CI for all species')
    parser.add_argument('--n_boot', type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed")
    parser.add_argument('--out', type=str, default='auc_ci_summary.csv', help="Output csv path")
    parser.add_argument('--analysis_type', type=int, default=1, help="Filter Analyst_summary.csv by AnalysisType")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel workers (-1 = all cores)")
    parser.add_argument('--backend', type=str, default='loky', choices=['loky', 'multiprocessing', 'threading'],
                        help="Joblib backend (loky recommended)")
    return parser.parse_args()

def compute_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))

def percentile_ci(x, alpha=0.05):
    lo = np.quantile(x, alpha/2)
    hi = np.quantile(x, 1 - alpha/2)
    return float(lo), float(hi)

def eval_one_species_auc_ci(clade_name, result_path, n_boot, seed_one):
    rng = np.random.default_rng(seed_one)

    all_df = []
    for cv_id in range(5):
        rs_path = os.path.join(result_path, clade_name, 'S3_Pred', f'fold{cv_id}_risk_scores.csv')
        if not os.path.exists(rs_path):
            return None, None, f"missing_file: {rs_path}"

        df = pd.read_csv(rs_path)
        if not {"target_binary", "risk_score"}.issubset(df.columns):
            return None, None, f"missing_columns in {rs_path}"

        all_df.append(df[["target_binary", "risk_score"]])

    all_df = pd.concat(all_df, ignore_index=True)
    y_true = all_df["target_binary"].astype(int).to_numpy()
    y_score = all_df["risk_score"].to_numpy()

    auc_point = compute_auc(y_true, y_score)
    if auc_point is None:
        return None, None, "single_class_in_all_data"

    n = len(y_true)
    boot_auc = []

    while len(boot_auc) < n_boot:
        idx = rng.integers(0, n, size=n)
        auc_b = compute_auc(y_true[idx], y_score[idx])
        if auc_b is None:
            continue
        boot_auc.append(auc_b)

    boot_auc = np.asarray(boot_auc, dtype=float)
    lo, hi = percentile_ci(boot_auc, alpha=0.05)

    # Summary row.
    summary_row = {
        "Species": clade_name,
        "AUC": auc_point,
        "AUC_CI_low": lo,
        "AUC_CI_high": hi,
        "n_samples_allfolds": int(n),
    }

    # Bootstrap results.
    boot_df = pd.DataFrame({
        "Species": clade_name,
        "bootstrap_id": np.arange(n_boot),
        "bootstrap_auc": boot_auc
    })

    return summary_row, boot_df, None

def main():
    args = parse_args()
    n_boot = args.n_boot
    seed = args.seed
    out_path = args.out
    analysis_type = args.analysis_type
    n_jobs = args.n_jobs
    backend = args.backend

    dpath = '/home1/LIJW/AbundanceData/'
    summary_df = pd.read_csv(os.path.join(dpath, 'Analyst_summary.csv'))
    summary_df = summary_df[summary_df['AnalysisType'].isin([analysis_type])]
    xijun_lst = summary_df['Analyst'].astype(str).tolist()
    xijun_lst = [x for x in xijun_lst if x and x.lower() != 'nan']
    xijun_lst = sorted(set(xijun_lst))


    result_path = '/home1/LIJW/JiangNan_results_all'

    # Run tasks in parallel.
    # tqdm shows progress and joblib handles parallel execution.
    tasks = (
        delayed(eval_one_species_auc_ci)(clade, result_path, n_boot, seed + i)
        for i, clade in enumerate(xijun_lst)
    )

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        tqdm(tasks, total=len(xijun_lst), desc="Bootstrap AUC CI", unit="species")
    )

    rows = []
    boot_all = []
    skipped = []
    for summary_row, boot_df, err in results:
        if err is not None:
            skipped.append({"reason": err})
            continue

        rows.append(summary_row)
        boot_all.append(boot_df)

    out_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    out_df.to_csv(out_path, index=False)
    # Save all bootstrap samples.
    boot_df_all = pd.concat(boot_all, ignore_index=True)
    boot_path = os.path.splitext(out_path)[0] + "_bootstrap_all.csv"
    boot_df_all.to_csv(boot_path, index=False)

    print(f"\n[OK] saved summary: {out_path}")
    print(f"[OK] saved bootstrap distribution: {boot_path}")
    print(f"ok={len(rows)} skipped={len(skipped)}")
    

    if skipped:
        skip_path = os.path.splitext(out_path)[0] + "_skipped.csv"
        pd.DataFrame(skipped).to_csv(skip_path, index=False)

    print(f"\n[OK] saved: {out_path} | ok={len(rows)} skipped={len(skipped)}")
    if skipped:
        print(f"[WARN] saved skipped list: {skip_path}")

if __name__ == "__main__":
    main()
