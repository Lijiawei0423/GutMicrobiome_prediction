import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description='Parallel bootstrap Pearson correlation for all bacteria in Analyst_summary')
    parser.add_argument('--analysis_type', type=int, default=2, help="AnalysisType to filter in Analyst_summary.csv (default: 2)")
    parser.add_argument('--dpath', type=str, default='/home1/LIJW/AbundanceData/', help="Path containing Analyst_summary.csv")
    parser.add_argument('--result_path', type=str, default='/home1/LIJW/JiangNan_results_all', help="Root path of prediction results")
    parser.add_argument('--n_boot', type=int, default=1000, help="Number of bootstrap resamples per clade")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed")
    parser.add_argument('--n_jobs', type=int, default=0, help="Number of parallel workers (0 => use os.cpu_count())")
    parser.add_argument('--out_csv', type=str, default='analysis_type2_bootstrap_pearson.csv', help="Output CSV filename")
    return parser.parse_args()

def percentile_ci(x, alpha=0.05):
    lo = np.quantile(x, alpha / 2)
    hi = np.quantile(x, 1 - alpha / 2)
    return lo, hi

def _stable_int_from_name(name: str) -> int:
    s = 0
    for i, ch in enumerate(name):
        s = (s * 131 + ord(ch)) % 2_147_483_647
    return s

def run_one_clade(clade_name: str, result_path: str, n_boot: int, seed: int):
    rs_dir = os.path.join(result_path, clade_name, 'S3_Pred')

    dfs = []
    missing = []
    for cv_id in range(5):
        rs_path = os.path.join(rs_dir, f'fold{cv_id}_risk_scores.csv')
        if not os.path.exists(rs_path):
            missing.append(rs_path)
            continue
        df = pd.read_csv(rs_path)
        if 'target' not in df.columns or 'target_log' not in df.columns or 'pred_value' not in df.columns:
            return None, None, {
                "clade_name": clade_name,
                "status": "fail",
                "message": f"Missing required columns (target/target_log/pred_value) in file: {rs_path}",
            }
        df = df[df['target'] != 0][['target_log', 'pred_value']]
        dfs.append(df)

    if missing:
        return None, None, {
            "clade_name": clade_name,
            "status": "fail",
            "message": f"Missing fold files: {len(missing)} (example: {missing[0]})",
        }

    all_df = pd.concat(dfs, ignore_index=True).dropna()
    y_true = all_df['target_log'].to_numpy()
    y_pred = all_df['pred_value'].to_numpy()
    n = len(y_true)

    if n < 2:
        return None, None, {
            "clade_name": clade_name,
            "status": "fail",
            "message": f"Insufficient valid samples (n={n})",
        }


    point_corr, _ = pearsonr(y_true, y_pred)

    # Create a deterministic per-clade seed to keep bootstrap runs reproducible.
    clade_seed = (seed + _stable_int_from_name(clade_name)) % 2_147_483_647
    rng = np.random.default_rng(clade_seed)

    boot_corr = []
    attempts = 0
    while len(boot_corr) < n_boot:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if np.std(yt) == 0 or np.std(yp) == 0:
            continue
        c, _ = pearsonr(yt, yp)
        if np.isfinite(c):
            boot_corr.append(c)

        # Guard against pathological cases that rarely produce valid bootstrap samples.
        if attempts > n_boot * 50 and len(boot_corr) < max(10, n_boot // 10):
            return None, None, {
                "clade_name": clade_name,
                "status": "fail",
                "message": f"Too few valid bootstrap samples: {len(boot_corr)}/{n_boot} (likely due to zero variance)",
            }

    boot_corr = np.array(boot_corr)
    lo, hi = percentile_ci(boot_corr, alpha=0.05)

    # summary
    summary_row = {
        "clade_name": clade_name,
        "status": "ok",
        "n_samples": n,
        "point_r": float(point_corr),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "boot_mean": float(boot_corr.mean()),
        "boot_sd": float(boot_corr.std(ddof=1)),
        "attempts": int(attempts),
        "seed_used": int(clade_seed),
    }

    # Bootstrap results.
    boot_df = pd.DataFrame({
        "clade_name": clade_name,
        "bootstrap_id": np.arange(n_boot),
        "bootstrap_r": boot_corr
    })

    return summary_row, boot_df, None

def main():
    args = parse_args()
    analysis_type = args.analysis_type
    dpath = args.dpath
    result_path = args.result_path
    n_boot = args.n_boot
    seed = args.seed
    n_jobs = args.n_jobs if args.n_jobs and args.n_jobs > 0 else (os.cpu_count() or 1)

    summary_path = os.path.join(dpath, 'Analyst_summary.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"File not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    if 'AnalysisType' not in summary_df.columns or 'Analyst' not in summary_df.columns:
        raise RuntimeError("Analyst_summary.csv must contain the columns AnalysisType and Analyst.")

    summary_df = summary_df[summary_df['AnalysisType'].isin([analysis_type])]
    xijun_lst = summary_df['Analyst'].astype(str).tolist()
    xijun_lst = [x for x in xijun_lst if x and x.lower() != 'nan']
    xijun_lst = sorted(set(xijun_lst))
    if len(xijun_lst) == 0:
        raise RuntimeError(f"No analysts found for AnalysisType={analysis_type}.")

    print(f"AnalysisType={analysis_type}, total clades={len(xijun_lst)}, n_boot={n_boot}, n_jobs={n_jobs}")

    results_summary = []
    boot_all = []
    failed = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {
            ex.submit(run_one_clade, clade, result_path, n_boot, seed): clade
            for clade in xijun_lst
        }

        for fut in as_completed(futures):
            clade = futures[fut]
            try:
                summary_row, boot_df, err = fut.result()
            except Exception as e:
                summary_row, boot_df, err = None, None, {
                    "clade_name": clade,
                    "status": "fail",
                    "message": f"Unexpected error: {repr(e)}"
                }

            if err is not None:
                failed.append(err)
                # print(f"[FAIL] {clade} {err['message']}")
                continue

            results_summary.append(summary_row)
            boot_all.append(boot_df)

            print(f"[OK] {clade} r={summary_row['point_r']:.4f} "
                f"CI=({summary_row['ci95_lo']:.4f},{summary_row['ci95_hi']:.4f}) "
                f"n={summary_row['n_samples']}")
    # Save summary.
    if len(results_summary) > 0:
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values("point_r", ascending=False)

        summary_out = args.out_csv
        summary_df.to_csv(summary_out, index=False)
        print(f"\nSaved summary results to: {summary_out}")

    # Save the full bootstrap distribution.
    if len(boot_all) > 0:
        boot_df_all = pd.concat(boot_all, ignore_index=True)
        boot_out = os.path.splitext(args.out_csv)[0] + "_bootstrap_all.csv"
        boot_df_all.to_csv(boot_out, index=False)
        print(f"Saved all bootstrap samples to: {boot_out}")

    # Save failures.
    if len(failed) > 0:
        fail_df = pd.DataFrame(failed)
        fail_out = os.path.splitext(args.out_csv)[0] + "_failed.csv"
        fail_df.to_csv(fail_out, index=False)
        print(f"Saved failure list to: {fail_out}")

if __name__ == "__main__":
    main()
