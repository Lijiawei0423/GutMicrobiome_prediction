import argparse
import numpy as np
import pandas as pd
import sys
import os
from collections import Counter

sys.path.append('/home1/LIJW/Xijun_full')
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

pd.options.mode.chained_assignment = None  


def normal_imp(mydict):
    mysum = float(sum(mydict.values()))
    if mysum == 0:
        return mydict
    for k in list(mydict.keys()):
        mydict[k] = mydict[k] / mysum
    return mydict


def parse_args():
    parser = argparse.ArgumentParser(description="Run model training with LightGBM, XGBoost, and CatBoost.")
    parser.add_argument('--clade_name', type=str, required=True, help="Clade name for classification")
    parser.add_argument('--action_type', type=int, required=True, help="Action type (integer value)")
    return parser.parse_args()


def train_fold(fold_id):

    print(f"[fold {fold_id}] start", flush=True)

    traindf = mydf[mydf['cv_id'] != fold_id].copy()
    traindf.reset_index(drop=True, inplace=True)

    cv_X_train = traindf[xijun_f_lst]
    cv_y_train = traindf["target_y"]


    lgb_imp_cv, xgb_imp_cv, cb_imp_cv = Counter(), Counter(), Counter()


    my_lgb = LGBMClassifier(
        objective='binary',
        metric='auc',
        is_unbalance=True,
        verbosity=-1,
        seed=MY_SEED,
        n_jobs=CPU_PER_FOLD
    )
    my_lgb.fit(cv_X_train, cv_y_train)
    totalgain_imp_lgb = dict(zip(
        my_lgb.booster_.feature_name(),
        my_lgb.booster_.feature_importance(importance_type='gain')
    ))
    lgb_imp_cv += Counter(normal_imp(totalgain_imp_lgb))


    cv_X_train_fill0 = cv_X_train.fillna(0)
    my_xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_jobs=CPU_PER_FOLD,
        tree_method="hist"
    )
    my_xgb.fit(cv_X_train_fill0, cv_y_train)
    totalgain_imp_xgb = my_xgb.get_booster().get_score(importance_type='gain')
    xgb_imp_cv += Counter(normal_imp(dict(totalgain_imp_xgb)))


    my_cb = CatBoostClassifier(
        loss_function='Logloss',
        verbose=0,
        thread_count=CPU_PER_FOLD,
        random_seed=MY_SEED,
        iterations=300  
    )
    my_cb.fit(cv_X_train_fill0, cv_y_train)
    totalgain_imp_cb = dict(zip(
        cv_X_train.columns,
        my_cb.get_feature_importance(type='FeatureImportance')
    ))
    cb_imp_cv += Counter(normal_imp(totalgain_imp_cb))


    feats = list(xijun_f_lst)
    lgb_imp_cv = {f: float(lgb_imp_cv.get(f, 0.0)) for f in feats}
    xgb_imp_cv = {f: float(xgb_imp_cv.get(f, 0.0)) for f in feats}
    cb_imp_cv  = {f: float(cb_imp_cv.get(f, 0.0))  for f in feats}


    merged_df = pd.DataFrame({
        "Analyst": feats,
        "LightGBM_Gain_cv": [lgb_imp_cv[f] for f in feats],
        "XGBoost_Gain_cv":  [xgb_imp_cv[f] for f in feats],
        "CatBoost_Gain_cv": [cb_imp_cv[f] for f in feats],
    })
    merged_df["Total_Gain_cv"] = merged_df[["LightGBM_Gain_cv", "XGBoost_Gain_cv", "CatBoost_Gain_cv"]].sum(axis=1)
    merged_df["Total_Gain_cv"] = merged_df["Total_Gain_cv"] / (merged_df["Total_Gain_cv"].sum() + 1e-12)
    merged_df.sort_values(by="Total_Gain_cv", ascending=False, inplace=True)

    fold_output_path = os.path.join(outpath, f'Importance_{action_type}_cv{fold_id}.csv')
    merged_df.to_csv(fold_output_path, index=False)
    print(f"[fold {fold_id}] finished. saved -> {fold_output_path}", flush=True)

    return lgb_imp_cv, xgb_imp_cv, cb_imp_cv


if __name__ == "__main__":
    args = parse_args()
    clade_name, action_type = args.clade_name, args.action_type

    MY_SEED = 2025
    dpath = '/home1/LIJW/AbundanceData/'
    outpath = os.path.join('/home1/LIJW/0129JiangNanResults', clade_name, 'S1_FS')
    os.makedirs(outpath, exist_ok=True)


    kf_df = pd.read_csv(dpath+'PhenotypeData.csv',usecols=['eid','cv_id'])
    al_df = pd.read_csv(os.path.join(dpath, f'Analyst_summary.csv'), usecols=['Analyst', 'AnalysisType'])
    al_df = al_df[al_df['AnalysisType'].isin([1, 2])]

    data_df = pd.read_csv(os.path.join(dpath, f'AbundanceData_preprocessed.csv'))


    mydf = pd.merge(data_df, kf_df, how='inner', on=['eid'])
    mydf['target_y'] = mydf[clade_name].apply(lambda x: 1 if x != 0 and not pd.isna(x) else 0)

    xijun_f_lst = [feature for feature in al_df['Analyst'].tolist() if feature != clade_name]
    print('Number of selected features in xijun_f_lst:', len(xijun_f_lst), flush=True)


    NUM_FOLDS = 5
    TOTAL_CPU = len(os.sched_getaffinity(0))
    CPU_PER_FOLD = max(1, TOTAL_CPU // 2)  
    print(f"TOTAL_CPU={TOTAL_CPU}, CPU_PER_FOLD={CPU_PER_FOLD}", flush=True)

    results = []
    for fold_id in range(NUM_FOLDS):
        results.append(train_fold(fold_id))


    lgb_imp, xgb_imp, cb_imp = Counter(), Counter(), Counter()
    for lgb_fold, xgb_fold, cb_fold in results:
        lgb_imp += Counter(lgb_fold)
        xgb_imp += Counter(xgb_fold)
        cb_imp += Counter(cb_fold)


    lgb_imp = normal_imp(dict(lgb_imp))
    xgb_imp = normal_imp(dict(xgb_imp))
    cb_imp  = normal_imp(dict(cb_imp))

    feats = list(xijun_f_lst)
    lgb_df = pd.DataFrame({"Analyst": feats, "LightGBM_Gain": [lgb_imp.get(f, 0.0) for f in feats]})
    xgb_df = pd.DataFrame({"Analyst": feats, "XGBoost_Gain":  [xgb_imp.get(f, 0.0) for f in feats]})
    cb_df  = pd.DataFrame({"Analyst": feats, "CatBoost_Gain": [cb_imp.get(f, 0.0) for f in feats]})

    merged_df = lgb_df.merge(xgb_df, on="Analyst", how="outer").merge(cb_df, on="Analyst", how="outer")
    merged_df.fillna(0, inplace=True)

    merged_df["Total_Gain"] = merged_df[["LightGBM_Gain", "XGBoost_Gain", "CatBoost_Gain"]].sum(axis=1)
    merged_df["Total_Gain"] = merged_df["Total_Gain"] / (merged_df["Total_Gain"].sum() + 1e-12)
    merged_df.sort_values(by="Total_Gain", ascending=False, inplace=True)

    all_output_path = os.path.join(outpath, f'Importance_{action_type}_all.csv')
    merged_df.to_csv(all_output_path, index=False)
    print(f'All folds finished. Final results saved -> {all_output_path}', flush=True)
