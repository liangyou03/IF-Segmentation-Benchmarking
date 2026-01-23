#!/usr/bin/env python3
"""Pooled (section-agnostic) 1-vs-rest Logistic Regression with repeated splits."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from _feat_utils import FEAT_MINIMAL, pick_auto_features

def make_clf():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"),
    )

def summarize_auc(df_rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ct, sub in df_rep.groupby("cell_type"):
        for metric in ["AUROC","AUPRC"]:
            vals = sub[metric].dropna().values; n=len(vals)
            if n==0: 
                rows.append({"cell_type":ct,"metric":metric,"mean":np.nan,"std":np.nan,"stderr":np.nan,"ci95_low":np.nan,"ci95_high":np.nan,"n_valid":0}); 
                continue
            mu = float(np.mean(vals)); sd = float(np.std(vals, ddof=1)) if n>1 else 0.0
            se = sd/np.sqrt(n); ci=1.96*se
            rows.append({"cell_type":ct,"metric":metric,"mean":mu,"std":sd,"stderr":se,"ci95_low":mu-ci,"ci95_high":mu+ci,"n_valid":n})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Pooled Logistic OVR baseline with repeats.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_repeats", type=int, default=100)
    ap.add_argument("--feature_mode", choices=["auto","minimal"], default="auto")
    ap.add_argument("--save_last_preds", action="store_true")
    ap.add_argument("--shuffle_labels", action="store_true", help="Leakage check: train/test on randomly permuted labels.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    if "qc_pass" in df.columns: df = df[df["qc_pass"]==1].copy()
    if args.feature_mode=="minimal":
        feat_cols = [c for c in FEAT_MINIMAL if c in df.columns]
        if len(feat_cols)<3: raise ValueError("Too few minimal features. Use --feature_mode auto.")
    else:
        feat_cols = pick_auto_features(df)

    label_col, group_col, roi_col = "unified_label","image_id","roi_id"
    need = [label_col, group_col, roi_col] + feat_cols
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"Missing columns: {miss}")
    if len(df)==0: raise ValueError("No rows after QC.")

    labels = sorted(df[label_col].astype(str).unique())
    per_rep, last_preds, last_split = [], [], None

    # Choose labels (original vs shuffled)
    rng = np.random.default_rng(args.random_state)
    base_y = df[label_col].astype(str).values
    if args.shuffle_labels:
        df["_label_for_training"] = rng.permutation(base_y)
        labcol = "_label_for_training"
    else:
        labcol = label_col

    for r in range(args.n_repeats):
        seed = args.random_state + r
        y_all = df[labcol].astype(str).values
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=seed)
        tr_idx, te_idx = next(sss.split(df, y=y_all))
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        Xtr, Xte = tr[feat_cols].to_numpy(float), te[feat_cols].to_numpy(float)
        clf = make_clf()

        for target in labels:
            ytr = (tr[labcol].values == target).astype(int)
            yte = (te[labcol].values == target).astype(int)
            if len(np.unique(yte))<2 or ytr.sum()==0:
                per_rep.append({"repeat":r,"seed":seed,"cell_type":target,
                                "n_train_pos":int(ytr.sum()),"n_train_total":len(ytr),
                                "n_test_pos":int(yte.sum()),"n_test_total":len(yte),
                                "AUROC":np.nan,"AUPRC":np.nan,"TP":np.nan,"FP":np.nan,"TN":np.nan,"FN":np.nan,
                                "note":"skipped_no_positive_in_split"})
                continue
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:,1]; yhat=(proba>=0.5).astype(int)
            auroc = roc_auc_score(yte, proba); auprc = average_precision_score(yte, proba)
            tn,fp,fn,tp = confusion_matrix(yte, yhat, labels=[0,1]).ravel()
            per_rep.append({"repeat":r,"seed":seed,"cell_type":target,
                            "n_train_pos":int(ytr.sum()),"n_train_total":len(ytr),
                            "n_test_pos":int(yte.sum()),"n_test_total":len(yte),
                            "AUROC":auroc,"AUPRC":auprc,"TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn),"note":""})
            if args.save_last_preds and r==args.n_repeats-1:
                last_preds.append(pd.DataFrame({
                    "cell_type":target,"image_id":te[group_col].astype(str).values,"roi_id":te[roi_col].astype(str).values,
                    "y_true":yte,"y_score":proba,"y_pred":yhat,"seed":seed,"repeat":r
                }))
        if args.save_last_preds and r==args.n_repeats-1:
            last_split = pd.concat([
                tr[[label_col, group_col, roi_col]].assign(set="train", repeat=r, seed=seed),
                te[[label_col, group_col, roi_col]].assign(set="test",  repeat=r, seed=seed),
            ], ignore_index=True)

    per_rep_df = pd.DataFrame(per_rep); per_rep_df.to_csv(outdir/"metrics_per_repeat.csv", index=False)
    summarize_auc(per_rep_df).to_csv(outdir/"metrics_summary.csv", index=False)
    if args.save_last_preds and last_preds:
        pd.concat(last_preds, ignore_index=True).to_csv(outdir/"predictions_last_repeat.csv", index=False)
        last_split.to_csv(outdir/"split_last_repeat.csv", index=False)
    print(f"[OK] per-repeat -> {outdir/'metrics_per_repeat.csv'}")
    print(f"[OK] summary    -> {outdir/'metrics_summary.csv'}")

if __name__ == "__main__":
    main()
