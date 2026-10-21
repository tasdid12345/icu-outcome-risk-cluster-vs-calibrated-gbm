"""Heart-failure mortality: k-means cluster-rate baseline vs class-balanced LightGBM with isotonic calibration."""
import io, json, sys, time, zipfile
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]  # randomized models are run with each seed
URL = "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip"
# 'time' = follow-up duration, only known after the outcome -> dropped to avoid leakage
DROP = ["time", "DEATH_EVENT"]
QUESTION = ("Can a class-balanced, isotonic-calibrated gradient-boosting model give better-ranked and "
            "better-calibrated patient mortality risk than a k-means cluster-rate baseline on a public ICU-style dataset?")
METHOD = ("Stratified 75/25 split (seed 42); preprocessing fit on train only. Baseline: k-means on standardized features, "
          "k chosen by silhouette on train, test patients get the train death rate of their cluster. Improved: "
          "class-balanced LightGBM (hyperparameters by 5-fold CV on train), isotonic calibration fit on out-of-fold train "
          "predictions. Test set used once. Seeds 42/43/44; mean and std reported.")


def download(url, tries=5):
    """GET with timeout=60; exponential backoff on HTTP 429/5xx and network errors."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"download failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)


def load():
    """Load the UCI Heart Failure Clinical Records CSV from the official archive zip."""
    with zipfile.ZipFile(io.BytesIO(download(URL))) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(name))
    return df.drop(columns=DROP), df["DEATH_EVENT"].to_numpy(int)


def baseline(Xtr, ytr, Xte, seed):
    """k-means cluster-rate model: test risk = training death rate of the assigned cluster."""
    sc = StandardScaler().fit(Xtr)  # scaler fit on train only
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    best = None
    for k in range(2, 9):  # choose k by silhouette on training data
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Ztr)
        s = silhouette_score(Ztr, km.labels_)
        if best is None or s > best[0]:
            best = (s, k, km)
    s, k, km = best
    rate = np.array([ytr[km.labels_ == c].mean() for c in range(k)])  # cluster death rates from train
    return rate[km.predict(Zte)], k, s


def search(Xtr, ytr):
    """Small hyperparameter grid, scored by 5-fold CV ROC-AUC on the training set only."""
    grid = {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1],
            "num_leaves": [7, 15], "min_child_samples": [5, 15]}
    gs = GridSearchCV(LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1), grid,
                      scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    gs.fit(Xtr, ytr)
    return gs.best_params_, gs.best_score_


def improved(Xtr, ytr, Xte, seed, params):
    """Balanced LightGBM + isotonic calibration fit on out-of-fold training predictions."""
    make = lambda: LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1, **params)
    oof = np.zeros(len(ytr))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr):
        oof[va] = make().fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, ytr)
    raw = make().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return iso.predict(raw)


def main():
    X, y = load()
    print(f"Loaded {len(y)} patients, {X.shape[1]} features, death rate {y.mean():.3f}")
    Xtr, Xte, ytr, yte = train_test_split(X.to_numpy(float), y, test_size=0.25, stratify=y, random_state=SEED)
    thr = ytr.mean()  # same decision threshold for both models: training prevalence
    print(f"Split: train {len(ytr)}, test {len(yte)}")

    params, cv_auc = search(Xtr, ytr)
    print(f"LightGBM best CV ROC-AUC {cv_auc:.3f} with {params}")

    rows = {"baseline": [], "improved": []}  # columns: ROC-AUC, Brier, balanced accuracy
    for s in SEEDS:
        pb, k, sil = baseline(Xtr, ytr, Xte, s)
        pi = improved(Xtr, ytr, Xte, s, params)
        for name, p in [("baseline", pb), ("improved", pi)]:
            rows[name].append([roc_auc_score(yte, p), brier_score_loss(yte, p),
                               balanced_accuracy_score(yte, (p >= thr).astype(int))])
        print(f"seed {s}: k-means k={k} (silhouette {sil:.3f}) | AUC baseline {rows['baseline'][-1][0]:.3f} "
              f"vs improved {rows['improved'][-1][0]:.3f}")

    A = {n: np.array(v) for n, v in rows.items()}
    b, i = A["baseline"].mean(0), A["improved"].mean(0)
    bs, is_ = A["baseline"].std(0), A["improved"].std(0)
    metrics = {
        "baseline_score": b[0], "improved_score": i[0], "improvement": i[0] - b[0],
        "baseline_score_std": bs[0], "improved_score_std": is_[0],
        "baseline_brier": b[1], "improved_brier": i[1],
        "baseline_brier_std": bs[1], "improved_brier_std": is_[1],
        "baseline_balanced_accuracy": b[2], "improved_balanced_accuracy": i[2],
        "improved_cv_roc_auc": cv_auc,
    }
    metrics = {key: float(v) for key, v in metrics.items()}
    if not all(np.isfinite(list(metrics.values()))):
        print("Non-finite metrics; failing.")
        sys.exit(1)

    # Optional chart: higher is better for all bars (Brier shown as 1 - Brier)
    x = np.arange(3)
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, [b[0], 1 - b[1], b[2]], 0.4, label="baseline (k-means cluster rate)")
    plt.bar(x + 0.2, [i[0], 1 - i[1], i[2]], 0.4, label="improved (balanced LightGBM + isotonic)")
    plt.xticks(x, ["ROC-AUC", "1 - Brier", "Balanced acc."])
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.title("Test-set comparison (mean over seeds)")
    plt.tight_layout()
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/comparison.png", dpi=120)

    out = {
        "question": QUESTION,
        "method": METHOD,
        "dataset": {
            "name": "UCI Heart Failure Clinical Records (ICU-style stand-in)",
            "source_url": "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records",
            "n_rows": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "ROC-AUC on held-out test set (mean over seeds 42, 43, 44)",
        "higher_is_better": True,
        "baseline": "k-means cluster analysis on standardized features; test risk = training death rate of cluster (k by silhouette)",
        "improved": "class-balanced LightGBM (CV-tuned on train) with isotonic calibration fit on out-of-fold train predictions",
        "metrics": metrics,
        "notes": ("Stand-in dataset, not the paper's pneumonia ICU data; the paper's result is not reproduced here. "
                  "Small sample, so test estimates are noisy. 'time' (follow-up duration) is excluded as post-outcome leakage. "
                  "Dataset is small enough that no subsampling was needed. Hyperparameters were chosen by CV on train only with seed 42. "
                  "Isotonic calibration uses out-of-fold predictions from 80% models, applied to the full-train model. "
                  "Brier and balanced accuracy use the training prevalence as threshold. Test set used once."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Wrote results.json")
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # non-zero exit if meaningful metrics cannot be produced
        print(f"FAILED: {e}")
        sys.exit(1)
