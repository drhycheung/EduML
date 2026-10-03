"""Train EduML models on xAPI-Edu-Data and export model/model.json.

Mirrors the EnvML pipeline: a regression target plus a 3-class classifier,
both gradient-boosted trees, serialised to flat arrays for browser traversal.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

SEED = 42
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(f'{HERE}/data/xAPI-Edu-Data.csv')

# Target: performance level  L->0, M->1, H->2
LEVEL = {'L': 0, 'M': 1, 'H': 2}
CLASS_NAMES = ['Low', 'Medium', 'High']
df['perf'] = df['Class'].map(LEVEL).astype(float)
df['absence_high'] = (df['StudentAbsenceDays'] == 'Above-7').astype(float)
df['semester_s'] = (df['Semester'] == 'S').astype(float)

FEATURES = ['raisedhands', 'VisITedResources', 'AnnouncementsView',
            'Discussion', 'absence_high', 'semester_s']

X = df[FEATURES].values.astype(float)
y_reg = df['perf'].values.astype(float)
y_cls = df['perf'].values.astype(int)

input_ranges = {f: [float(X[:, i].min()), float(X[:, i].max())]
                for i, f in enumerate(FEATURES)}

# Defaults: a moderately engaged, attending student in semester F.
defaults = {f: float(np.median(X[:, i])) for i, f in enumerate(FEATURES)}
defaults['absence_high'] = 0.0
defaults['semester_s'] = 0.0

# ---------------- Models ----------------
reg_params = dict(n_estimators=250, max_depth=3, learning_rate=0.06, random_state=SEED)
clf_params = dict(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=SEED)

reg = GradientBoostingRegressor(**reg_params)
reg.fit(X, y_reg)
clf = GradientBoostingClassifier(**clf_params)
clf.fit(X, y_cls)

# ---------------- Out-of-fold evaluation (5-fold shuffled) ----------------
cv_folds = 5
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
oof_reg = np.zeros(len(y_reg))
oof_cls_pred = np.zeros(len(y_cls), dtype=int)
oof_cls_prob = np.zeros((len(y_cls), 3))
fold_test_sizes, fold_train_sizes = [], []
for tr, te in kf.split(X):
    fold_train_sizes.append(int(len(tr)))
    fold_test_sizes.append(int(len(te)))
    r = GradientBoostingRegressor(**reg_params).fit(X[tr], y_reg[tr])
    oof_reg[te] = r.predict(X[te])
    c = GradientBoostingClassifier(**clf_params).fit(X[tr], y_cls[tr])
    oof_cls_pred[te] = c.predict(X[te])
    oof_cls_prob[te] = c.predict_proba(X[te])

# Regression metrics
def reg_metrics(pred):
    return dict(r2=float(r2_score(y_reg, pred)),
                rmse=round(float(np.sqrt(mean_squared_error(y_reg, pred))), 4),
                mae=round(float(mean_absolute_error(y_reg, pred)), 4))

base_mean = np.full_like(y_reg, y_reg.mean())
lin = LinearRegression().fit(X, y_reg)
lin_oof = np.zeros(len(y_reg))
for tr, te in kf.split(X):
    lin_oof[te] = LinearRegression().fit(X[tr], y_reg[tr]).predict(X[te])

reg_metrics_out = {
    'baseline_mean': reg_metrics(base_mean),
    'linear': reg_metrics(lin_oof),
    'model': reg_metrics(oof_reg),
}

# Classification metrics
CLASS_EDGES = [0.5, 1.5]
def band_from_perf(p):
    return int(np.digitize(p, CLASS_EDGES))

derived = np.array([band_from_perf(p) for p in oof_reg])
clf_metrics_out = {
    'baseline_majority': float(round(accuracy_score(y_cls, np.full_like(y_cls, int(np.bincount(y_cls).argmax()))), 4)),
    'model': float(round(accuracy_score(y_cls, oof_cls_pred), 4)),
    'balanced_accuracy': float(round(balanced_accuracy_score(y_cls, oof_cls_pred), 4)),
    'class_support': [int(v) for v in np.bincount(y_cls, minlength=3)],
    'confusion_matrix': confusion_matrix(y_cls, oof_cls_pred, labels=[0, 1, 2]).tolist(),
    'derived_from_regression': float(round(accuracy_score(y_cls, derived), 4)),
}

# Feature importance from the classifier
feat_imp = sorted([(f, float(v)) for f, v in zip(FEATURES, clf.feature_importances_)],
                  key=lambda t: -t[1])

# ---------------- Empirical prediction bands (residual percentiles) ----------------
edges = [0.0, 0.5, 1.5, 2.0]
buckets = []
for k in range(3):
    lo, hi = edges[k], edges[k + 1]
    m = (oof_reg >= lo) & (oof_reg < hi if k < 2 else oof_reg <= hi)
    if m.sum() >= 20:
        resid = y_reg[m] - oof_reg[m]
        buckets.append({'resid_lo10': round(float(np.percentile(resid, 10)), 4),
                        'resid_p90': round(float(np.percentile(resid, 90)), 4),
                        'n': int(m.sum())})
    else:
        buckets.append(None)

# ---------------- Serialise trees ----------------
def flat_tree(est):
    t = est.tree_
    out = []
    for i in range(t.node_count):
        feat = int(t.feature[i])
        thr = float(t.threshold[i])
        left = int(t.children_left[i])
        right = int(t.children_right[i])
        val = float(t.value[i].ravel()[0])
        if left == -1:
            out.extend([-2, -2, -1, -1, val])
        else:
            out.extend([feat, thr, left, right, 0.0])
    return out

reg_trees = [flat_tree(e) for e in reg.estimators_[:, 0]]
reg_init = float(reg.init_.mean) if hasattr(reg.init_, 'mean') else float(reg.init_.constant_.ravel()[0])

clf_trees = []
for stage in range(clf.estimators_.shape[0]):
    for k in range(3):
        clf_trees.append({'cls': k, 'tree': flat_tree(clf.estimators_[stage, k])})
clf_init = [float(v) for v in clf.init_.class_prior_]
clf_init_log = [float(np.log(max(p, 1e-12))) for p in clf_init]

# ---------------- Held-out sample records for "load a real student" ----------------
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(df), size=60, replace=False)
samples = []
for i in sample_idx:
    s = {f: float(X[i, j]) for j, f in enumerate(FEATURES)}
    s['observed_perf'] = float(y_reg[i])
    s['observed_class'] = CLASS_NAMES[int(y_cls[i])]
    samples.append(s)

model = {
    'meta': {
        'dataset': 'xAPI-Edu-Data (Kalboard 360)',
        'source_url': 'https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data',
        'citation': 'Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). Mining Educational Data to Predict Student\u2019s academic Performance using Ensemble Methods. International Journal of Database Theory and Application, 9(8), 119\u2013136.',
        'rows_used': int(len(df)),
        'raw_rows': int(len(df)),
        'features': FEATURES,
        'class_names': CLASS_NAMES,
        'class_edges': CLASS_EDGES,
        'seed': SEED,
        'cv_folds': cv_folds,
        'fold_test_sizes': fold_test_sizes,
        'fold_train_sizes': fold_train_sizes,
        'cv_note': '5-fold shuffled cross-validation, out-of-fold predictions',
        'reg_params': reg_params,
        'clf_params': clf_params,
        'input_ranges': input_ranges,
        'defaults': defaults,
    },
    'regressor': {'learning_rate': reg_params['learning_rate'], 'init': reg_init, 'trees': reg_trees},
    'classifier': {'learning_rate': clf_params['learning_rate'], 'init': clf_init_log, 'trees': clf_trees},
    'bands': {'edges': edges, 'resid': buckets},
    'metrics': {'regression': reg_metrics_out, 'classification': clf_metrics_out,
                'feature_importance': feat_imp},
    'samples': samples,
}

with open(f'{HERE}/model/model.json', 'w') as fp:
    json.dump(model, fp)

print('rows', len(df))
print('reg', reg_metrics_out['model'])
print('clf acc', clf_metrics_out['model'], 'balanced', clf_metrics_out['balanced_accuracy'],
      'derived', clf_metrics_out['derived_from_regression'])
print('feat_imp', feat_imp)
