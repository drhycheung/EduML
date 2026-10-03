"""Train EduML models on xAPI-Edu-Data and export model/model.json.

A regression target (performance level 0-2) and a 3-class classifier, both
gradient-boosted trees, serialised to flat arrays for browser traversal.

Evaluation is 5-fold *stratified* cross-validation repeated over several seeds.
Exported metrics are means with standard deviations across the repeats, so the
reported figure cannot be a lucky split. The confusion matrix, prediction bands
and the sample rows are taken from one canonical split (seed 42) and labelled
as such.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

SEED = 42
REPEATS = 5
FOLDS = 5
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(f'{HERE}/data/xAPI-Edu-Data.csv')

# Target: performance level  L->0, M->1, H->2
LEVEL = {'L': 0, 'M': 1, 'H': 2}
CLASS_NAMES = ['Low', 'Medium', 'High']
CLASS_EDGES = [0.5, 1.5]
df['perf'] = df['Class'].map(LEVEL).astype(float)
df['absence_high'] = (df['StudentAbsenceDays'] == 'Above-7').astype(float)
df['semester_s'] = (df['Semester'] == 'S').astype(float)

FEATURES = ['raisedhands', 'VisITedResources', 'AnnouncementsView',
            'Discussion', 'absence_high', 'semester_s']
DEMOGRAPHIC = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
               'SectionID', 'Topic', 'Relation', 'ParentAnsweringSurvey',
               'ParentschoolSatisfaction']

X = df[FEATURES].values.astype(float)
y_reg = df['perf'].values.astype(float)
y_cls = df['perf'].values.astype(int)

input_ranges = {f: [float(X[:, i].min()), float(X[:, i].max())]
                for i, f in enumerate(FEATURES)}
defaults = {f: float(np.median(X[:, i])) for i, f in enumerate(FEATURES)}
defaults['absence_high'] = 0.0
defaults['semester_s'] = 0.0

reg_params = dict(n_estimators=250, max_depth=3, learning_rate=0.06, random_state=SEED)
clf_params = dict(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=SEED)

# Full-data fit: the model the page deploys, and feature importance.
reg = GradientBoostingRegressor(**reg_params).fit(X, y_reg)
clf = GradientBoostingClassifier(**clf_params).fit(X, y_cls)


def reg_metrics(pred):
    return dict(r2=float(r2_score(y_reg, pred)),
                rmse=float(np.sqrt(mean_squared_error(y_reg, pred))),
                mae=float(mean_absolute_error(y_reg, pred)))


def band_from_perf(p):
    return int(np.digitize(p, CLASS_EDGES))


def agg(vals):
    return {'mean': round(float(np.mean(vals)), 4), 'std': round(float(np.std(vals)), 4)}


# ---------------- Repeated stratified cross-validation ----------------
def oof_predictions(seed):
    """One full out-of-fold pass over all rows for a given split seed."""
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    oof_reg = np.zeros(len(y_reg))
    oof_lin = np.zeros(len(y_reg))
    oof_bm = np.zeros(len(y_reg))
    oof_clf = np.zeros(len(y_cls), dtype=int)
    oof_prob = np.zeros((len(y_cls), 3))
    for tr, te in skf.split(X, y_cls):
        oof_reg[te] = GradientBoostingRegressor(**reg_params).fit(X[tr], y_reg[tr]).predict(X[te])
        oof_lin[te] = LinearRegression().fit(X[tr], y_reg[tr]).predict(X[te])
        oof_bm[te] = y_reg[tr].mean()
        c = GradientBoostingClassifier(**clf_params).fit(X[tr], y_cls[tr])
        oof_clf[te] = c.predict(X[te])
        oof_prob[te] = c.predict_proba(X[te])
    return oof_reg, oof_lin, oof_bm, oof_clf, oof_prob


reg_runs = {k: {'r2': [], 'rmse': [], 'mae': []}
            for k in ('baseline_mean', 'linear', 'model')}
clf_runs = {k: {'acc': [], 'balanced': []}
            for k in ('baseline_majority', 'model')}
derived_runs = []
maj = int(np.bincount(y_cls).argmax())
majority = np.full_like(y_cls, maj)

for seed in range(REPEATS):
    oof_reg, oof_lin, oof_bm, oof_clf, _ = oof_predictions(seed)
    for name, pred in [('baseline_mean', oof_bm), ('linear', oof_lin), ('model', oof_reg)]:
        m = reg_metrics(pred)
        for k in m:
            reg_runs[name][k].append(m[k])
    clf_runs['baseline_majority']['acc'].append(accuracy_score(y_cls, majority))
    clf_runs['baseline_majority']['balanced'].append(balanced_accuracy_score(y_cls, majority))
    clf_runs['model']['acc'].append(accuracy_score(y_cls, oof_clf))
    clf_runs['model']['balanced'].append(balanced_accuracy_score(y_cls, oof_clf))
    derived_runs.append(accuracy_score(y_cls, [band_from_perf(p) for p in oof_reg]))

# Canonical split for the confusion matrix, bands and sample rows.
_cr, _cl, _cb, _cc, _cp = oof_predictions(SEED)
canonical = dict(oof_reg=_cr, oof_lin=_cl, oof_clf=_cc, oof_prob=_cp)

regression_metrics = {k: {m: agg(v) for m, v in d.items()} for k, d in reg_runs.items()}
classification_metrics = {k: {m: agg(v) for m, v in d.items()} for k, d in clf_runs.items()}
classification_metrics['derived_from_regression'] = {'acc': agg(derived_runs)}
classification_metrics['class_support'] = [int(v) for v in np.bincount(y_cls, minlength=3)]
classification_metrics['confusion_matrix'] = confusion_matrix(
    y_cls, canonical['oof_clf'], labels=[0, 1, 2]).tolist()

# ---------------- Fairness diagnostic: what the dropped attributes buy ----------------
# Same classifier, same folds, with the ten demographic/context attributes added
# (one-hot encoded). Reported so the page can state the cost of excluding them.
Xd = pd.get_dummies(df[FEATURES + DEMOGRAPHIC],
                    columns=DEMOGRAPHIC).values.astype(float)
fair = {'engagement_only_acc': [], 'with_demographics_acc': []}
for seed in range(REPEATS):
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    p_eng = np.zeros(len(y_cls), dtype=int)
    p_dem = np.zeros(len(y_cls), dtype=int)
    for tr, te in skf.split(X, y_cls):
        p_eng[te] = GradientBoostingClassifier(**clf_params).fit(X[tr], y_cls[tr]).predict(X[te])
        p_dem[te] = GradientBoostingClassifier(**clf_params).fit(Xd[tr], y_cls[tr]).predict(Xd[te])
    fair['engagement_only_acc'].append(accuracy_score(y_cls, p_eng))
    fair['with_demographics_acc'].append(accuracy_score(y_cls, p_dem))
fairness_metrics = {k: agg(v) for k, v in fair.items()}

# ---------------- Feature importance (from the full-fit classifier) ----------------
feat_imp = sorted([(f, float(v)) for f, v in zip(FEATURES, clf.feature_importances_)],
                  key=lambda t: -t[1])

# ---------------- Empirical prediction bands (canonical OOF residuals) ----------------
edges = [0.0, 0.5, 1.5, 2.0]
buckets = []
for k in range(3):
    lo, hi = edges[k], edges[k + 1]
    m = (canonical['oof_reg'] >= lo) & (canonical['oof_reg'] < hi if k < 2 else canonical['oof_reg'] <= hi)
    if m.sum() >= 20:
        resid = y_reg[m] - canonical['oof_reg'][m]
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

# ---------------- Sample rows for "load a real student" ----------------
# Each sample carries its OUT-OF-FOLD prediction, so the page never shows a
# prediction the deployed model was trained on.
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(df), size=60, replace=False)
samples = []
for i in sample_idx:
    s = {f: float(X[i, j]) for j, f in enumerate(FEATURES)}
    s['observed_perf'] = float(y_reg[i])
    s['observed_class'] = CLASS_NAMES[int(y_cls[i])]
    s['oof_perf'] = round(float(canonical['oof_reg'][i]), 4)
    s['oof_class'] = CLASS_NAMES[band_from_perf(canonical['oof_reg'][i])]
    samples.append(s)

cv_note = (f'{FOLDS}-fold stratified cross-validation, repeated over {REPEATS} seeds; '
           f'figures are mean \u00b1 std across repeats')

model = {
    'meta': {
        'dataset': 'xAPI-Edu-Data (Kalboard 360)',
        'source_url': 'https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data',
        'citation': 'Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). Mining Educational Data to Predict Student\u2019s academic Performance using Ensemble Methods. International Journal of Database Theory and Application, 9(8), 119\u2013136.',
        'rows_used': int(len(df)),
        'raw_rows': int(len(df)),
        'features': FEATURES,
        'dropped_features': DEMOGRAPHIC,
        'class_names': CLASS_NAMES,
        'class_edges': CLASS_EDGES,
        'seed': SEED,
        'cv_folds': FOLDS,
        'cv_repeats': REPEATS,
        'fold_test_sizes': [int(round(len(y_cls) / FOLDS))] * FOLDS,
        'fold_train_sizes': [int(len(y_cls) - round(len(y_cls) / FOLDS))] * FOLDS,
        'cv_note': cv_note,
        'reg_params': reg_params,
        'clf_params': clf_params,
        'input_ranges': input_ranges,
        'defaults': defaults,
    },
    'regressor': {'learning_rate': reg_params['learning_rate'], 'init': reg_init, 'trees': reg_trees},
    'classifier': {'learning_rate': clf_params['learning_rate'], 'init': clf_init_log, 'trees': clf_trees},
    'bands': {'edges': edges, 'resid': buckets},
    'metrics': {'regression': regression_metrics,
                'classification': classification_metrics,
                'fairness': fairness_metrics,
                'feature_importance': feat_imp},
    'samples': samples,
}

with open(f'{HERE}/model/model.json', 'w') as fp:
    json.dump(model, fp)

print('rows', len(df))
print('reg baseline', regression_metrics['baseline_mean'])
print('reg linear  ', regression_metrics['linear'])
print('reg model   ', regression_metrics['model'])
print('clf majority', classification_metrics['baseline_majority'])
print('clf model   ', classification_metrics['model'])
print('clf derived ', classification_metrics['derived_from_regression'])
print('fairness    ', fairness_metrics)
print('feat_imp', [(f, round(v, 3)) for f, v in feat_imp])
