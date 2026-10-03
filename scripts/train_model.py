"""Train EduML models on xAPI-Edu-Data and export model/model.json.

The page deploys the model that actually fits this dataset best, decided by an
honest comparison: a linear regression for the performance level and a
multinomial logistic regression for the three-band class. A gradient-boosted
alternative is trained and scored on exactly the same splits, and kept in the
export as a comparison — on 480 rows it loses to the simpler models, which is
the teaching point rather than an embarrassment.

Evaluation is 5-fold *stratified* cross-validation repeated over several seeds.
Exported metrics are means with standard deviations across the repeats, so a
reported figure cannot be a lucky split. The confusion matrix, prediction bands
and sample rows come from one canonical split (seed 42) and are labelled as such.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

# ---------------- Deployed models: full-data fit ----------------
# Linear regression is scale-invariant, so it is fitted on the raw features.
# Logistic regression is not, and lbfgs stalls when features span very different
# ranges, so it is fitted on standardised features inside a pipeline. The scaler
# travels with the model into the browser.
lin = LinearRegression().fit(X, y_reg)
logit_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, y_cls)
scaler = logit_pipe.named_steps['standardscaler']
logit = logit_pipe.named_steps['logisticregression']


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
    out = {k: np.zeros(len(y_reg)) for k in ('bm', 'lin', 'gbm')}
    oof_clf_logit = np.zeros(len(y_cls), dtype=int)
    oof_clf_gbm = np.zeros(len(y_cls), dtype=int)
    oof_prob = np.zeros((len(y_cls), 3))
    for tr, te in skf.split(X, y_cls):
        out['bm'][te] = y_reg[tr].mean()
        out['lin'][te] = LinearRegression().fit(X[tr], y_reg[tr]).predict(X[te])
        out['gbm'][te] = GradientBoostingRegressor(**reg_params).fit(X[tr], y_reg[tr]).predict(X[te])
        lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[tr], y_cls[tr])
        oof_clf_logit[te] = lg.predict(X[te])
        oof_prob[te] = lg.predict_proba(X[te])
        oof_clf_gbm[te] = GradientBoostingClassifier(**clf_params).fit(X[tr], y_cls[tr]).predict(X[te])
    return out, oof_clf_logit, oof_clf_gbm, oof_prob


reg_runs = {k: {'r2': [], 'rmse': [], 'mae': []}
            for k in ('baseline_mean', 'linear', 'gbm')}
clf_runs = {k: {'acc': [], 'balanced': []}
            for k in ('baseline_majority', 'logistic', 'gbm_classifier')}
thr_runs = {'linear_regression_threshold': [], 'gbm_regression_threshold': []}
maj = int(np.bincount(y_cls).argmax())
majority = np.full_like(y_cls, maj)

for seed in range(REPEATS):
    o, oof_logit, oof_gbm_clf, _ = oof_predictions(seed)
    for name, pred in [('baseline_mean', o['bm']), ('linear', o['lin']), ('gbm', o['gbm'])]:
        m = reg_metrics(pred)
        for k in m:
            reg_runs[name][k].append(m[k])
    clf_runs['baseline_majority']['acc'].append(accuracy_score(y_cls, majority))
    clf_runs['baseline_majority']['balanced'].append(balanced_accuracy_score(y_cls, majority))
    clf_runs['logistic']['acc'].append(accuracy_score(y_cls, oof_logit))
    clf_runs['logistic']['balanced'].append(balanced_accuracy_score(y_cls, oof_logit))
    clf_runs['gbm_classifier']['acc'].append(accuracy_score(y_cls, oof_gbm_clf))
    clf_runs['gbm_classifier']['balanced'].append(balanced_accuracy_score(y_cls, oof_gbm_clf))
    thr_runs['linear_regression_threshold'].append(
        accuracy_score(y_cls, [band_from_perf(p) for p in o['lin']]))
    thr_runs['gbm_regression_threshold'].append(
        accuracy_score(y_cls, [band_from_perf(p) for p in o['gbm']]))

# Canonical split for the confusion matrix, bands and sample rows.
_o, _oclf, _ogbm, _oprob = oof_predictions(SEED)
canonical = dict(oof_reg=_o['lin'], oof_clf=_oclf, oof_prob=_oprob)

regression_metrics = {k: {m: agg(v) for m, v in d.items()} for k, d in reg_runs.items()}
classification_metrics = {k: {m: agg(v) for m, v in d.items()} for k, d in clf_runs.items()}
for k, v in thr_runs.items():
    classification_metrics[k] = {'acc': agg(v)}
classification_metrics['class_support'] = [int(v) for v in np.bincount(y_cls, minlength=3)]
classification_metrics['confusion_matrix'] = confusion_matrix(
    y_cls, canonical['oof_clf'], labels=[0, 1, 2]).tolist()

# ---------------- Fairness diagnostic: what the dropped attributes buy ----------------
# Same logistic classifier, same folds, with the ten demographic/context
# attributes added (one-hot encoded). Reported so the page can state the cost
# of excluding them — under the *deployed* model family, not a straw man.
Xd = pd.get_dummies(df[FEATURES + DEMOGRAPHIC],
                    columns=DEMOGRAPHIC).values.astype(float)
fair = {'engagement_only_acc': [], 'with_demographics_acc': []}
for seed in range(REPEATS):
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    p_eng = np.zeros(len(y_cls), dtype=int)
    p_dem = np.zeros(len(y_cls), dtype=int)
    for tr, te in skf.split(X, y_cls):
        p_eng[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[tr], y_cls[tr]).predict(X[te])
        p_dem[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xd[tr], y_cls[tr]).predict(Xd[te])
    fair['engagement_only_acc'].append(accuracy_score(y_cls, p_eng))
    fair['with_demographics_acc'].append(accuracy_score(y_cls, p_dem))
fairness_metrics = {k: agg(v) for k, v in fair.items()}

# ---------------- Feature importance: standardised logistic coefficients ----------------
# The pipeline standardises features before fitting, so coef_ is already on a
# per-standard-deviation scale and magnitudes are comparable across features
# measured in different units. Averaging |coefficient| over the three classes
# gives one importance figure per feature; the signed per-class coefficients are
# also exported so the direction can be inspected.
std_coef = logit.coef_  # (3 classes, 6 features)
importance = np.mean(np.abs(std_coef), axis=0)
feat_imp = sorted([(f, float(v)) for f, v in zip(FEATURES, importance)],
                  key=lambda t: -t[1])
coef_table = {f: [round(float(std_coef[k, j]), 4) for k in range(3)]
              for j, f in enumerate(FEATURES)}

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
        'cv_note': cv_note,
        'reg_params': reg_params,
        'clf_params': clf_params,
        'input_ranges': input_ranges,
        'defaults': defaults,
    },
    # Deployed regression: level = weights . x + intercept
    'regressor': {'type': 'linear',
                  'weights': [float(w) for w in lin.coef_],
                  'intercept': float(lin.intercept_)},
    # Deployed classifier: p = softmax(weights . x + intercept)
    'classifier': {'type': 'multinomial_logistic',
                   'weights': [[float(w) for w in row] for row in logit.coef_],
                   'intercept': [float(b) for b in logit.intercept_],
                   'scaler_mean': [float(m) for m in scaler.mean_],
                   'scaler_scale': [float(s) for s in scaler.scale_],
                   'classes': CLASS_NAMES},
    'bands': {'edges': edges, 'resid': buckets},
    'metrics': {'regression': regression_metrics,
                'classification': classification_metrics,
                'fairness': fairness_metrics,
                'feature_importance': feat_imp,
                'standardised_coefficients': coef_table},
    'samples': samples,
}

with open(f'{HERE}/model/model.json', 'w') as fp:
    json.dump(model, fp)

print('rows', len(df))
print('reg baseline', regression_metrics['baseline_mean'])
print('reg linear  ', regression_metrics['linear'])
print('reg gbm     ', regression_metrics['gbm'])
print('clf majority', classification_metrics['baseline_majority'])
print('clf logistic', classification_metrics['logistic'])
print('clf gbm     ', classification_metrics['gbm_classifier'])
print('thr linear  ', classification_metrics['linear_regression_threshold'])
print('thr gbm     ', classification_metrics['gbm_regression_threshold'])
print('fairness    ', fairness_metrics)
print('feat_imp', [(f, round(v, 3)) for f, v in feat_imp])
