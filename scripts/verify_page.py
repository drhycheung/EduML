"""Prove the in-page inference agrees with the trained scikit-learn models.

Extracts the pure-model core from index.html, runs it under Node.js on a grid
of synthetic inputs, and compares every output with an independent Python
re-implementation of the same linear / logistic arithmetic. It also fits the
same models fresh with scikit-learn and checks the exported weights reproduce
them, so the JSON cannot silently drift from the code that produced it.
"""
import json, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(HERE, 'index.html'), encoding='utf-8').read()
MODEL = json.load(open(os.path.join(HERE, 'model', 'model.json'), encoding='utf-8'))

# ---- Extract pure-model core + inline a test harness under Node ----
start = HTML.index('const MODEL = ')
end = HTML.index('PURE-MODEL CORE — END')
core = HTML[start:end]
core = core[:core.rindex('/* =====')]

rng = np.random.default_rng(0)
ranges = MODEL['meta']['input_ranges']
feats = MODEL['meta']['features']
inputs = []
for _ in range(200):
    row = []
    for f in feats:
        lo, hi = ranges[f]
        row.append(float(rng.uniform(lo, hi)))
    inputs.append(row)

node_src = core + """
const INPUTS = %s;
const out = INPUTS.map(x => ({ reg: regPredict(x), cls: clfScores(x) }));
console.log(JSON.stringify(out));
""" % json.dumps(inputs)

proc = subprocess.run(['node', '-e', node_src], capture_output=True, text=True)
if proc.returncode != 0:
    print(proc.stderr)
    sys.exit('node failed')
js = json.loads(proc.stdout)


def py_reg(x):
    m = MODEL['regressor']
    return m['intercept'] + sum(w * v for w, v in zip(m['weights'], x))


def py_clf(x):
    m = MODEL['classifier']
    z = [(v - mu) / s for v, mu, s in
         zip(x, m['scaler_mean'], m['scaler_scale'])]
    return [b + sum(w * zi for w, zi in zip(row, z))
            for row, b in zip(m['weights'], m['intercept'])]

max_reg = 0.0
max_cls = 0.0
for i, x in enumerate(inputs):
    max_reg = max(max_reg, abs(js[i]['reg'] - py_reg(x)))
    cs = js[i]['cls']
    ps = py_clf(x)
    max_cls = max(max_cls, max(abs(a - b) for a, b in zip(cs, ps)))

print('rows checked:', len(inputs))
print('max regression abs diff :', max_reg)
print('max classifier abs diff :', max_cls)
assert max_reg < 1e-9 and max_cls < 1e-9, 'JS and Python disagree'
print('OK: browser inference matches the exported JSON exactly')

# ---- Independently refit with scikit-learn and compare to the export ----
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(os.path.join(HERE, 'data', 'xAPI-Edu-Data.csv'))
df['absence_high'] = (df['StudentAbsenceDays'] == 'Above-7').astype(float)
df['semester_s'] = (df['Semester'] == 'S').astype(float)
CLASS = {'L': 0, 'M': 1, 'H': 2}
y_reg = df['Class'].map(CLASS).astype(float).values
y_cls = df['Class'].map(CLASS).astype(int).values
X = df[MODEL['meta']['features']].values.astype(float)

sk_reg = LinearRegression().fit(X, y_reg)
sk_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, y_cls)
sk_logit = sk_pipe.named_steps['logisticregression']

dm = MODEL['regressor']
cm = MODEL['classifier']
d_reg_w = max(abs(a - b) for a, b in zip(dm['weights'], sk_reg.coef_))
d_reg_b = abs(dm['intercept'] - sk_reg.intercept_)
d_clf_w = max(abs(a - b) for ra, rb in zip(cm['weights'], sk_logit.coef_)
              for a, b in zip(ra, rb))
d_clf_b = max(abs(a - b) for a, b in zip(cm['intercept'], sk_logit.intercept_))
print('max |exported - sklearn| :', max(d_reg_w, d_reg_b, d_clf_w, d_clf_b))
assert max(d_reg_w, d_reg_b, d_clf_w, d_clf_b) < 1e-9, 'export drifted from sklearn'
print('OK: exported weights reproduce a fresh scikit-learn fit')
