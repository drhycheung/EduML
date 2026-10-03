"""Prove the in-page traversal agrees with scikit-learn.

Extracts the pure-model core from index.html, runs it under Node.js on a grid
of synthetic inputs, and compares every output with an independent Python
re-implementation of the same tree traversal.
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


def flat_traverse(tree, x):
    o = 0
    while tree[o] >= 0:
        o = 5 * (tree[o + 2] if x[int(tree[o])] <= tree[o + 1] else tree[o + 3])
    return tree[o + 4]


def py_reg(x):
    m = MODEL['regressor']
    out = m['init'] + m['learning_rate'] * sum(flat_traverse(t, x) for t in m['trees'])
    return out


def py_clf(x):
    m = MODEL['classifier']
    score = list(m['init'])
    for t in m['trees']:
        score[t['cls']] += m['learning_rate'] * flat_traverse(t['tree'], x)
    return score

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
print('OK: browser traversal matches Python exactly')
