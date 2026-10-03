# Model notes (EduML)

This page does two things at once, and it is worth separating them:

- a **regression** task — predict a student's performance *level* on a 0–2 scale;
- a **classification** task — predict a three-band class, Low / Medium / High.

They use the same six inputs but answer slightly different questions. The
comparison between them is itself a teaching point (§6).

## 1. The algorithms in plain language

**Baseline (predict the mean).** The simplest possible "model": ignore the
inputs and always output the average level. Its R² is 0 by construction, because
a constant cannot explain any variance. Every other model must beat it to be
worth anything.

**Multiple linear regression.** Fit one straight-line relationship of the form
`level ≈ a + b·resources + c·hands + …`, choosing the coefficients that minimise
squared error. It is transparent, hard to overfit on six inputs, and on this
dataset surprisingly strong.

**Decision tree.** Repeatedly split the data on the single input that best
separates the outcome, forming a pyramid of if/else rules. A single deep tree
fits the training data well and generalises poorly.

**Gradient boosting.** Build many shallow trees, each fitted to the *errors* of
the trees so far, and add them together with a small learning rate. This usually
beats a single tree — but it is not guaranteed to beat a simple linear model,
especially on small data (§5).

**Cross-validation.** Split the data into *k* folds. For each fold, fit on the
other *k−1* folds and score the held-out fold. Every record is scored exactly
once, by a model that never saw it. The figures on the page are these
out-of-fold scores. This is why the page can claim the numbers are honest: no
student is scored by a model trained on that student.

## 2. Features

| Feature | Meaning |
| --- | --- |
| `raisedhands` | times the student raised a hand |
| `VisITedResources` | course resources opened in the LMS |
| `AnnouncementsView` | announcements viewed |
| `Discussion` | messages posted in discussions |
| `absence_high` | 1 if the student had 7 or more absence days |
| `semester_s` | 1 if the course ran in the second semester |

## 3. How a prediction is computed in the browser

Each tree is stored as a flat array, five numbers per node:
`[feature, threshold, left, right, value]`, with `left`/`right` child pointers
stored as node indices. A leaf is marked by a negative `feature`. Traversal
follows scikit-learn's rule: `feature <= threshold` goes **left**.

```
o = 0
while array[o] >= 0:
    o = 5 * (array[o+2] if x[array[o]] <= array[o+1] else array[o+3])
leaf = array[o+4]
```

The regression output is `init + learning_rate · Σ leaf values`. The classifier
keeps three such sums (one per class) and takes the largest. `scripts/verify_page.py`
runs this exact code under Node.js and checks it against an independent Python
implementation, so "the page agrees with the training code" is a tested claim,
not an assertion.

## 4. The prediction interval

The `0.8 – 2.1`-style range next to the headline is **not** a textbook confidence
interval. It is the empirical 10th–90th percentile of out-of-fold residuals,
bucketed by predicted level: *"for students the model placed around here, where
did the truth actually land 80% of the time?"* That is a more honest statement
than a formula that assumes a distribution the data may not follow.

## 5. Why boosting loses to linear regression here

On the regression task the boosted trees score **R² = 0.594**, below plain
multiple linear regression at **0.637**. This is not a bug. With only 480
students and six fairly additive inputs, the boosted model has ample capacity to
fit sampling noise that does not reappear in the held-out fold. The linear model
cannot do that, so it generalises slightly better.

The lesson is not "boosting is bad". It is that **model complexity must earn its
keep**, and the only way to know is to put the simple baseline in the table
first. Many published learning-analytics results would look different if their
authors had done this.

## 6. Regression vs classification

Thresholding the regression output at 0.5 and 1.5 gives 69.6% accuracy; the
dedicated classifier gives 71.5%. The gap comes from borderline students whose
regression output sits near a boundary. "Predict a number, then threshold it" and
"classify" are different tasks with different loss functions — choosing between
them is a design decision, not a contest.

## 7. What the model does *not* know

Prior attainment, motivation, health, home circumstances and teaching quality
are absent. Engagement indicators are correlates, not causes: a student who opens
more resources is not *made* more able by the clicking. Roughly 41% of the
variance in level is unexplained even by the best model here, and that remainder
is the part that a human tutor still has to handle.
