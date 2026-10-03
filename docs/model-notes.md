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

**Logistic regression.** The classification counterpart of a straight line.
Instead of one number, it fits one weighted sum per class, then turns those sums
into a probability distribution over classes (the *softmax*). It is fitted on
*standardised* inputs — each feature rescaled to mean 0, standard deviation 1 —
because its optimiser stalls when features are measured in wildly different
units (25 extra resource visits and a 0/1 absence flag do not share a scale).
Standardising also makes the coefficients directly comparable across features,
which is what the feature-importance table on the page reports.

**Decision tree.** Repeatedly split the data on the single input that best
separates the outcome, forming a pyramid of if/else rules. A single deep tree
fits the training data well and generalises poorly.

**Gradient boosting.** Build many shallow trees, each fitted to the *errors* of
the trees so far, and add them together with a small learning rate. This usually
beats a single tree — but it is not guaranteed to beat a simple linear model,
especially on small data (§5). On this dataset it does not, which is why the page
deploys the simpler linear and logistic models and shows boosting only as a
comparison.

**Cross-validation.** Split the data into *k* folds. For each fold, fit on the
other *k−1* folds and score the held-out fold. Every record is scored exactly
once, by a model that never saw it. The figures on the page are these
out-of-fold scores. This is why the page can claim the numbers are honest: no
student is scored by a model trained on that student.

**Stratified, repeated cross-validation.** With three classes, an ordinary
shuffled split can by chance put very few of a small class in a test fold, so we
use *stratified* folds that preserve the class proportions. And because a single
split is still a small sample, the whole procedure is *repeated* over five
different seeds; every reported figure is the mean across repeats with a
**± std**, so a lucky split is visible rather than disguised. One concrete reason
this matters: across individual splits, the difference between the logistic
classifier and simply thresholding the regression output swings by a couple of
points in both directions, but averaged over repeats the gap is 0.1 points —
inside the noise (§6). Reporting a single split would have told whichever story
that split happened to favour.

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

There are no trees on this page. Both deployed models are exported as plain
numbers, so the whole forward pass is a handful of additions and multiplications.

**Regression (ordinary least squares).** The model file stores six weights and one
intercept. The predicted level is just a dot product:

```
level = intercept + Σ (weight[j] · x[j])
```

Then it is clamped to the 0–2 range for display.

**Classification (multinomial logistic regression).** The coefficients were fitted
on standardised features, so the first step is to z-score the input with the
scaler that travels in the model file:

```
z[j] = (x[j] − scaler_mean[j]) / scaler_scale[j]
score[k] = intercept[k] + Σ (weight[k][j] · z[j])      for each class k
prediction = argmax(score)
```

The page only needs the argmax, so it does not compute the full softmax; the
stored coefficients are the softmax's linear scores. `scripts/verify_page.py`
runs this exact code under Node.js, checks it against an independent Python
re-implementation, and separately refits both models with scikit-learn to confirm
the exported numbers reproduce the fit. So "the page agrees with the training
code" is a tested claim, not an assertion.

## 4. The prediction interval

The `0.8 – 2.1`-style range next to the headline is **not** a textbook confidence
interval. It is the empirical 10th–90th percentile of out-of-fold residuals,
bucketed by predicted level: *"for students the model placed around here, where
did the truth actually land 80% of the time?"* That is a more honest statement
than a formula that assumes a distribution the data may not follow.

## 5. Why boosting loses to linear regression here

On the regression task the boosted trees score **R² = 0.600 ± 0.012**, below
plain multiple linear regression at **0.637 ± 0.002**. This is not a bug, and not
a lucky split: the two ranges do not overlap across repeats. With only 480
students and six fairly additive inputs, the boosted model has ample capacity to
fit sampling noise that does not reappear in the held-out fold. The linear model
cannot do that, so it generalises slightly better.

The lesson is not "boosting is bad". It is that **model complexity must earn its
keep**, and the only way to know is to put the simple baseline in the table
first. Many published learning-analytics results would look different if their
authors had done this.

## 6. Regression vs classification

Thresholding the regression output at 0.5 and 1.5 gives 71.9% ± 0.7% accuracy;
the multinomial logistic classifier gives 71.8% ± 0.3%. The two are
**statistically indistinguishable** on this dataset: the 0.1-point gap is well
inside the spread across repeated splits — and across individual splits the sign
of the gap flips. The honest conclusion is not "the classifier is better" but "on
480 rows you cannot tell". "Predict a number, then threshold it" and "classify"
are genuinely different tasks with different loss functions; whether the extra
classifier is worth it is a design decision that this data cannot settle. The
page deploys the classifier because it returns probabilities and is better
calibrated for the marker on the scale, not because it is measurably more
accurate. Both gradient-boosted variants trail both of these by two to three
points; that part *is* outside the noise.

## 7. What was deliberately excluded

The dataset also contains ten demographic and context attributes (gender,
nationality, place of birth, stage, grade, section, topic, parental relation, and
two parent-survey answers). Adding them raises accuracy from 0.718 ± 0.003 to
0.731 ± 0.013 — a modest but real 1.3-point gain — and on their own they reach
0.547 ± 0.010, ten points above the 0.440 majority baseline. They are still
excluded. A model keyed on
who a student *is* rather than what they *do* cannot be acted on by a tutor, and
deployed for triage it would encode historical group inequity in a decision about
a named student. A higher score bought that way is worse than a lower honest one.

## 8. What the model does *not* know

Prior attainment, motivation, health, home circumstances and teaching quality
are absent. Engagement indicators are correlates, not causes: a student who opens
more resources is not *made* more able by the clicking. Roughly 36% of the
variance in level is unexplained even by the best model here, and that remainder
is the part that a human tutor still has to handle.
