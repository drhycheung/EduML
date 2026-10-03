# Reproducing the demo with vibe coding

Companion guide to the [main README](../README.md). This is the teaching pack for the
lesson: why the page is designed the way it is, how it was actually built with an AI
coding tool, and the complete prompt to hand to one.

**Last verified: 3 October 2026** against OpenCode, Node.js v26.7.0, scikit-learn 1.5.1,
pandas 2.2.2, and the deployed page at <https://drhycheung.github.io/EduML/>.

## Contents

1. [Design thinking: from a dashboard that only looks back to a prediction you can act on](#1-design-thinking-from-a-dashboard-that-only-looks-back-to-a-prediction-you-can-act-on)
2. [How the page was built](#2-how-the-page-was-built)
3. [Further work for students](#3-further-work-for-students)
4. [The reproduction prompt](#4-the-reproduction-prompt) ← jump here if you just want to build it

---

## 1. Design thinking: from a dashboard that only looks back to a prediction you can act on

The page is the output of one design-thinking loop applied to a real teaching problem:
learning platforms already record a great deal of behavioural data, and learning-analytics
dashboards for displaying it are everywhere. This is precisely why the data alone changes
nothing.

| Stage | This project's arc |
|---|---|
| **1. Empathise** | The user pain: a tutor or programme leader opens an engagement dashboard and sees attendance, clicks and submission counts rendered clearly — all of it about the past. Every question they actually have is forward-looking: *which students are heading for trouble, who should I reach out to now, while there is still time to help?* A chart cannot answer any of them. The data are abundant, but the decision remains unavailable. |
| **2. Define** | Problem statement: *we have the learning data but cannot predict outcomes from it, so the data does not lead to an action.* Design goal: the same engagement signals must produce a forward-looking estimate the tutor can act on — and honestly enough that they trust it enough to act. |
| **3. Ideate** | Options considered: (a) yet another engagement dashboard — rejected, because that is what already exists and it looks backwards; (b) a single pass/fail flags with no supporting information — rejected, cannot be acted upon; (c) a predicted performance level, an uncertainty range, and an actionable band (Low / Medium / High) on the scale that *is* the decision; (d) baseline comparisons, so the result can be judged; (e) a panel stating which attributes were excluded, and why. Chosen: (c), (d) and (e), with the performance band as the actionable output. |
| **4. Prototype** | The single-file page. The prediction occupies the main position; a marker on the 0–2 scale shows where the estimate lands against the 0.5 and 1.5 band boundaries that *are* the action triggers; the interval is explained in words ("Among historical students the model placed around here, the truth landed in this range 80% of the time"); regression and classification baselines are shown side by side; and a highlighted panel states that ten demographic attributes were excluded deliberately *even though including them scored higher*. |
| **5. Test** | Parity tests between two independent implementations (JavaScript and Python), an independent refit with scikit-learn, a stylesheet-coverage check that every class used is actually implemented, and three faults that measurement caught but visual inspection did not — see section 2. |

The measurable outcome was **decision usefulness**: a tutor must be able to set a student's
engagement profile, obtain an estimate, see which band it falls in, and know how far to trust
it. Trustworthiness is a secondary requirement, required for the same reason — an estimate the
tutor does not trust will not be acted upon.

### Context: From description to prediction

Learning analytics can be understood through three activities: **monitor**, **analyse** and
**predict**. This project collects the engagement signals a dashboard would display (monitor),
fits a statistical model that relates those signals to performance (analyse), and turns them
into a forward-looking estimate with the uncertainty attached (predict). It is a demonstration
of method, not a deployed early-warning system.

---

## 2. How the page was built

The page was built with OpenCode, driven through Playwright, using a method in which every
claim is measured before it is accepted: write the smallest useful version, measure it, and
accept a claim only after something independent has confirmed it. The prompt in
[Part 4](#4-the-reproduction-prompt) encodes the findings below, so that a working page
should be produced on the first attempt.

1. **Profile the data before modelling anything.** The file has no missing values and two
   duplicate rows; the classes are imbalanced (Medium is 44% on its own), so the evaluation
   had to be stratified with a balanced-accuracy readout. That single profiling pass
   determined the whole evaluation design.
2. **Establish baselines first.** A predict-the-mean regressor and a predict-the-majority
   classifier were computed before any real model, so every later number had something to be
   compared against. The majority baseline is 44.0%; without it, "71.8%" means nothing.
3. **Measure the cost of the fairness choice instead of asserting it.** The tempting claim
   is "we dropped the demographic attributes for fairness." Instead, the *same* logistic
   classifier was refit with all ten included, under the same folds, so the page reports a
   measured gain (0.718 → 0.731) and a measured standalone score (0.547) rather than a value
   recorded once. A higher score bought with group membership is worse than a lower honest
   one.
4. **Let the held-out score choose the model, not the reference project.** The demo first
   mirrored a companion project and deployed gradient-boosted trees. An honest audit showed
   plain linear regression (R² 0.637 vs 0.600) and logistic regression (accuracy 71.8% vs
   69.3%) beat boosting on **both** tasks, so the simpler models are the ones deployed and
   boosting is kept only as a labelled comparison. See the bugs below.
5. **Serialise the model, then re-derive it.** Both models export to flat numbers: a linear
   model as one weight per feature plus an intercept, and the classifier as per-class
   weights, a stored z-score scaler, and intercepts. The in-page prediction is recomputed
   from those numbers by two independent implementations.
6. **Confirm that the browser and Python agree.** The pure-model region of the page is
   extracted and executed under Node.js, then diffed against an independently written Python
   implementation across 200 synthetic inputs. Current agreement: features to 4.4 × 10⁻¹⁶,
   classifier scores to 8.9 × 10⁻¹⁶ — effectively exact — and the exported weights reproduce
   a fresh scikit-learn fit to the last bit.
7. **Confirm that every class the markup uses is implemented in the vendored stylesheet**,
   because the stylesheet is a fixed snapshot (see the third bug below).

### Bugs that measurement caught and looking did not

All three produced a page that *looked finished*. None produced a JavaScript error.

- **The model was the wrong one, and nothing on the page said so.** Because the demo was
  built to mirror a companion project, it deployed gradient-boosted trees. The page rendered
  perfectly and every number on it was correct — but the model had been chosen by imitation,
  not by evidence. When both models were evaluated under identical repeated stratified
  cross-validation, the boosted trees lost on **both** tasks (regression R² 0.600 ± 0.012 vs
  0.637 ± 0.002; classification 69.3% ± 1.0% vs 71.8% ± 0.3%), with non-overlapping ranges.
  With 480 rows and six near-additive inputs, boosting had enough capacity to fit sampling
  noise that did not reappear in the held-out fold. Fixed by deploying the simpler model and
  reporting boosting as an honest loss. The lesson: a visually finished page can still be
  built on the wrong decision.

- **The classifier was fitted on unscaled features and had not converged.** Logistic
  regression stalls when its inputs span wildly different units (25 resource visits next to a
  0/1 absence flag), and scikit-learn raised a `ConvergenceWarning`. The page still showed
  numbers, so the fault was invisible — but the "converged-looking" accuracy of the
  non-converged fit was **0.7229**, and once the features were standardised and the optimiser
  converged properly it fell to **0.7179**. The unscaled number was inflated by an
  incomplete fit. Fixed by wrapping the classifier in a `StandardScaler` pipeline
  (`max_iter=5000`); the scaler now travels into the browser with the model. A silent
  warning was the only evidence that the reported accuracy was too high.

- **A vendored stylesheet has no way to tell you a class is missing.** The page inlines a
  snapshot of Tailwind CSS and recolours a few utilities to give the demo its violet
  identity. A bar chart was given the class `bg-violet-500`, which is *not* in the snapshot
  and *not* one of the recoloured utilities, so every bar rendered **transparent** — the
  feature-importance panel looked empty, and there was no error anywhere. The same fault
  appeared a second time: the "Medium" scale label used `left-[47%]`, an arbitrary value the
  snapshot had never generated, so the label sat at the wrong position. Both were fixed by
  using a class that *is* present (`bg-blue-500`, which the theme remaps to violet) or an
  inline style. The durable fix is `scripts/verify_page.py`, which now extracts every
  utility class the markup and JS rely on and **fails the build** if any is missing from the
  stylesheet.

> [!IMPORTANT]
> These three faults form the central lesson of this project. In each case the page appeared
> complete and every number on it was either chosen without evidence, inflated by a hidden
> fit failure, or invisible without complaint. An AI coding tool will produce all three, will
> describe the result as working, and will express confidence in it. The only reliable method
> is to state the expected result *before* running anything, and then to check it.

---

## 3. Further work for students

This project is deliberately **not** a research contribution. Predicting student performance
from engagement data is a well-established area, and this dataset has been used for it for
years. That is intentional: this is a teaching baseline, not a state-of-the-art result.

Students are encouraged to extend it, or to build something adjacent — a different cohort, a
different platform, or a sequence-aware feature set — so that their work addresses a question
that is genuinely not yet answered. Several extensions are suggested by the limitations in
the [main README](../README.md#6-known-limitations); the most direct are to add prior
attainment or assessment history, to move from a single snapshot per student to a genuine
time series, and to test whether the model remains valid at a different institution. A
further honest task is to replace the coarse system-assigned band with a real outcome, such
as a pass/fail or a graded mark.

---

## 4. The reproduction prompt

Give the prompt below to Gemini, OpenCode, Claude, ChatGPT or any coding agent. It encodes
every pitfall above, so a working page should come out first-pass.

```text
Build a complete, standalone, single-file HTML page for a student-performance prediction
demo (learning analytics), deployable on GitHub Pages. Everything inline, native ES6 only, no
frameworks. English throughout, including code comments.

PURPOSE — read this before designing anything. Learning data is abundant and dashboards to
display it are easy, but a chart can only describe what ALREADY happened, so it never leads
to an action. Someone asking "is this student heading for trouble — should we reach out now,
while there is still time to help?" cannot be helped by any amount of history. This page
closes that gap: the same engagement signals must produce a forward-looking estimate the
tutor can ACT on. Therefore the PREDICTION is the centre of the page, in the prime visual
position, with the evaluation tables supporting it and never competing with it. A tutor must
be able to set a student's profile, read a predicted level, see which Low/Medium/High band it
falls in, and know how far to trust it.

HARD CONSTRAINT — the finished index.html must make ZERO network requests. No CDN, no web
fonts, no fetch(), no XHR. A student must be able to double-click the file from disk, offline,
and have it work. Verify this by loading the page with every request EXCEPT the top-level
document aborted, and confirming it still renders and predicts. Do not report success until
that check passes.

STEP 0 — GET THE DATA FIRST. Assume the student does NOT have it. Never invent or fabricate a
dataset; download the real one and verify it before using it.
  mkdir -p data
  kaggle datasets download -d aljarah/xAPI-Edu-Data -p /tmp/xapi
  unzip -o /tmp/xapi/xAPI-Edu-Data.zip -d data     # yields xAPI-Edu-Data.csv
  (Requires the Kaggle CLI and a free API token at ~/.kaggle/kaggle.json.)
  VERIFY, and stop if these do not match:
    wc -l data/xAPI-Edu-Data.csv   -> 481  (480 data rows + 1 header)
    head -1 data/xAPI-Edu-Data.csv -> gender,NationalITy,PlaceofBirth,...
  Dataset: xAPI-Edu-Data (Kalboard 360), Amrieh, Hamtini & Aljarah (2016), 480 student
  records, 17 columns. Licence CC BY-SA 4.0. Each row is one anonymised student, recorded
  once — there is no per-lesson history.

STEP 1 — profile the data BEFORE choosing features, and report what you find:
  - No missing values (0 of 480x17). Use all 480 rows.
  - 2 exact duplicate rows (two identical pairs). Keep them; with no student id they may be
    two real students with identical attributes. Note the choice.
  - Class is imbalanced: M=211, H=142, L=127. Always-predict-M is right 44.0% of the time,
    so report BALANCED accuracy as well and use STRATIFIED folds.
  - StudentAbsenceDays (Under-7 / Above-7) is very strongly associated with the class
    (Cramer's V ~ 0.69). It is modifiable behaviour, so keep it, but state that much of the
    model's skill is this one coarse signal.

STEP 2 — features. Exactly these 6, in this order (the exported arrays index them):
  raisedhands, VisITedResources, AnnouncementsView, Discussion,
  absence_high (=1 if StudentAbsenceDays == 'Above-7' else 0),
  semester_s (=1 if Semester == 'S' else 0)
  Target: Class mapped L->0, M->1, H->2. Low/Medium/High bands are cuts of the 0-2 scale at
  0.5 and 1.5.
  Do NOT include the ten demographic/context columns (gender, NationalITy, PlaceofBirth,
  StageID, GradeID, SectionID, Topic, Relation, ParentAnsweringSurvey,
  ParentschoolSatisfaction). Measure their cost under the SAME folds (see STEP 3) and report
  it rather than asserting it: they lift accuracy from ~0.718 to ~0.731 and score ~0.547 on
  their own, yet a tutor cannot act on who a student is, so they are excluded.

STEP 3 — models (scikit-learn, seed 42). Before fitting anything, compute the BASELINES:
  - regression: DummyRegressor (predict the mean). Expect R2 = 0 by construction.
  - classification: DummyClassifier(most_frequent). Expect ~44.0%.
  Deploy what actually wins on held-out data. Do NOT assume boosting wins because another
  project used it. Compare under 5-fold STRATIFIED cross-validation repeated over 5 seeds,
  reporting mean +/- std:
    regression: LinearRegression vs GradientBoostingRegressor(n_estimators=250, max_depth=3,
      learning_rate=0.06). Expect linear to WIN: R2 ~0.637 +/- 0.002 vs ~0.600 +/- 0.012.
    classification: multinomial LogisticRegression (wrapped in a StandardScaler pipeline,
      max_iter=5000) vs GradientBoostingClassifier(n_estimators=200, max_depth=3,
      learning_rate=0.08). Expect logistic to win: ~71.8% vs ~69.3%.
    Also report accuracy from thresholding the regression output on the same folds (~71.9%),
      and say plainly that the classifier and the thresholded regression are
      indistinguishable on 480 rows.
  ALL reported metrics must be OUT-OF-FOLD, never training fit. The classifier MUST be fitted
  on standardised features: unscaled, lbfgs does not converge and silently reports a higher
  accuracy (~0.7229 vs ~0.7179) from an incomplete fit.

STEP 4 — export the models so no ML library is needed in the browser.
  - Regressor: one weight per feature plus an intercept. level = intercept + sum(w[j]*x[j]);
    clamp to 0-2 for display.
  - Classifier: per-class weight rows, per-class intercepts, and the StandardScaler's
    mean and scale. z[j] = (x[j]-mean[j])/scale[j]; score[k] = intercept[k] + sum(w[k][j]*z[j]);
    prediction = argmax(score). You do NOT need the softmax for the label.
  Then IMMEDIATELY re-derive predictions from the exported numbers with a hand-written
  implementation and diff against sklearn's own predict. A mismatch will not raise an error;
  it will return wrong numbers silently. Require JS-vs-Python agreement < 1e-9 and an
  independent sklearn refit that reproduces the exported weights.
  Feature importance: mean |standardised logistic coefficient| across the three classes, so
  magnitudes are comparable across features measured in different units. Expect absence_high
  largest, then VisITedResources and raisedhands.

STEP 5 — the page. English throughout.
  - LEAD WITH THE ACTIONABLE OUTPUT. Show the predicted level and the Low/Medium/High class
    at the top, in the largest type on the page, with a marker on the 0-2 scale against the
    0.5 and 1.5 band boundaries, because those boundaries ARE the decision. Evaluation tables
    come after this, never before it.
  - Support "what if I support this student?" directly: sliders for raised hands, resources
    visited, announcements viewed and discussion posts, plus toggles for the absence band and
    semester, with the prediction updating live. A user who can only replay historical rows
    has been given a dashboard, not a predictor.
  - Slider bounds MUST be derived at runtime from the model's own input_ranges. Do NOT
    hard-code min/max: a slider narrower than the training data is a silent failure.
  - A prominent evaluation panel: the regression table (baseline / linear / boosted), the
    classification table (majority / logistic / thresholded regression / boosted) with
    balanced accuracy, the full confusion matrix with row percentages, and feature
    importances with a plain-language reading.
  - An 80% prediction interval from empirical out-of-fold residual quantiles, bucketed by
    predicted level, explained in words. Do not present it as a probabilistic interval.
  - A "load a random real student" button cycling through 60 held-out rows, showing the
    observed class beside the prediction. Use a seeded/deterministic cycle, not Math.random.
  - A provenance panel with the dataset, the citation, the exact hyperparameters, the fairness
    exclusion and its measured cost, and the external-validity limit: this model learned one
    Kalboard 360 cohort only and is a demonstration of method, NOT a live early-warning system.

STEP 6 — if you use a utility-class CSS framework (Tailwind, Bootstrap), do NOT leave it on a
CDN. Load the page with the CDN active, exercise EVERY interactive state so its observer
emits rules for classes that only appear after JS inserts them (drag every slider, click every
toggle, click the random-student button), then read the generated <style> out of the DOM and
vendor it into the file. Note the consequence: adding a new utility class without regenerating
yields an unstyled element and NO error — so your test suite MUST cross-check every class used
in the markup against the class selectors present in the stylesheet, and fail the build if any
is missing. This is not theoretical: a bar chart once shipped invisible because its colour
class was not in the snapshot.

STEP 7 — do not stop until all of the following are true, and report each as PASS/FAIL with
the actual measured number, not a summary:
  1. Extract the pure-model region of index.html (mark it with explicit BEGIN/END comment
     markers), run it under Node.js, and diff its predictions against an INDEPENDENT Python
     implementation across the default inputs, all 60 samples, and the corners of each input
     range. Write the Python side without importing your training code.
  2. Independently refit both models with scikit-learn and confirm the exported weights match
     to < 1e-9.
  3. Confirm every class used by the markup is implemented in the vendored stylesheet.
  4. Load the page in a browser with all non-document requests aborted. Confirm zero blocked
     needed requests and zero console errors.
  5. Confirm the page's displayed number equals the Python-computed number for the default
     inputs.
  6. Confirm no horizontal overflow at 390px and at 1280px.
```

---

Back to the [main README](../README.md) ·
Model and verification notes: [model-notes.md](model-notes.md) ·
Dataset card: [dataset.md](dataset.md)
