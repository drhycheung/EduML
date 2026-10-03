# Student Performance Prediction Demo (Learning Analytics Teaching Demo)

**Live demo (GitHub Pages): <https://drhycheung.github.io/EduML/>**

![Student Performance Prediction Demo screenshot](docs/screenshot.png)

A single-file, front-end-only interactive demo that predicts **student performance level**
and a three-band performance class from engagement and attendance indicators, using
gradient-boosted trees that run entirely in the browser. Built for classroom demonstration
in **Learning Analytics**, **Educational Data Mining** and **AI in Education** courses.

File: `index.html` — no build step, no backend, no API key, **no network requests at
all**. Double-click it from disk and it works, offline. Drop it into a GitHub Pages
repository and it is deployed.

**Why this exists.** Learning data is abundant and dashboards to display it are easy, but a
chart can only ever describe what has already happened. Someone asking *"is this student
heading for trouble — should we reach out now, while there is still time to help?"* cannot
be helped by any amount of history. Data without prediction produces description, not
decisions. This page takes the same engagement signals a dashboard would show and turns
them into a forward-looking estimate a tutor can act on, with the uncertainty attached so
they know how far to trust it.

A prediction nobody trusts is as useless as no prediction at all, so the page also shows
its own evaluation against trivial baselines, says plainly where the flexible model
**loses** to a simple one, and explains the group of attributes it deliberately threw away.

---

## 1. From data to decision

The whole project is one distinction, and it is worth stating before the feature list:

| Question                                    | Answered by                                                   | Not answered by |
| ------------------------------------------- | ------------------------------------------------------------- | --------------- |
| "How engaged has this student been?"        | A dashboard, a chart, a click count                           | —               |
| "Is this student heading for trouble?"      | —                                                             | A dashboard     |
| "Should I offer targeted support this week?"| **A prediction, shown against the band that triggers action** | —               |
| "How much should I believe it?"             | Baselines, an empirical interval, and a stated exclusion      | —               |

The prediction therefore occupies the prime position on the page, in the largest type,
with the Low / Medium / High scale showing where the estimate lands against the 0.5 and 1.5
level boundaries — because that band *is* the decision the tutor is trying to make. The
evaluation tables are supporting evidence, not the headline.

A user must be able to set a student's profile and get a prediction. A tool that can only
replay historical rows has quietly become a dashboard; that distinction is the teaching
point.

## 2. What the page does

| Feature                                          | Implementation                                                                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Predict performance level from 4 sliders + 2 toggles** | The headline output, centre of the page. Gradient-boosted trees serialised to flat arrays; traversal in ~15 lines of vanilla JS |
| **Actionable class against a band**              | A second ensemble gives Low / Medium / High, with a marker on the 0–2 level scale at the 0.5 and 1.5 boundaries                  |
| Evaluate any student profile                     | Sliders accept any engagement counts, absence band and semester within the training range                                       |
| No ML library in the browser                     | scikit-learn's fitted trees exported as `[feature, threshold, left, right, value]` at stride 5                                  |
| Honest evaluation panel                          | Out-of-fold 5-fold CV metrics, with a predict-the-mean baseline and a linear-regression baseline beside the model               |
| Prediction interval                              | Empirical 10th–90th percentile of out-of-fold residuals, bucketed by predicted level                                            |
| An honest loss                                   | The page states that boosting scores *below* plain linear regression here (R² 0.594 vs 0.637) and explains why that is not a bug |
| Task-design comparison                           | Accuracy of direct classification vs. thresholding the regression output                                                        |
| Feature importance                               | Table with bars, plus a plain-language reading of why resource-visiting dominates                                               |
| "Load a random real student"                     | 60 held-out rows; shows the model's prediction next to the observed class                                                       |
| Slider bounds from the data                      | Every slider range is read from the model file, so it can never be narrower than the training data                              |
| Zero dependencies                                | All CSS vendored inline, no CDN, no web fonts, no `fetch()`                                                                     |

> **New to machine learning?** §1 of the [model notes](docs/model-notes.md#1-the-algorithms-in-plain-language)
> explains baselines, linear regression, decision trees, gradient boosting and
> cross-validation from first principles, and traces a real prediction through the
> exported trees.

## 3. Data, and the attributes that had to go

Source: [xAPI-Edu-Data (Kalboard 360)](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data)
(Amrieh, Hamtini & Aljarah, 2016), 480 student records from a multi-agent learning
management system. The dataset has no missing values, so all **480** rows are used.

<details>

<summary><strong>Download the data</strong> (if you do not have it — a copy is already committed at <code>data/xAPI-Edu-Data.csv</code>)</summary>

```bash
mkdir -p data
kaggle datasets download -d aljarah/xAPI-Edu-Data -p /tmp/xapi
unzip -o /tmp/xapi/xAPI-Edu-Data.zip -d data     # yields xAPI-Edu-Data.csv
```

This needs the Kaggle CLI and a free API token at `~/.kaggle/kaggle.json`
(Kaggle → Account → Create New API Token).

Verify before you use it:

```bash
wc -l data/xAPI-Edu-Data.csv     # must be 481  (480 rows + header)
head -1 data/xAPI-Edu-Data.csv   # gender,NationalITy,PlaceofBirth,...,Class
```

The committed copy is byte-identical to a fresh download.

</details>

Six features: times the student raised a hand, course resources visited, announcements
viewed, discussion posts, an absence band (7 or more absence days), and the semester.

**Who the student is, rather than what they do, is missing — and that is the most
important thing on the page.** The dataset also carries ten demographic and context
attributes: gender, nationality, place of birth, educational stage, grade, section, topic,
parental relation, whether a parent answered the survey, and parental school satisfaction.

Adding those ten attributes raises cross-validated accuracy from 0.715 to **0.771**. That
"improvement" is not something a tutor can act on: the model would be predicting a
student's outcome from *who they are* — nationality, family, which section they were
sorted into — rather than from anything support could change. Used to triage real support,
it would launder historical group inequity into a decision about a named child. The gain is
bought with group membership, not with learning, so the attributes are dropped and the page
says so out loud. On their own, the demographics alone reach only 0.631 — close to the
majority-class baseline — which is the honest measure of how much they explain directly.

## 4. Results

All out-of-fold, 5-fold shuffled cross-validation, seed 42.

**Regression — performance level (Low 0 · Medium 1 · High 2)**

| Method                                | R²        | RMSE  | MAE   |
| ------------------------------------- | --------- | ----- | ----- |
| Always predict the training mean      | −0.000    | 0.748 | 0.573 |
| Multiple linear regression            | **0.637** | 0.450 | 0.365 |
| **Gradient-boosted trees (the page)** | 0.594     | 0.476 | 0.368 |

Here the *simple* model wins. With only 480 rows and six near-additive inputs, the boosted
trees have ample capacity to fit sampling noise that does not reappear in the held-out fold,
so plain linear regression generalises slightly better. Model complexity has to earn its
keep, and the page reports the loss rather than hiding it. The remaining ~41% of the
variance is not a modelling failure: engagement is not the same thing as ability.

**Classification — three-band performance level** (Low < 0.5 · Medium 0.5–1.5 · High ≥ 1.5)

| Method                                              | Accuracy   | Balanced accuracy |
| --------------------------------------------------- | ---------- | ----------------- |
| Always predict the majority class                   | 43.96%     | 50.0%             |
| **Gradient-boosted classifier (the page)**          | **71.46%** | 72.71%            |
| Bands derived by thresholding the regression output | 69.58%     | —                 |

That last row is a design lesson, not a model ranking: a level error of half a band flips
borderline students, so "predict a number then threshold it" and "classify" are different
tasks with different error budgets. The full confusion matrix is printed on the page.

## 5. How to run

- **Students, teachers, anyone**: double-click `index.html`. It needs nothing else —
  no server, no network, no installation. This is the intended way to use it.
- **Locally with a server** (only if you want to edit): `python3 -m http.server 8000`,
  then visit `http://localhost:8000/index.html`. Functionally identical.
- **GitHub Pages (your own deployment)**: push `index.html` to *your* repository, then
  enable Pages via **Settings → Pages → Deploy from a branch** (branch + `/ (root)`).
  Yours will live at `https://<your-username>.github.io/<repo-name>/`. The link at the
  top of this README is the author's own deployment.

### Rebuilding from source

`index.html` is generated. Edit the sources, not the output.

```bash
python3 scripts/train_model.py    # retrain -> model/model.json   (~10 s)
python3 scripts/build_page.py     # inline model + CSS -> index.html
python3 scripts/verify_page.py    # prove the page still agrees with Python
```

`verify_page.py` runs the in-page tree traversal under Node.js on 200 synthetic rows and
exits non-zero if it disagrees with Python by more than 1e-9. See the
[model notes](docs/model-notes.md) for what it covers and why it exists.

## 6. Known limitations

Several of these are consequences of the constraints in §3 rather than oversights:

| Limitation                                          | Consequence                                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Trained **only** on one Kalboard 360 cohort         | Applying it to another institution, age group or platform degrades markedly. It is a demonstration of method, not a live early-warning system |
| **Demographics and context excluded**               | Caps accuracy at 71.5%; §3 explains why the extra 5.6 points are not worth the fairness cost                  |
| Engagement inputs only — no prior attainment        | The unexplained ~41% of variance is structural, not fixable by a better learner                               |
| Single-platform training set                        | No claim of generalisation across institutions, sectors or instrument types                                  |
| Class bands are ordinal level cuts at 0.5 / 1.5     | Adjacent bands have no sharp educational boundary, which is why the confusion matrix is full of near-misses    |
| Prediction interval is empirical, not probabilistic | It reports where the truth *usually* landed for similar students, not a calibrated 80% credible interval      |
| Correlation, not causation                          | A student who opens more resources is not *made* more able by the clicking                                    |
| A model that predicts risk can also create it       | Used to withdraw support, or to label a flagged student, it can harm the very outcomes it predicts             |
| Leaf values rounded when exporting                  | The JS-vs-Python self-check absorbs this (max deviation 4.7 × 10⁻¹⁵), but the browser is not bit-identical to a live sklearn model |
| `src/vendored.css.html` is a vendored snapshot      | Adding a new utility class without regenerating it yields an unstyled element; `verify_page.py` fails the build to stop this |
| Desktop-first layout                                | Tested at 390 px and 1280 px with no horizontal overflow, but not a full mobile design                        |

## 7. Documentation

| Document                               | What it covers                                                                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Model notes](docs/model-notes.md)** | The algorithms from first principles, tree serialisation format, the empirical interval, the JS-vs-Python verification harness, and why boosting loses to linear regression here |

## 8. Licences & attribution

- **Code**: MIT — see [LICENSE](LICENSE), © 2026 drhycheung.
- **Data**: [xAPI-Edu-Data (Kalboard 360)](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data)
  (Amrieh, Hamtini & Aljarah, 2016), CC BY-SA 4.0, distributed via Kaggle. Copyright
  remains with the original authors, The University of Jordan.

The two are separately licensed: the MIT licence covers this repository's code only and
does not extend to the dataset, which stays under CC BY-SA 4.0.
