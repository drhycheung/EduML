# EduML — Learning Analytics At-Risk Prediction Demo

**Live demo (GitHub Pages): <https://drhycheung.github.io/EduML/>**

![EduML demo screenshot](docs/screenshot.png)

A single-file, front-end-only interactive demo that predicts a student's
**performance level** and a three-band **Low / Medium / High** class from
engagement and attendance indicators, using gradient-boosted trees that run
entirely in the browser. Built as a classroom companion to the EnvML
environmental-ML demo, for **Learning Analytics**, **Educational Data Mining**
and **AI in Education** courses.

File: `index.html` — no build step, no backend, no API key, **no network
requests at all**. Double-click it from disk and it works, offline. Drop it
into a GitHub Pages repository and it is deployed.

**Why this exists.** Learning analytics dashboards make it easy to *describe*
engagement after the fact — clicks, logins, submissions. But a chart can only
ever describe what has already happened. Someone asking *"is this student
heading for trouble — should I reach out now, while there is still time to
help?"* cannot be helped by any amount of history. Data without prediction
produces description, not decisions. This page takes the same engagement
indicators and turns them into a forward-looking estimate a tutor can act on,
with the uncertainty attached.

A prediction nobody trusts is as useless as no prediction at all, so the page
also shows its own evaluation against trivial baselines, states plainly where
the flexible model *loses* to a simple one, and explains the one feature it
deliberately threw away.

---

## 1. From data to decision

| Question | Answered by | Not answered by |
| --- | --- | --- |
| "How engaged has this student been?" | A dashboard, a chart, a click count | — |
| "Is this student at risk of low performance?" | — | A dashboard |
| "Should I offer targeted support this week?" | **A prediction, shown against the band that triggers action** | — |
| "How much should I believe it?" | Baselines, an empirical interval, and stated limits | — |

The prediction therefore occupies the prime position on the page, in the largest
type, with the Low / Medium / High scale showing where the estimate lands —
because that band *is* the decision the tutor is trying to make. The evaluation
tables are supporting evidence, not the headline.

A tool that can only replay historical rows has quietly become a dashboard;
this demo exists to show the difference.

## 2. What the page does

| Feature | Implementation |
| --- | --- |
| **Predict performance level from 4 sliders + 2 toggles** | The headline output. Gradient-boosted trees serialised to flat arrays; traversal in ~15 lines of vanilla JS |
| **Actionable class against a band** | A second ensemble gives Low / Medium / High, with a marker on the 0–2 scale |
| Evaluate any student profile | Sliders accept any engagement counts, absence band and semester in the training range |
| No ML library in the browser | scikit-learn's fitted trees exported as `[feature, threshold, left, right, value]` at stride 5 |
| Honest evaluation panel | Out-of-fold 5-fold CV, with a predict-the-mean and a multiple-linear-regression baseline beside the model |
| Prediction interval | Empirical 10th–90th percentile of out-of-fold residuals, bucketed by predicted level |
| An honest loss | On this small dataset the boosted regressor scores **lower** than plain linear regression, and the page says so and explains why |
| Task-design comparison | Accuracy of direct classification vs. thresholding the regression output |
| Feature importance | Table with bars, plus a plain-language reading of why resource-visiting dominates |
| "Load a random real student" | 60 real rows; shows the model's prediction next to the observed class |
| Slider bounds from the data | Every slider range is read from the model file, so it can never be narrower than the training data |
| Zero dependencies | All CSS vendored inline, no CDN, no web fonts, no `fetch()` |

> **New to machine learning?** `docs/model-notes.md` explains baselines, linear
> regression, decision trees, gradient boosting and cross-validation from first
> principles.

## 3. Data, and the one feature that had to go

Source: [xAPI-Edu-Data (Kalboard 360)](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data),
480 student records. The target is the student's performance class, mapped to an
ordinal level (Low = 0, Medium = 1, High = 2).

Features: `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`,
an absence band (`Under-7` / `Above-7`) and the semester.

**Deliberately excluded:** the student's final class result and anything derived
from it, plus nationality, place of birth and parental-survey answers. Excluding
the result is what stops the otherwise trivial "predict the result from the
result" leak.

Class balance: Low 127 · Medium 211 · High 142 (majority-class baseline 44.0%).

## 4. Honest evaluation at a glance

All figures are out-of-fold from 5-fold shuffled cross-validation: each record
is scored by a model that never saw it during fitting.

| Task | Metric | Baseline | Linear | This page (GBM) |
| --- | --- | --- | --- | --- |
| Regression (level 0–2) | R² | 0.000 | **0.637** | 0.594 |
| Regression (level 0–2) | RMSE | 0.748 | **0.450** | 0.476 |
| Classification (3 bands) | Accuracy | 0.440 | — | **0.715** |

Two lessons the page makes explicit:

1. **A more complicated model is not automatically better.** Linear regression
   beats the boosted trees on the regression task here — on 480 students the
   boosted model has enough capacity to fit noise. Always put the simple
   baseline in the table before trusting the fancy result.
2. **"Predict a number, then threshold it" is not the same task as "classify".**
   Thresholding the regression output gives 69.6% accuracy; the dedicated
   classifier gives 71.5%. That is a task-design choice, not a modelling victory.

## 5. Repository layout

```
index.html            generated single-file demo (the deliverable)
model/model.json      trained weights + metrics + sample rows
data/xAPI-Edu-Data.csv  the dataset
src/vendored.css.html vendored Tailwind snapshot (no CDN at runtime)
scripts/train_model.py  retrain and export model.json
scripts/build_page.py   inline model.json + CSS into index.html
scripts/verify_page.py  prove the JS traversal matches Python
docs/model-notes.md     first-principles model explanation
```

```
pip install scikit-learn pandas numpy
python3 scripts/train_model.py     # retrain and export model.json
python3 scripts/build_page.py      # rebuild index.html
python3 scripts/verify_page.py     # JS == Python check
```

## 6. Limits

The model has only learned the engagement–performance relationship of one
cohort using the Kalboard 360 platform. Applying it to another institution, age
group or platform will degrade markedly. Treat the numbers as a demonstration of
"what machine learning can achieve on this dataset", not as a live early-warning
system.

A model that predicts risk can also create it, if a tutor withdraws support from
a student the model labels "fine", or labels a student the model flags. Any real
deployment needs bias evaluation across groups and must be used to *direct
support*, never to justify its withdrawal.

## Licence

Code released under the MIT licence. Data: xAPI-Edu-Data (Kalboard 360),
distributed via Kaggle (Amrieh, Hamtini & Aljarah, 2016).
