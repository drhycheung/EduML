# Dataset card — xAPI-Edu-Data (Kalboard 360)

## Source

| | |
|---|---|
| **Name** | xAPI-Edu-Data (Kalboard 360) |
| **Provider** | Kaggle (originally collected from the Kalboard 360 learning management system) |
| **Link** | <https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data> |
| **Local copy** | `data/xAPI-Edu-Data.csv` (about 40 KB) |
| **Licence** | CC BY-SA 4.0 |
| **Citation** | Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). Mining Educational Data to Predict Student's Academic Performance using Ensemble Methods. *International Journal of Database Theory and Application, 9*(8), 119–136. |

## Contents

**480** rows and **17** columns. Each row is **one anonymised student**, recorded once —
not one lesson, one click or one session. There is no student identifier, so rows cannot
be linked into a longitudinal history.

The `Class` column is the target: a three-way performance band assigned by the system.

| Column | Meaning | Used? |
|---|---|---|
| `gender` | Student gender | no — demographic, see below |
| `NationalITy` | Student nationality | no — demographic |
| `PlaceofBirth` | Student place of birth | no — demographic |
| `StageID` | Educational stage (`lowerlevel` / `middleschool` / `highschool`) | no — context |
| `GradeID` | Grade within the stage (`G-01` … `G-12`) | no — context |
| `SectionID` | Class section (`A` / `B` / `C`) | no — context |
| `Topic` | Course topic (e.g. `French`, `Math`, `Science`) | no — context |
| `Semester` | Term the record belongs to (`F` / `S`) | yes — as `semester_s` |
| `Relation` | Parent responsible for the student (`Father` / `Mum`) | no — demographic |
| `raisedhands` | Times the student raised a hand | **yes** |
| `VisITedResources` | Course resources opened in the LMS | **yes** |
| `AnnouncementsView` | Announcements viewed | **yes** |
| `Discussion` | Discussion-group messages posted | **yes** |
| `ParentAnsweringSurvey` | Whether a parent answered the survey (`Yes` / `No`) | no — context |
| `ParentschoolSatisfaction` | Parental satisfaction (`Good` / `Bad`) | no — context |
| `StudentAbsenceDays` | Absence band (`Under-7` / `Above-7`) | yes — as `absence_high` |
| `Class` | Performance band (`L` / `M` / `H`) | **target** |

## How this project uses it

- **Target.** `Class` is mapped to a level: `L → 0`, `M → 1`, `H → 2`. The Low / Medium /
  High class is then a cut of that 0–2 scale at **0.5** and **1.5**. Regression predicts the
  level; classification predicts the three bands directly (`scripts/train_model.py`).
- **Feature engineering.** Two columns are turned into 0/1 indicators:
  `absence_high = 1` when `StudentAbsenceDays == 'Above-7'`, and
  `semester_s = 1` when `Semester == 'S'`. The other four features are used as-is.
- **Six features, in this order:** `raisedhands`, `VisITedResources`, `AnnouncementsView`,
  `Discussion`, `absence_high`, `semester_s`. The order is load-bearing: the exported model
  arrays are indexed by it.
- **All 480 rows are used.** The file has no missing values, so nothing is dropped or
  imputed before training.
- **The ten demographic and context attributes are deliberately excluded.** Adding all ten
  to the deployed logistic classifier raises out-of-fold accuracy from **0.718 ± 0.003** to
  **0.731 ± 0.013** — a modest but real 1.3-point gain — and on their own they reach
  **0.547 ± 0.010**, ten points above the 0.440 majority-class baseline, so they clearly
  carry predictive signal. They are still dropped. A tutor cannot act on a student's
  nationality or which section they were sorted into, and a model that ranks students by
  *who they are* rather than *what they do* would launder historical group inequity into a
  decision about a named child. The trade-off is stated on the page and measured in
  `scripts/train_model.py` rather than asserted.

## Data-quality notes

- **No missing values.** All 480 × 17 cells are populated, so no cleaning is needed to fit.
  This is unusual for educational data and is worth stating: the honesty of this demo rests
  on fairness choices, not on repair of a broken file.
- **2 duplicate rows.** Two pairs of rows are exactly identical (four rows in total). They
  are kept: with no student identifier, identical profiles could be two real students who
  happen to share every recorded attribute, so deleting them would require a rule that the
  data cannot justify. Their effect on 480-row cross-validation is negligible.
- **Class imbalance.** The bands are Medium 211, High 142, Low 127. Always-predict-Medium
  is therefore right **44.0%** of the time, so the page reports *balanced* accuracy
  alongside raw accuracy, and cross-validation is **stratified** so a fold cannot by chance
  lose a small class.
- **`absence_high` is close to a direct proxy for the label.** The absence band is very
  strongly associated with the performance class (Cramér's V ≈ 0.69): of the 191 students
  with `Above-7`, 116 are Low; of the 289 with `Under-7`, 138 are High. It is legitimately
  modifiable behaviour, not a demographic, so it is kept — and it is the single most
  important feature on the page. But much of the model's apparent skill is this one coarse
  signal, which is worth stating plainly.
- **The labels are coarse.** `Class` is a system-assigned band, not an examination mark.
  The 0.5 and 1.5 cuts are arbitrary points on an ordinal scale, which is why the confusion
  matrix is full of near-misses between neighbouring bands.
- **Feature correlations with the level** (Pearson): `VisITedResources` +0.677,
  `absence_high` −0.671, `raisedhands` +0.646, `AnnouncementsView` +0.527, `Discussion`
  +0.308, `semester_s` +0.126. These are correlations, not causal effects.

## Licence and attribution

The data is published under **CC BY-SA 4.0**. Copyright remains with the original authors,
The University of Jordan. The committed copy in `data/` is redistributed under the same
licence; derivative uses of the data must carry the same licence. This project's **code**
is MIT licensed, which is separate: the MIT licence covers the repository's code only, not
the dataset.
