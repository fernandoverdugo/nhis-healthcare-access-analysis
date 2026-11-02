# 2025 NHIS Healthcare Access Analysis

A reproducible Python project combining **complex-survey inference, multiple imputation, and predictive modeling** to examine cost-related delayed medical care among U.S. adults using the **2025 National Health Interview Survey (NHIS)**.

The project deliberately separates two questions:

1. **Population inference:** Which groups experience the highest prevalence of delaying medical care because of cost?
2. **Predictive modeling:** How well can a curated set of demographic, socioeconomic, access, and health characteristics identify adults at elevated risk?

NHIS is a complex probability survey, so survey-weighted population estimates and predictive model performance are treated as related but distinct analytical products.

**Review the analysis:** [Exploratory notebook](notebooks/01_exploratory_analysis.ipynb) Â· [Survey + modeling notebook](notebooks/02_survey_and_modeling_analysis.ipynb) Â· [Nonlinear benchmark notebook](notebooks/03_nonlinear_benchmark_and_interpretability.ipynb)

![Weighted model discrimination metrics](assets/model_discrimination_metrics.svg)

## Portfolio highlights

- 2025 NHIS Sample Adult analysis with **24,215 respondents**
- survey-weighted prevalence estimates using the final annual weight, public stratum, and PSU variables
- Taylor-linearized confidence intervals for overall and subgroup estimates
- all **10 official income imputations** retained and pooled rather than reduced to a single imputation
- curated feature design instead of indiscriminately modeling all 577 public-use adult variables
- held-out logistic-regression baselines and one nonlinear gradient-boosting benchmark
- weighted ROC-AUC, average precision, Brier score, log loss, calibration, and risk-concentration diagnostics
- held-out permutation importance for model interpretability
- **3 fully executed review notebooks** with rendered tables and linked SVG figures
- automated source validation and analytical tests
- reproducible CDC/NCHS data download with checksum verification

## Primary outcome

`MEDDL12M_A` â€” delayed medical care because of cost during the past 12 months.

The closely related `MEDNG12M_A` variable â€” needed medical care but did not get it because of cost â€” is retained as a **secondary outcome** and intentionally excluded from the predictor matrix to prevent target leakage.

## Key findings

These results describe the 2025 NHIS public-use Sample Adult data and should be interpreted as **associational, not causal**.

### Survey-weighted population estimates

- Estimated prevalence of delayed medical care because of cost is **7.05%** (95% CI **6.62%â€“7.48%**).
- Adults without health insurance have an estimated prevalence of **26.6%**, compared with **5.1%** among covered adults.
- Adults with no usual place of care are at **15.0%**, compared with **6.1%** among adults with one usual place of care.
- Adults reporting poor general health are at **17.2%**, compared with **4.1%** among adults reporting excellent health.
- Adults with a functioning difficulty are at **9.4%**, compared with **4.7%** among adults without one.
- Renters are at **10.7%**, compared with **5.4%** among adults living in an owned or being-bought residence.
- The South has the highest regional estimate at **8.6%**, while the Northeast is lowest at **5.1%**.

### Multiply imputed income analysis

Using all 10 official income imputations, pooled delayed-care prevalence is approximately:

- **10.4%** below 100% of the federal poverty level
- **11.1%** at 100â€“199% FPL
- **8.6%** at 200â€“399% FPL
- **3.6%** at 400%+ FPL

Each completed income dataset is analyzed separately. Point estimates and Taylor-linearized within-imputation variances are then combined using Rubin pooling with the Barnardâ€“Rubin degrees-of-freedom adjustment.

### Predictive modeling

On the fixed held-out test set:

| Model | Weighted ROC-AUC | Weighted average precision | Weighted Brier score |
|---|---:|---:|---:|
| Prevalence baseline | 0.500 | 0.072 | 0.0668 |
| Core logistic regression | **0.738** | **0.231** | **0.0612** |
| Expanded logistic regression | **0.773** | **0.272** | **0.0599** |
| Expanded gradient boosting | **0.781** | **0.267** | **0.0599** |

The expanded specification adds general health and functioning difficulty to the access/demographic core model. Gradient boosting provides a modest improvement in weighted ROC-AUC and probability-loss metrics, but its weighted average precision is slightly below expanded logistic regression. The more complex model therefore does **not** dominate on every metric.

Adding multiply imputed poverty ratio produces a further modest and highly stable improvement in the logistic specifications: mean weighted ROC-AUC rises to approximately **0.751** for the core-plus-income model and **0.781** for the expanded-plus-income model across the 10 imputations.

The nonlinear model is useful as a risk-ranking benchmark: the **highest-risk 10%** of the held-out sample contains about **42.4%** of weighted positive outcomes, and the highest-risk 20% contains about **61.0%**. Held-out permutation importance ranks insurance status and age highest, followed by functioning difficulty and general health. These are predictive importance measures, not causal effects.

Because the positive outcome is uncommon, a default probability cutoff of 0.50 yields low recall. The project therefore emphasizes discrimination, precision-recall performance, calibration-oriented loss metrics, and risk ranking rather than raw accuracy or an arbitrary operating threshold.

## Analytical design

**Core model:** age, sex, race/ethnicity, education, marital status, family composition, housing tenure, employment, nativity, insurance status, usual place of care, region, and urban-rural classification.

**Expanded model:** the core model plus general health status and functioning difficulty.

**Nonlinear benchmark:** histogram gradient boosting using the expanded feature set and the identical held-out test set.

**Income extensions:** the core and expanded logistic models plus top-coded family poverty ratio, refit independently for each of the 10 official income imputations.

Citizenship status is retained for descriptive analysis but intentionally excluded from the default predictive models.

## Notebooks

### 1. [Exploratory analysis](notebooks/01_exploratory_analysis.ipynb)

Compact review of outcome prevalence and survey-weighted subgroup estimates generated by the reproducible analysis pipeline.

### 2. [Survey analysis and predictive modeling](notebooks/02_survey_and_modeling_analysis.ipynb)

Reviews survey-weighted prevalence with Taylor-linearized confidence intervals, pooled income estimates across all 10 imputations, and held-out predictive-model performance generated by the pipeline.

### 3. [Nonlinear benchmark and interpretability](notebooks/03_nonlinear_benchmark_and_interpretability.ipynb)

Reviews the nonlinear benchmark through permutation importance, weighted calibration, risk deciles, and cumulative concentration of positive outcomes.

## Repository structure

```text
.
â”œâ”€â”€ .github/workflows/quality-checks.yml
â”œâ”€â”€ assets/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â””â”€â”€ README.md
â”‚   â”œâ”€â”€ processed/
â”‚   â”‚   â”œâ”€â”€ survey_group_estimates.csv
â”‚   â”‚   â”œâ”€â”€ income_pooled_estimates.csv
â”‚   â”‚   â”œâ”€â”€ model_metrics.csv
â”‚   â”‚   â”œâ”€â”€ boosting_permutation_importance.csv
â”‚   â”‚   â”œâ”€â”€ boosting_calibration.csv
â”‚   â”‚   â”œâ”€â”€ boosting_risk_deciles.csv
â”‚   â”‚   â””â”€â”€ â€¦â”‚   â””â”€â”€ README.md
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_exploratory_analysis.ipynb
â”‚   â”œâ”€â”€ 02_survey_and_modeling_analysis.ipynb
â”‚   â””â”€â”€ 03_nonlinear_benchmark_and_interpretability.ipynb
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ download_data.py
â”‚   â”œâ”€â”€ prepare_data.py
â”‚   â”œâ”€â”€ survey_analysis.py
â”‚   â”œâ”€â”€ train_models.py
â”‚   â””â”€â”€ run_pipeline.py
â”œâ”€â”€ tests/
â”œâ”€â”€ DATA_USE.md
â”œâ”€â”€ LICENSE
â”œâ”€â”€É•ÅÕ¥É•µ•¹ÑÌ¹ÑáĞ+ŠRSŠRŠR I5¹µ)€()I•ÍÁ½¹‘•¹Ğµ±•Ù•°ÁÉ½•ÍÍ•¥¹Ñ•Éµ•‘¥…Ñ•Ì…É”É•ÁÉ½‘Õ¥‰±”‰ÕĞ¥¹Ñ•¹Ñ¥½¹…±±ä•á±Õ‘•™É½´¥Ğ¡¥ÍÑ½Éä¸Q¡”É•Á½Í¥Ñ½ÉäÉ•Ñ…¥¹Ì½¹±ä…É•…Ñ”½µ½‘•°µÍÕµµ…Éä…¹…±åÑ¥…°½ÕÑÁÕÑÌ¹••‘•Ñ¼É•Ù¥•ÜÑ¡”É•ÍÕ±ÑÌ¸((ŒŒI•ÁÉ½‘Õ”Ñ¡”ÁÉ½©•Ğ()Q¡”É•Á½Í¥Ñ½ÉäÑ…É•ÑÌ€¨©AåÑ¡½¸€Ì¸ÄÔ¨¨…¹Á¥¹ÌÑ¡”Ñ•ÍÑ•‘•Á•¹‘•¹äÍ•Ğ¥¸É•ÅÕ¥É•µ•¹ÑÌ¹ÑáÑ€¸()‰…Í )ÁåÑ¡½¸Ì€µ´Ù•¹Ø€¹Ù•¹Ø)Í½ÕÉ”€¹Ù•¹Ø½‰¥¸½…Ñ¥Ù…Ñ”)ÁåÑ¡½¸€µ´Á¥À¥¹ÍÑ…±°€´µÕÁÉ…‘”Á¥À)ÁåÑ¡½¸€µ´Á¥À¥¹ÍÑ…±°€µÈÉ•ÅÕ¥É•µ•¹ÑÌ¹ÑáĞ)€()½İ¹±½…Ñ¡”Ñİ¼½™™¥¥…°€ÈÀÈÔ9!%LÁÕ‰±¥ŒµÕÍ”™¥±•Ì…¹Ù•É¥™äÑ¡•´……¥¹ÍĞÑ¡”9!LµÁÕ‰±¥Í¡•5Ô¡•­ÍÕµÌè()‰…Í )ÁåÑ¡½¸ÍÉ¥ÁÑÌ½‘½İ¹±½…‘}‘…Ñ„¹Áä)€()IÕ¸Ñ¡”½µÁ±•Ñ”…¹…±åÑ¥…°Á¥Á•±¥¹”è()‰…Í )ÁåÑ¡½¸ÍÉ¥ÁÑÌ½ÉÕ¹}Á¥Á•±¥¹”¹Áä)€()IÕ¸Ñ¡”Ù…±¥‘…Ñ¥½¸ÍÕ¥Ñ”è()‰…Í )ÁåÑ¡½¸€µ´ÁåÑ•ÍĞ€µÄ€µÀ¹¼é…¡•ÁÉ½Ù¥‘•È)€()=Á•¸Ñ¡”¹½Ñ•‰½½­Ì±½…±±äİ¥Ñ è()‰…Í )©ÕÁåÑ•È±…ˆ)€()¥Ñ!ÕˆÑ¥½¹ÌÉ•Á•…ÑÌÍ½ÕÉ”‘½İ¹±½…½¡•­ÍÕ´Ù•É¥™¥…Ñ¥½¸°Ñ¡”…¹…±åÑ¥…°É•‰Õ¥±°…¹Ñ¡”™Õ±°Ù…±¥‘…Ñ¥½¸ÍÕ¥Ñ”½¸•Ù•ÉäÁÕÍ ½ÈÁÕ±°É•ÅÕ•ÍĞ¸((ŒŒ5•Ñ¡½‘½±½¥…°Õ…É‘É…¥±Ì((ŒŒŒ½µÁ±•àÍÕÉÙ•ä¥¹™•É•¹”()A½ÁÕ±…Ñ¥½¸•ÍÑ¥µ…Ñ•ÌÕÍ”è((´]Q}€ƒŠP™¥¹…°…¹¹Õ…°M…µÁ±”‘Õ±Ğİ•¥¡Ğ(´AMQIQ€ƒŠP…¹¹Õ…°ÁÕ‰±¥ŒÙ…É¥…¹”ÍÑÉ…ÑÕ´(´AAMU€ƒŠP…¹¹Õ…°ÁÕ‰±¥ŒÙ…É¥…¹”AMT()MÕ‰É½ÕÀ•ÍÑ¥µ…Ñ•Ì…É”ÑÉ•…Ñ•…ÌÍÕÉÙ•ä‘½µ…¥¹ÌÉ…Ñ¡•ÈÑ¡…¸¥¹‘•Á•¹‘•¹ĞÍ¥µÁ±”É…¹‘½´Í…µÁ±•Ì¸½¹™¥‘•¹”¥¹Ñ•ÉÙ…±ÌÕÍ”Q…å±½Èµ±¥¹•…É¥é•Ù…É¥…¹•Ì…¹ÁÕ‰±¥Œµ‘•Í¥¸‘•É••Ì½˜™É••‘½´¸((ŒŒŒ5Õ±Ñ¥Á±”¥µÁÕÑ…Ñ¥½¸()Q¡”½É‘¥¹…ÉäM…µÁ±”‘Õ±Ğ™¥±”½¹Ñ…¥¹ÌÁ½Ù•ÉÑäµ•…ÍÕÉ•Ì‰…Í•½¸Ñ¡”€¨©™¥ÉÍĞ¨¨¥µÁÕÑ…Ñ¥½¸¸Q¡¥ÌÁÉ½©•Ğ¥¹ÍÑ•…ÕÍ•ÌÑ¡”Í•Á…É…Ñ”M…µÁ±”‘Õ±Ğ%¹½µ”™¥±”…¹É•Ñ…¥¹Ì…±°€¨¨ÄÀ¨¨¥µÁÕÑ…Ñ¥½¹Ì¸()%¹½µ”µ‘•Á•¹‘•¹ĞÍÕÉÙ•ä•ÍÑ¥µ…Ñ•Ì…É”½µÁÕÑ•¥¸•… ½µÁ±•Ñ•‘…Ñ…Í•Ğ…¹Á½½±•ÕÍ¥¹œİ¥Ñ¡¥¸´…¹‰•Ñİ••¸µ¥µÁÕÑ…Ñ¥½¸Ù…É¥…¹”İ¥Ñ Ñ¡”	…É¹…É“ŠMIÕ‰¥¸‘•É••Ìµ½˜µ™É••‘½´…‘©ÕÍÑµ•¹Ğ¸½ÈÁÉ•‘¥Ñ¥Ù”µ½‘•±¥¹œ°¥¹½µ”µ•¹¡…¹•µ½‘•±Ì…É”É•™¥ĞÍ•Á…É…Ñ•±ä…É½ÍÌ…±°€ÄÀ½µÁ±•Ñ•‘…Ñ…Í•ÑÌ°…¹Á•É™½Éµ…¹”Ù…É¥…Ñ¥½¸¥ÌÉ•Á½ÉÑ•…Ì„Í•¹Í¥Ñ¥Ù¥Ñä…¹…±åÍ¥Ì¸Q¡”€ÄÀ¥µÁÕÑ•É•½É‘Ì…É”¹•Ù•ÈÑÉ•…Ñ•…Ì€ÄÀ‘¥™™•É•¹ĞÉ•ÍÁ½¹‘•¹ÑÌ¸((ŒŒŒAÉ•‘¥Ñ¥Ù”µ½‘•±¥¹œ¥Ì¹½Ğ…ÕÍ…°¥¹™•É•¹”()Q¡”ÁÉ•‘¥Ñ¥Ù”µ½‘•±Ì…É”•Ù…±Õ…Ñ•…ÌÉ¥Í¬µÉ…¹­¥¹œÑ½½±Ì¸MÕÉÙ•äİ•¥¡ÑÌ…É”ÕÍ•…Ìµ½‘•°½•Ù…±Õ…Ñ¥½¸Í…µÁ±”İ•¥¡ÑÌ°‰ÕĞ±½¥ÍÑ¥Œ½•™™¥¥•¹ÑÌ…¹‰½½ÍÑ¥¹œ™•…ÑÕÉ”¥µÁ½ÉÑ…¹”…É”€¨©¹½Ğ¨¨ÁÉ•Í•¹Ñ•…ÌÍÕÉÙ•äµ…‘©ÕÍÑ•…ÕÍ…°•™™•ÑÌ½È™½Éµ…°Á½ÁÕ±…Ñ¥½¸¥¹™•É•¹”¸ÍÍ½¥…Ñ¥½¹Ìµ…äÉ•™±•Ğ½¹™½Õ¹‘¥¹œ°µ•…ÍÕÉ•µ•¹Ğ‘¥™™•É•¹•Ì°…¹ÍÕÉÙ•äµÉ•ÍÁ½¹Í”ÁÉ½•ÍÍ•Ì¸((ŒŒ…Ñ„Í½ÕÉ”…¹…ÑÑÉ¥‰ÕÑ¥½¸((¨©…Ñ„M½ÕÉ”è¨¨9…Ñ¥½¹…°•¹Ñ•È™½È!•…±Ñ MÑ…Ñ¥ÍÑ¥Ì°9…Ñ¥½¹…°!•…±Ñ %¹Ñ•ÉÙ¥•ÜMÕÉÙ•ä°€ÈÀÈÔ¸()=™™¥¥…°€ÈÀÈÔ‘½Õµ•¹Ñ…Ñ¥½¸…¹ÁÕ‰±¥ŒµÕÍ”‘½İ¹±½…‘Ì…É”…Ù…¥±…‰±”™É½´Ñ¡”m½9!L€ÈÀÈÔ9!%LÉ•±•…Í•t¡¡ÑÑÁÌè¼½İİÜ¹‘Œ¹½Ø½¹¡Ì½¹¡¥Ì½‘½Õµ•¹Ñ…Ñ¥½¸¼ÈÀÈÔµ¹¡¥Ì¹¡Ñµ°¤¸()Q¡”…¹…±åÍ¥Ì°¥¹Ñ•ÉÁÉ•Ñ…Ñ¥½¹Ì°…¹½¹±ÕÍ¥½¹Ì¥¸Ñ¡¥ÌÉ•Á½Í¥Ñ½Éä…É”Ñ¡½Í”½˜Ñ¡”…ÕÑ¡½È…¹…É”¹½Ğ…ÑÑÉ¥‰ÕÑ…‰±”Ñ¼Ñ¡”9…Ñ¥½¹…°•¹Ñ•È™½È!•…±Ñ MÑ…Ñ¥ÍÑ¥Ì°İ¡¥ ¥ÌÉ•ÍÁ½¹Í¥‰±”½¹±ä™½ÈÑ¡”¥¹¥Ñ¥…°‘…Ñ„¸()Q¡”É…Ü½9!L™¥±•Ì…É”¥¹Ñ•¹Ñ¥½¹…±±ä•á±Õ‘•™É½´¥Ğ¡¥ÍÑ½Éä¸M•”mQ}UM¹µ‘€¡Q}UM¹µ¤…¹m‘…Ñ„½É…Ü½I5¹µ‘€¡‘…Ñ„½É…Ü½I5¹µ¤™½ÈÍ½ÕÉ”°¡•­ÍÕ´°…¹‘…Ñ„µÕÍ”‘•Ñ…¥±Ì¸((ŒŒ1¥•¹Í”()=É¥¥¹…°½‘”…¹É•Á½Í¥Ñ½Éä‘½Õµ•¹Ñ…Ñ¥½¸…É”…Ù…¥±…‰±”Õ¹‘•ÈÑ¡”m5%P1¥•¹Í•t¡1%9M¤®*£ŠsŠSŠt)Q¡”5%P1¥•¹Í”‘½•Ì€¨©¹½Ğ¨¨É•±¥•¹Í”Ñ¡”9…Ñ¥½¹…°!•…±Ñ %¹Ñ•ÉÙ¥•ÜMÕÉÙ•äÍ½ÕÉ”‘…Ñ„ìÕÍ”½˜9!%L‘…Ñ„É•µ…¥¹ÌÍÕ‰©•ĞÑ¼Ñ¡”9!L‘…Ñ„µÕÍ”Ñ•ÉµÌ‘•ÍÉ¥‰•¥¸mQ}UM¹µ‘t¡Q}UM¹µ¤¸(