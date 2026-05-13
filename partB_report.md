# Part B: Classical Cox Proportional Hazards Model for Mortgage Prepayment

---

## 1. Setup and Model Specification

### 1.1 Event Definition and Data

The analysis targets **prepayment** of 30-year fixed-rate Freddie Mac single-family mortgages. A loan is defined as having prepaid when it receives a zero-balance code of "01" (full voluntary payoff). Loans that terminate through default (zb_codes 03, 06, 09, 15, or 180+ days delinquent) or other causes are treated as **right-censored** at their last observation month for the purpose of the prepayment model — that is, we observe only that they had not yet prepaid by that point. This is the standard single-event survival analysis treatment; a competing-risks extension would jointly model both outcomes.

The duration variable `event_time_months` measures the number of months from first payment date to prepayment or censoring. Loans with zero or missing duration are excluded. The training sample spans **vintages 2006–2022**, with up to 2,000,000 loans drawn by random sample (seed 42) to remain computationally tractable for the partial-likelihood optimizer. Vintages 2024–2025 are held out for out-of-sample evaluation (2023 Q3–Q4 are absent from the upstream data release).

### 1.2 Feature Set

The static model uses 15 origination-time covariates. Each encodes a distinct channel through which borrower and loan characteristics affect the propensity to refinance or pay off early:

| Covariate | Type | Economic mechanism |
|---|---|---|
| `fico` | Continuous | Credit score determines access to refinancing; higher score lowers refi barriers |
| `ltv` | Continuous | Loan-to-value; high LTV means less equity, reducing refi eligibility and incentive |
| `dti` | Continuous | Debt-to-income; high DTI makes qualifying for a new loan harder |
| `orig_rate` | Continuous | The coupon rate at origination; determines the size of any future refi incentive |
| `orig_upb` | Continuous | Original unpaid balance; larger loans generate larger dollar savings per basis-point rate drop |
| `loan_purpose` | Categorical (P/C/N) | Purchase, cash-out refi, or no-cash refi; different mobility and equity profiles |
| `channel` | Categorical (R/B/T/C) | Retail, broker, correspondent, or other; reflects borrower sophistication and market access |
| `n_borrowers` | Continuous | One or two borrowers; two-borrower households have more financial flexibility |
| `n_units` | Continuous | Number of units (1–4); investment/multi-unit properties prepay differently than owner-occupied |
| `mi_pct` | Continuous | Mortgage insurance percentage; high MI can constrain refi |
| `occupancy` | Categorical (P/I/S) | Primary, investor, or second home |
| `prop_type` | Categorical (SF/CO/PU/MH/CP) | Property type; manufactured housing (MH) has a very thin refi market |
| `first_time_homebuyer` | Binary (Y/N) | First-time buyers tend to be less financially mobile |
| `state` | Categorical (54 levels) | Captures regional rate environments, housing markets, and regulatory costs |
| `vintage_year` | Continuous | Origination year; absorbs the macro rate-cycle era in which the loan was born |

### 1.3 Preprocessing

Categorical variables are one-hot encoded with `drop_first=True`, which selects implicit reference categories (cash-out refi, broker channel, investor occupancy, condo property, no for first-time buyer). Any row missing a value in any feature or the duration column is dropped; this affected a small fraction of the sample. An L2 penalty (`penalizer=0.01`) is applied to the partial likelihood to stabilize the 50+ state dummy coefficients while barely affecting the dominant numeric features at n=2M.

---

## 2. (i) Cox Proportional Hazards Model — Estimation

### 2.1 Model Form

The Cox proportional hazards model specifies the instantaneous hazard for loan $i$ at time $t$ as:

$$\lambda(t \mid X_i) = \lambda_0(t) \cdot \exp(\beta^\top X_i)$$

where $\lambda_0(t)$ is a completely unspecified baseline hazard, $X_i$ is the vector of covariates for loan $i$, and $\beta$ is the coefficient vector estimated by maximizing the **partial likelihood**:

$$\mathcal{L}(\beta) = \prod_{i: \delta_i=1} \frac{\exp(\beta^\top X_i)}{\sum_{j \in \mathcal{R}(t_i)} \exp(\beta^\top X_j)}$$

where $\delta_i = 1$ indicates a prepayment event and $\mathcal{R}(t_i)$ is the risk set at time $t_i$. The key advantage of this semiparametric formulation is that no distributional assumption is placed on $\lambda_0(t)$; the baseline hazard is estimated nonparametrically from the data via the **Breslow estimator** after $\beta$ is obtained.

### 2.2 Breslow Baseline Hazard and Survival Function

Figure 1 shows the estimated baseline hazard $\hat{\lambda}_0(t)$ and corresponding baseline survival $\hat{S}_0(t) = \exp(-\hat{H}_0(t))$.

**Figure 1** — *Breslow baseline hazard h₀(t) (left) and baseline survival S₀(t) (right). Static origination feature model, 2006–2022 vintages.*

The baseline hazard exhibits a clear **seasoning arc**: starting from near zero at origination, it rises steeply through the first two to three years as loans become financially eligible to refinance (enough time to build equity, reach break-even on closing costs), plateaus at approximately 0.020 per month in the 50–100 month range, and then declines toward zero beyond month 120 as the surviving population becomes increasingly composed of loans that have not yet found a refi opportunity. The high-frequency oscillations at long durations reflect thinning of the risk set as fewer loans remain under observation. The baseline survival function — corresponding to a loan at all reference-category covariate values — reaches approximately 0.40 by month 100 and 0.04 by month 220, confirming that the majority of loans at the reference profile eventually prepay within 18 years.

The baseline profile corresponds to: a cash-out refi loan, brokered, investor occupancy, condo property type, not first-time homebuyer, with all numeric covariates at zero. The practical survival curve for any given loan is $\hat{S}_0(t)^{\exp(\hat{\beta}^\top X_i)}$.

---

## 3. (ii) Coefficient Interpretation and Hazard Ratios

All coefficients are reported as hazard ratios (HR = $e^\beta$): a value above 1 means higher hazard (faster prepayment), below 1 means lower hazard. All results below are from the static model; Figure 2 shows the full forest plot.

**Figure 2** — *Forest plot of estimated hazard ratios with 95% confidence intervals. Covariates sorted by HR ascending. Red = HR > 1 (increases hazard); blue = HR < 1 (decreases hazard).*

### 3.1 Dominant Numeric Predictors

**`orig_rate`** is the single strongest predictor by an enormous margin: HR = **1.650** per percentage point (z = 171, p ≈ 0). A loan originated at 7% vs. 6% has a 65% higher instantaneous prepayment hazard, all else equal. This encodes the *latent refinancing incentive*: a higher coupon rate means a larger spread over the eventual prevailing market rate, making the loan more likely to be in the money for a refinance at any future point. The strength of this coefficient — by far the largest in the model — reflects that the timing of prepayment is primarily driven by the rate environment rather than borrower characteristics.

**`orig_upb`** (original balance) has HR ≈ 1.000017 per dollar (z = 93, p ≈ 0). While the per-dollar effect is tiny, the economic magnitude across the loan balance distribution is meaningful: a $100,000 increase in balance raises the hazard by approximately 19%. This is consistent with the fixed-cost nature of refinancing — closing costs of several thousand dollars are more easily justified on a $500,000 loan than a $100,000 loan, so larger-balance borrowers have a lower effective rate threshold for refinancing.

**`vintage_year`** has HR = **1.054** per year (z = 78, p ≈ 0). Later-vintage loans prepaid more rapidly, on average, reflecting the secular decline in mortgage rates from 2006 through 2021. A 2019-vintage loan, for instance, was originated just before the rate environment that produced the 2020–2021 refinancing wave, and its expected prepayment hazard is approximately 1.054^13 ≈ 2.0× that of a 2006-vintage loan.

**`fico`** has HR = **1.0025** per credit score point (z = 55, p ≈ 0). A borrower with a 780 score has approximately 1.0025^(780-660) = 1.35× the prepayment hazard of an otherwise identical 660-score borrower. This positive sign is economically correct and often surprising to first-time readers: higher-FICO borrowers are *more* likely to prepay, not less, because they have better access to refinancing programs, lower offered rates, and greater financial sophistication. FICO here captures refi access rather than distress.

**`ltv`** has HR = **0.9981** per LTV point (z = −11.9, p ≈ 10⁻³³). Higher LTV reduces prepayment hazard: a loan at 95 LTV has approximately 0.9981^(95-70) = 0.955× the hazard of a 70 LTV loan. High-LTV borrowers have less equity — either they cannot refinance without mortgage insurance or they are constrained from doing so by current underwriting standards.

**`dti`** has HR = **0.9982** per DTI point (z = −9.6, p ≈ 10⁻²²). Higher debt-to-income ratios reduce prepayment hazard, consistent with borrowers with high existing debt loads finding it harder to qualify for a new mortgage under debt-service capacity tests.

**`n_units`** has HR = **0.797** per unit (z = −23.4, p ≈ 10⁻¹²⁰). Investment properties with more units have markedly lower prepayment hazards. Multi-unit landlords are less likely to refinance opportunistically and more sensitive to the administrative costs of refinancing a rental property.

**`n_borrowers`** has HR = **1.092** (z = 22.5, p ≈ 10⁻¹¹²). Two-borrower households prepay slightly faster, likely because dual-income households have more financial flexibility and are better able to absorb the cash flow demands of refinancing.

**`mi_pct`**: HR ≈ 1.0000, p = 0.84. Mortgage insurance percentage is **not statistically significant** conditional on the other covariates. This suggests its effect is fully absorbed by LTV and other features in the model.

### 3.2 Categorical Predictors

**Occupancy type** produces the largest categorical effect. Primary-residence loans (vs. investor, the reference) have HR = **1.427** (z = 41.9, p ≈ 0) — primary homeowners refinance 43% more aggressively than investors. Second-home loans have HR = **1.216** (z = 16.0). This ordering is economically natural: primary homeowners have the most to gain from reducing their monthly payment burden and have the smoothest path through refi underwriting.

**Origination channel** is highly significant. Correspondent-originated loans (`channel_T`) have HR = **0.692** (z = −39.7, p ≈ 0) — 31% lower hazard than broker-originated loans (the reference category). Retail (`channel_R`) has HR = **0.907** (z = −15.3). The correspondent channel serves relatively less financially sophisticated borrowers who are less likely to opportunistically monitor rates and initiate a refinance. Broker-originated borrowers are already accustomed to working with mortgage intermediaries and have a lower barrier to refinancing.

**Property type**: Manufactured housing (`prop_type_MH`) has HR = **0.465** (z = −21.2), reflecting a severely limited refinancing market — MH loans are difficult to refinance due to property classification issues and limited secondary market demand. Planned unit developments (`prop_type_PU`) have HR = **1.117** (z = 13.8). Single-family (`prop_type_SF`) has HR = **0.980** (z = −2.8), a modest and borderline-significant discount vs. the condo reference.

**First-time homebuyer**: HR = **0.898** (z = −17.7). First-time buyers are roughly 10% less likely to prepay at any point in time, consistent with lower financial mobility, less familiarity with refinancing, and potentially tighter budget constraints.

**Loan purpose**: Purchase loans (`loan_purpose_P`) have HR = **1.129** (z = 21.4) relative to cash-out refis (reference). No-cash-out refis (`loan_purpose_N`) have HR = **1.067** (z = 11.7). Purchase borrowers move, change jobs, and upgrade homes more frequently than those who have already refinanced, producing higher prepayment through turnover in addition to rate-driven refis.

**State effects**: There is substantial geographic dispersion. New York (HR = **0.658**, z = −22.8) and Puerto Rico (HR = **0.307**, z = −11.8) show the largest negative state effects, likely reflecting high transaction costs, complex regulatory environments, and slower labor market mobility. Utah (HR = **1.374**, z = 16.3), Wisconsin (HR = **1.347**, z = 15.7), and Colorado (HR = **1.272**, z = 13.8) show the highest prepayment hazards. These patterns broadly track regional housing market dynamism and household income growth.

### 3.3 Sign Consistency Summary

| Covariate | Predicted sign | Observed sign | Consistent? |
|---|:---:|:---:|:---:|
| fico | + | + (HR=1.0025) | Yes |
| ltv | − | − (HR=0.998) | Yes |
| dti | − | − (HR=0.998) | Yes |
| orig_rate | + | + (HR=1.650) | Yes |
| orig_upb | + | + (HR≈1.000+) | Yes |
| n_borrowers | + | + (HR=1.092) | Yes |
| n_units | − | − (HR=0.797) | Yes |
| mi_pct | − | ≈0, n.s. | Neutral |
| occupancy_P (primary) | + | + (HR=1.427) | Yes |
| channel_T (correspondent) | − | − (HR=0.692) | Yes |
| prop_type_MH (mfg. housing) | − | − (HR=0.465) | Yes |
| first_time_homebuyer_Y | − | − (HR=0.898) | Yes |
| loan_purpose_P (purchase) | + | + (HR=1.129) | Yes |
| vintage_year | + | + (HR=1.054) | Yes |

All statistically significant covariates have signs consistent with economic theory.

---

## 4. (iii) Testing the Proportional Hazards Assumption

The Cox model's key identifying assumption is that the hazard ratio between any two covariate profiles is **constant over time** — the proportional hazards (PH) assumption. If this fails, the estimated $\hat{\beta}$ represents a time-averaged effect that may obscure important dynamics.

### 4.1 Schoenfeld Residuals Test

The standard test uses Schoenfeld residuals, which are the difference between the observed covariate value for an event and its expected value under the fitted model at that event time. Under the null hypothesis of proportional hazards, the Schoenfeld residuals for each covariate should be uncorrelated with time. The test statistic is distributed as $\chi^2_1$ for each covariate.

**Figure 3** — *Schoenfeld residuals test: −log₁₀(p) for each covariate. Red bars indicate violation of the PH assumption at the 5% level. The dashed line marks p = 0.05 (−log₁₀ = 1.30).*

The results show widespread violations, which is expected for mortgage data:

**Strongest violations:**

| Covariate | χ² | p-value | Economic interpretation |
|---|---:|---|---|
| `orig_rate` | 1325.5 | ≈10⁻²⁹⁰ | Refi incentive is inherently time-varying as market rates evolve |
| `channel_T` | 467.9 | ≈10⁻¹⁰⁴ | Correspondent-channel effect strengthens as loans season |
| `first_time_homebuyer_Y` | 362.9 | ≈10⁻⁸¹ | First-timers gain financial sophistication over time; catch up on refi rates |
| `vintage_year` | 347.7 | ≈10⁻⁷⁷ | Cohort effects change as the macro rate environment evolves |
| `occupancy_P` | 242.7 | ≈10⁻⁵⁴ | Primary-occupancy advantage varies with rate cycles |
| `channel_R` | 111.3 | ≈10⁻²⁶ | Retail-channel effect time-varying |
| `loan_purpose_N` | 104.1 | ≈10⁻²⁴ | No-cash refi loans show changing prepayment pattern over time |
| `mi_pct` | 94.6 | ≈10⁻²² | Despite near-zero average coefficient, MI effect is time-varying |
| `n_borrowers` | 63.3 | ≈10⁻¹⁵ | Two-borrower advantage changes with household lifecycle |
| `ltv` | 41.3 | ≈10⁻¹⁰ | High-LTV loans pay down over time, eventually becoming eligible to refi |

**Covariates satisfying PH (p > 0.05):**

`dti` (χ² = 0.42, p = 0.515) satisfies PH most cleanly, as do several low-population state dummies (state_NE, state_KY, state_WY, state_DC, and others).

The `orig_rate` violation deserves particular attention. With a test statistic of 1325 and p ≈ 10⁻²⁹⁰, it is by far the most severe PH violation. This is economically intuitive: the coefficient on `orig_rate` in a static Cox model captures the *average* effect of the origination rate over the loan's entire lifetime. In reality, the effect is near zero in the early months (when market rates may still be above the loan rate, so no refi incentive exists) and then spikes dramatically when rates fall below the origination rate. A time-varying covariate specification — where the rate incentive is computed each month as `orig_rate − current_market_rate` — would be the correct treatment. This is deferred to the advanced extensions.

### 4.2 Log-Log Survival Plots

A complementary visual test plots $\log(-\log \hat{S}(t))$ vs. $\log(t)$ for groups defined by binning a covariate. Under PH, the curves for different groups should be parallel vertical shifts of one another. Crossing or converging curves indicate time-varying effects.

**Figure 4** — *Log-log survival plots for FICO score bins (<660, 660–700, 700–740, 740–780, 780+).*

For **FICO**, the five bins produce nearly parallel curves across the full time range from log(t) = 0 to 5 (i.e., months 1–220). The uniform vertical separation, with higher FICO bins consistently above lower bins (higher log(-log S) = higher cumulative hazard = faster prepayment), is consistent with proportional hazards. There is mild convergence at very long durations where the lowest FICO bin (<660) catches up slightly — these borrowers who survive to 15+ years eventually find a way to refinance or sell — but the deviation is small. The visual test supports the Schoenfeld result that FICO's PH assumption is relatively well-satisfied.

**Figure 5** — *Log-log survival plots for LTV bins (≤60, 60–80, 80–90, 90–95, 95+).*

For **LTV**, the picture is more mixed. The low-LTV group (≤60) sits consistently above the moderate-LTV groups (60–80, 80–90) in a roughly parallel arrangement for most of the observed range, consistent with PH. However, the high-LTV group (95+) deviates markedly: it starts far below all other groups at early durations (log(t) < 2, roughly the first year) and then rises steeply, nearly converging with the 80–90 group by log(t) = 5. This reflects the dynamic of high-LTV loans paying down the principal over time — an initially deeply underwater loan gradually accumulates sufficient equity to become eligible for refinancing. This dynamic LTV process is not captured by the static origination LTV, and the convergence pattern confirms a mild PH violation. The Schoenfeld test found χ² = 41.3 for LTV (p ≈ 10⁻¹⁰).

**Figure 6** — *Log-log survival plots for origination rate bins (<3.5%, 3.5–4.5%, 4.5–5.5%, 5.5–6.5%, 6.5%+).*

For **orig_rate**, the log-log plot shows the clearest PH violation of the three. The five rate bins show a complex crossing pattern: at early durations (log(t) < 2), the high-rate bins (6.5%+, 5.5–6.5%) sit well above the low-rate bins, reflecting the early-seasoning effect common to all loans. From roughly log(t) = 2 to 3.5 (months 7–33), the groups partially reorder, with the highest-rate bins maintaining their lead. Beyond log(t) ≈ 4 (month 55), the lowest-rate bin (<3.5%) falls further behind while the mid-rate bins converge, reflecting the 2020–2021 refinancing wave that disproportionately eliminated high-rate loans from the risk set — exactly the non-proportional dynamic that the Schoenfeld test flagged so dramatically. The divergence and eventual crossing of these curves confirm that the hazard ratio between, say, a 6.5%+ loan and a <3.5% loan is far from constant over time.

---

## 5. (iv) Macro-Covariate Extension

### 5.1 Motivation and Feature Construction

The static model captures *at-origination* characteristics but ignores the macroeconomic environment that determines *when* the latent refi incentive gets triggered. The macro-extended model adds five features derived from FRED monthly data, each joined to the loan at its `first_payment_date` (the month the loan begins amortizing):

**`rate_incentive_orig`** = `orig_rate − MORTGAGE30US`. This is the rate spread between the loan coupon and the prevailing 30-year mortgage rate at origination. A positive value means the loan was already above the current market rate at the moment of first payment — i.e., it was born with an immediate refinancing incentive. A negative value means market rates had already moved up, reducing the initial incentive.

**`mort_treasury_spread`** = `MORTGAGE30US − GS10`. The spread between 30-year mortgage rates and 10-year Treasury yields measures the risk premium embedded in mortgage rates. A wider spread indicates tighter mortgage credit conditions relative to risk-free rates.

**`UNRATE`**: The national unemployment rate at origination month. High unemployment reduces household income and makes qualifying for a new mortgage harder.

**`log_hpi`** = $\log$(CSUSHPISA). Log of the Case-Shiller national house price index at origination. High HPI at origination means the borrower bought at a peak; if prices subsequently correct, the loan can go underwater, eliminating the refinancing option.

**`log_cpi`** = $\log$(CPIAUCSL). Log of the consumer price index at origination. High CPI reflects inflationary conditions that erode real purchasing power and may reflect periods of monetary tightening, both of which reduce the likelihood of future rate declines sufficient to trigger refinancing.

Raw levels of MORTGAGE30US, CSUSHPISA, and CPIAUCSL are dropped after constructing the derived features. GS10 and UNRATE enter directly.

### 5.2 Results

**Figure 7** — *Macro-model Breslow baseline hazard h₀(t) (left) and baseline survival S₀(t) (right).*

The macro model's baseline hazard follows the same seasoning arc as the static model — rising from near zero, plateauing around months 50–100, then declining — but the peak level is lower (≈0.018 vs. ≈0.020). This is expected: the macro covariates explain some of the rate-cycle variation that was previously absorbed by the baseline, slightly pulling the baseline downward. The baseline survival function is similarly shifted, reflecting that the macro-enriched baseline profile corresponds to a slightly different risk level.

The macro covariates all achieve statistical significance with large z-statistics, confirming that the macroeconomic environment at origination has substantial and independent explanatory power for prepayment timing.

**`rate_incentive_orig`**: HR = **1.331** (z = 30.0, p ≈ 10⁻¹⁹⁸). Each additional percentage point of rate incentive at origination raises the prepayment hazard by 33%. This is the most directly interpretable macro covariate: it measures how far out of the money the existing loan is at the moment of first payment, and it has the expected strong positive effect. Loans that began with above-market rates will never experience a period of being below-market and will be permanently motivated to refinance whenever rates stabilize.

**`mort_treasury_spread`**: HR = **1.392** (z = 30.4, p ≈ 10⁻²⁰³). Wider mortgage-Treasury spreads at origination are associated with higher prepayment hazard. This may initially appear counterintuitive — a wider spread implies more expensive mortgage credit — but reflects a compositional effect: wide spreads tend to occur during periods of strong housing demand and loose credit availability (e.g., the mid-2000s), which also correlate with higher borrower mobility and greater willingness to refinance. Alternatively, a wide spread at origination means there is more room for that spread to compress, providing a larger refi incentive when market conditions normalize.

**`GS10`**: HR = **1.265** (z = 26.6, p ≈ 10⁻¹⁵⁵). Higher 10-year Treasury yields at origination are positively associated with eventual prepayment. Loans originated in high-rate environments have larger rate incentives to refinance when rates ultimately fall, and the 10-year yield directly determines the baseline level of the mortgage rate cycle.

**`UNRATE`**: HR = **0.968** (z = −25.9, p ≈ 10⁻¹⁴⁷). Higher unemployment at origination reduces prepayment hazard by approximately 3.2% per percentage-point of unemployment. This is the expected sign: in weak labor markets, households face income uncertainty, have more difficulty qualifying for a new mortgage, and are less likely to move for employment opportunities. A loan originated at 8% unemployment is approximately 0.968^(8−4) ≈ 0.88× as likely to prepay at any given moment as one originated at 4% unemployment.

**`log_hpi`**: HR = **0.272** (z = −56.6, p ≈ 0). This is the largest macro coefficient in magnitude. Loans originated when house prices were high face a much lower prepayment hazard. A one-unit increase in log HPI — roughly a doubling of house prices — is associated with a 73% reduction in prepayment hazard. This reflects the downside risk of high-price originations: if prices subsequently fall (as they did nationally after 2006), borrowers find themselves underwater and unable to refinance. Even if prices merely stagnate, high-HPI loans were often originated at high valuations with little room for appreciation to build equity. The 2006-vintage loans, originated at the peak of the housing bubble with the highest HPI values in the sample, subsequently suffered severe prepayment suppression through the underwater channel.

**`log_cpi`**: HR = **0.053** (z = −35.3, p ≈ 10⁻²⁷⁴). The CPI effect is the most extreme in the model: a one-unit increase in log CPI — a very large shift — is associated with a 95% reduction in prepayment hazard. Loans originated in high-inflation environments are associated with persistently lower prepayment rates. This may reflect that high-CPI periods tend to accompany rising interest rate regimes (the Federal Reserve tightening cycle), in which future mortgage rates remain elevated and the refi incentive never materializes. CPI and HPI are highly collinear over the sample period, so the point estimates should be interpreted together rather than individually.

### 5.3 Structural Change vs. Static Model

The addition of macro covariates changes two key coefficients in interpretively important ways:

**`orig_rate`** coefficient falls from 0.501 to **0.318** (HR: 1.650 → 1.374). The rate incentive variable (`rate_incentive_orig` = orig_rate − current market rate) directly captures a portion of the mechanism that orig_rate was proxying in the static model. When the market rate at origination is controlled for, the residual effect of the coupon rate is smaller — the macro variable explains the part of orig_rate's effect that comes from the contemporaneous rate environment.

**`vintage_year`** coefficient rises from 0.052 to **0.143** (HR: 1.054 → 1.154). With the macro environment at origination now partially accounted for by HPI, CPI, and rate variables, vintage year absorbs more of the residual cohort-level variation — including secular trends in underwriting standards, borrower behavior, and loan composition that are not captured by the five macro series.

### 5.4 PH Test on the Macro Model

**Figure 8** — *Schoenfeld residuals test for the macro model.*

The PH violations identified in the static model persist in the macro-extended model. Most notably, `orig_rate` remains the dominant violator — the macro covariates are measured *at origination* and cannot resolve the time-varying nature of the refi incentive, which depends on the *current* market rate at each point in the loan's life. The fundamental PH limitation of the static Cox framework applies equally to both model specifications.

---

## 6. Discussion and Limitations

**Time-invariant macro covariates.** Both models join macro data at the origination month — a static snapshot. The economically correct specification for `rate_incentive_orig` would update it monthly as `orig_rate − MORTGAGE30US(t)`, capturing the live in-the-money status of each loan at each point in time. This time-varying covariate formulation eliminates the need to use origination-time proxies and would substantially reduce the PH violations observed above. It requires the monthly loan panel rather than the origination snapshot, and belongs to the advanced extension in Part E.

**PH violations.** The Schoenfeld test finds that at least 10 covariates formally violate proportionality. The Cox model estimated here provides **lifecycle-average** estimates of each covariate's effect. For practical prediction this is often sufficient, but it means the model cannot accurately characterize the prepayment hazard at specific loan ages — particularly the early seasoning period and the late-life plateau. A stratified Cox model (stratifying on vintage_year to allow separate baselines per cohort) or a model with time-varying coefficients would be more structurally accurate.

**Competing risks.** Defaulted loans are censored at default for the purpose of this prepayment model. In a competing-risks framework using cause-specific hazards or subdistribution hazards (Fine-Gray), the probability of prepayment would properly account for the fact that default eliminates the possibility of eventual prepayment. The effect is most significant for high-LTV, low-FICO, high-DTI borrowers where default risk is non-trivial.

**Static LTV.** Origination LTV is used throughout, but the economically relevant quantity for refinancing eligibility is the *current* LTV, which changes as the loan amortizes and house prices evolve. The log-log plot for LTV (Figure 5) clearly shows the convergence that results from this dynamic: high-LTV loans that survive to long durations have often paid down substantially, but the model is unable to capture this because it uses only the origination value.
