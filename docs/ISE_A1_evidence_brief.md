# ISE-A1 | Problem and Stakeholder Evidence Brief

This file directly answers the ISE-A1 "Problem and Stakeholder Evidence Brief"
requirements for the AI Trust Gate project. Where an item is already satisfied
by another file in this repository, this brief gives a short answer and links
to the full detail rather than duplicating it. Where a gap existed, the content
is written here.

**Objective addressed:** show that request-level risk screening in an LLM
workflow is a problem (the Trust Gate baseline classifier).

## 1. Investigate an authentic workflow using at least two independent source types

**Workflow:** `User request → AI/LLM receives request → Trust Gate evaluates
risk → Allow or Block → (if allowed) request continues to the AI system.`

Two independent source types were used, and every factual claim drawn from
them was checked against the original source (evidence_ledger.md):

1. **Peer-reviewed/academic:** Chao et al., "JailbreakBench: An Open
   Robustness Benchmark for Jailbreaking Large Language Models," NeurIPS 2024
   Datasets and Benchmarks Track (arXiv:2404.01318) — grounds the dataset,
   its 100/100 harmful/benign design, and its intent (testing robustness, not
   just plain classification).
2. **Authoritative institutional:** NIST AI Risk Management Framework
   (AI RMF 1.0, Jan 2023, DOI 10.6028/NIST.AI.100-1) and its Generative AI
   Profile (NIST-AI-600-1, Jul 2024, DOI 10.6028/NIST.AI.600-1) — grounds the
   framing of false negatives/false positives as distinct, named risk-management
   concerns rather than an invented distinction.

## 2. Identify users, affected parties, decision makers, constraints, and current responsibility

Expanded stakeholder map (extends the summary table in [doc.md](../doc.md)):

| Stakeholder | Role | Current responsibility today (baseline) | Key constraint on them |
|---|---|---|---|

| End users | User | Trust the LLM provider's built-in alignment; no visibility into risk screening | Cannot audit or configure how their requests are screened |

| AI/LLM developers | Decision maker (model behavior) | Own the model's refusal/alignment behavior; typically the *only* screening layer today | Must not break legitimate use cases while tightening refusals |

| Organizations deploying AI systems | Decision maker (deployment policy) | Decide whether/how to add any pre-model screening; currently often add none | Limited engineering time/budget for a semester-scale pilot; liability exposure |

| Safety/security evaluators | Affected party / auditor | Assess systems post-hoc, often without an independent, reproducible measurement layer to point to | Need reproducible metrics, not vendor-reported claims |

| Affected third parties | Affected party | Bear harm if a harmful request is allowed through and acted on; have no direct role in the workflow | No visibility or control over the gate at all |

## 3. Quantify pain, baseline, value measure, and failure cost

All numbers below are copied from this project's own executed results
(`results/metrics.json`, `random_state=42`):

- **Baseline (current process):** the workflow today has **no independent
  request-level screening step** — the "do nothing" baseline is equivalent to
  always allowing every request through to the LLM (0% of harmful requests
  intercepted by any layer outside the model itself).
- **Pain, quantified against that baseline:** on the 40-example held-out test
  split, the Trust Gate baseline **blocks 16 of 20 harmful test requests**
  (recall_harmful = 0.80, i.e. false-negative rate = 0.20) that the "do
  nothing" baseline would let all 20 through. This is a measured reduction in
  unscreened harmful traffic, not an estimate.
- **Value measures used:** recall (harmful), false-negative rate, precision,
  false-positive rate, F1-score, and reproducibility (fixed seed, frozen split,
  checked-in metrics) — see [doc.md](../doc.md#problem-statement) and
  [README.md](../README.md#results-from-actual-execution).
- **Failure cost (asymmetric, not equated):**
  - *False negative cost* (harmful request allowed, FNR = 0.20 measured here):
    a harmful request reaches the downstream AI system undetected — the more
    severe failure mode for a safety gate.
  - *False positive cost* (benign request blocked, FPR = 7/20 = 0.35 measured
    here): a legitimate user is wrongly refused — a usability/trust cost, not
    a safety incident, but repeated false positives risk users abandoning or
    bypassing the gate.
  - These two costs are not claimed to be numerically equal; NIST's AI RMF
    Generative AI Profile explicitly treats such asymmetric risks as requiring
    separate management, which is why both rates are reported separately
    rather than collapsed into accuracy alone.

## 4. Compare a software intervention with at least one non-software alternative

Summarized here; full discussion in
[doc.md](../doc.md#non-software-alternative). Software (the Trust Gate) is
scalable and produces repeatable, auditable metrics, but is limited to the
lexical signal it can see (see Section 6 below). Manual human review can apply
contextual judgment the classifier cannot, but does not scale to high request
volumes and introduces latency and reviewer inconsistency. Neither is
presented as universally superior — the choice depends on request volume,
latency budget, and acceptable review cost.

## 5. Bound semester scope, non-goals, assumptions, and stop conditions

**Semester scope (in-scope for this prototype):**
- A single, simple, reproducible baseline classifier (TF-IDF + Logistic
  Regression) evaluated on the full official 200-example JBB-Behaviors set.
- One fixed, documented train/test split (seed 42, 80/20, stratified) plus a
  training-only cross-validation diagnostic.
- Documentation of the workflow, stakeholders, evidence, scenarios,
  assumptions, and at least one real failure case.

**Out of scope (explicitly, see also [doc.md](../doc.md#non-goals)):** large
or local LLMs, deep learning architectures, web applications, databases,
Docker/cloud infrastructure, multi-turn conversation context, production
deployment, or any claim of complete jailbreak prevention.

**Assumptions:** see [doc.md](../doc.md#unresolved-assumptions).

**Stop conditions (new — when to stop trusting/extending this baseline
approach as-is):**
1. **Stop condition observed in this project:** 5-fold cross-validation on the
   training split alone averages only 0.538 accuracy (±0.041) — close to the
   0.50 chance level for this balanced binary task (see Section 7, Negative
   results). This is itself a stop condition: it means a purely lexical
   (TF-IDF) baseline should **not** be extended toward production use, scaled
   up, or trusted beyond this teaching/research context without first adding
   semantic/contextual modeling — which is out of this scope.
2. If false-negative rate on any held-out re-evaluation exceeds the
   false-negative rate of simply blocking everything containing no request at
   all (i.e., the gate performs worse than a trivial rule), the classifier
   approach should be stopped and reconsidered.
3. If the dataset or task changes such that the benign/harmful classes are no
   longer topically paired (i.e., real production traffic), this baseline's
   reported metrics should not be assumed to transfer — a fresh evaluation is
   required before reuse.

## 6. Negative results / evidence that challenges the claim

This project ran two rounds of model search, both using cross-validation **on
the training split only**, to avoid tuning against the fixed test set:

**Round 1 (reported previously):** TF-IDF n-gram range, sublinear scaling,
char n-grams, Logistic Regression vs. linear SVM, regularization strength,
5-fold stratified CV.

| Search axis | Result |
|---|---|
| Best cross-validated config found | TF-IDF unigrams, `sublinear_tf=True`, Logistic Regression `C=0.3` |
| Mean CV accuracy (training split, 5-fold) | 0.538 (± 0.041) |
| Accuracy on the single fixed 80/20 test split (`seed=42`) with that config | 0.725 |

**Round 2 (wider search, for reliability):** to check whether Round 1's
result was a fluke of the model family rather than a property of the feature
space, a broader search was run with **5-fold x 10-repeat** stratified CV
(50 scored folds per config instead of 5, to shrink the noise in the estimate)
across Multinomial/Complement Naive Bayes, `SGDClassifier` (hinge and
log-loss), a soft-voting ensemble of Logistic Regression + Naive Bayes +
calibrated linear SVM, and a Random Forest, all on the same TF-IDF features:

| Model family | Mean CV accuracy (± std, 50 folds) |
|---|---|
| TF-IDF + Random Forest | 0.558 (± 0.085) |
| TF-IDF + Logistic Regression (`C=0.1`) | 0.541 (± 0.082) |
| TF-IDF + SGD (hinge) | 0.537 (± 0.079) |
| TF-IDF + Logistic Regression (`C=0.3`, the config used) | 0.535 (± 0.082) |
| TF-IDF + soft-voting ensemble (LR+NB+SVM) | 0.472 (± 0.070) |
| CountVec / TF-IDF + Naive Bayes (all alphas tried) | 0.42 – 0.49 |

**This gap and this wider search are disclosed deliberately.** Every
config's cross-validation mean sits inside the same noisy band (roughly
0.47–0.56, with standard deviations of 0.07–0.09), while the single fixed
test split scores notably higher (0.725) for the chosen config. Reporting
only the 0.725 number without this context would overstate how well this
baseline actually generalizes. The wider search confirms Round 1 was not a
model-choice artifact: no model family tried — including a non-linear
Random Forest and a voting ensemble — reliably clears the ~0.55 ceiling, and
the differences between configs are within each other's standard deviation
(i.e., not a statistically meaningful improvement). Per the assignment's own
constraint to retain "failed runs, negative results, and evidence that
challenges the claim," this project reports all of these numbers rather than
only the favorable one, and keeps the originally specified TF-IDF + Logistic
Regression architecture (not the marginally-higher-but-noisier Random Forest)
since the assignment explicitly calls for "a simple and reproducible binary
text classifier: TF-IDF → Logistic Regression," not open-ended model
selection. This project does **not** further tune hyperparameters against
the fixed test set to chase a higher headline accuracy — doing so would be a
form of overfitting/fabrication, which the assignment explicitly prohibits
("do not fabricate ... results"). Combined with JailbreakBench's own design
intent (the benign set is deliberately thematically paired with the harmful
set to stress-test defenses), this is strong, repeated evidence that a
lexical/bag-of-words baseline has a real, measured ceiling well below 80%
robust accuracy on this benchmark, and that reaching materially higher
accuracy would require either semantic/contextual models (excluded by this
project's scope) or a larger, non-paired dataset.

## 7. AI-use and source-verification log

| Task | Advice given | Accepted / Rejected | Why | Verification |
|---|---|---|---|---|
| Scaffold `src/prepare_data.py`, `src/train.py`, `src/evaluate.py` | Generated the TF-IDF + Logistic Regression pipeline and file structure | Accepted (with edits) | Code matched the assignment's scikit-learn-only, no-deep-learning constraint | Executed `python src/prepare_data.py` and `python src/train.py`; confirmed outputs in `results/` |
| Hyperparameter optimization to raise accuracy | Proposed word n-grams, char n-grams, TF-IDF scaling variants, SVM, Naive Bayes, feature unions | Partially accepted, partially rejected | Accepted the CV methodology (search on train split only); rejected any approach that would have required peeking at/tuning against the fixed test set to inflate the reported number, and rejected deep-learning/embedding suggestions as out of scope | Re-ran the chosen config's cross-validation and single test-split evaluation via the terminal; |
| Draft documentation (README, doc.md, evidence ledger, failure analysis, this brief) | Drafted prose and tables | Accepted after review | Content cross-checked against `results/metrics.json`/`predictions.csv` and against the original JailbreakBench/NIST sources | Numbers manually diffed against `results/metrics.json`; source claims checked against the original webpages, not accepted as given by the assistant |
| Retrieve JBB-Behaviors dataset content when direct `curl` to huggingface.co failed in the sandbox | Fetched dataset content via an allowed web-fetch tool and reconstructed the CSVs | Accepted, disclosed | No other network path to the official dataset was available in this environment | Row counts (100/100), column schema, and spot-checked row content were verified against the JailbreakBench GitHub README's description of the dataset |

No claim in this repository is sourced from Copilot/ChatGPT output alone —
every factual claim traces to the original dataset files, the original
paper/repository, the original NIST pages, or this project's own executed
code/results.

## Deliverables checklist

| Deliverable | Location |
|---|---|
| Evidence ledger | [evidence_ledger.md](evidence_ledger.md) |
| Problem/value statement | [doc.md](../doc.md#problem-statement) |
| Stakeholder/workflow map | [doc.md](../doc.md#stakeholder-map) + Section 2 above |
| Normal, boundary, and abuse scenarios | [doc.md](../doc.md#scenarios) |
| AI-use and source-verification log | Section 7 above |
| Negative results / failed runs | Section 6 above + [failure_analysis.md](failure_analysis.md) |

## Common requirements checklist

- **Executable top-to-bottom; frozen seeds/dependencies/data/environment:**
  `pip install -r requirements.txt && python src/prepare_data.py && python
  src/train.py`; `random_state=42` throughout; `requirements.txt` pins
  minimum versions; raw data files are checked into `data/raw/`.
- **Retain failed runs, negative results, evidence that challenges the
  claim:** Section 6 above; [failure_analysis.md](failure_analysis.md).
- **Disclose AI tool/task, accepted/rejected advice, independent
  verification:** Section 7 above.
- **No fabricated data, sources, experiments, users, execution, or results;
  no credentials/personal data:** all metrics are generated by executing the
  code in this repository; the dataset is the public JBB-Behaviors benchmark;
  no API keys, credentials, or personal data appear anywhere in this
  repository.
