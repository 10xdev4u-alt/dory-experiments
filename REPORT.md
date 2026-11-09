# Dory 20h Lap — Final Research Report

> **Report status:** Template / placeholder draft  
> **Lap identifier:** `[LAP_ID]`  
> **Report date:** `[YYYY-MM-DD]`  
> **Authors:** `[AUTHOR_LIST]`  
> **Reviewers:** `[REVIEWER_LIST]`  
> **Version:** `[v0.1]`

---

## 1. Executive Summary

This report documents the outcomes, observations, and decisions from the Dory 20-hour research lap. The lap evaluated loop-count scaling, long-context feasibility, low-precision inference, routing by difficulty, loop regression behavior, distillation, adversarial robustness, quantization with model souping, and verifier design. The goal was to produce evidence-backed guidance for the next development cycle rather than to optimize a single benchmark in isolation.

**Primary objective:** `[STATE PRIMARY OBJECTIVE]`

**Secondary objectives:**
- `[OBJECTIVE_A]`
- `[OBJECTIVE_B]`
- `[OBJECTIVE_C]`

**Headline outcomes:**

| Area | Status | Confidence | Key takeaway |
|---|---|---:|---|
| E1 Loop-count scaling | `[PASS / PARTIAL / FAIL / INCONCLUSIVE]` | `[LOW/MED/HIGH]` | `[ONE-LINE TAKEAWAY]` |
| E2 256K context feasibility | `[STATUS]` | `[CONF]` | `[TAKEAWAY]` |
| E3 NVFP4 reality | `[STATUS]` | `[CONF]` | `[TAKEAWAY]` |
| E4 Difficulty router | `[STATUS]` | `[CONF]` | `[TAKEAWAY]` |
| E5 Loop regression taxonomy | `[STATUS]` | `[CONF]` | `[TAKEAWAY]` |
| E6 Distillation | `[STATUS]` | `[CONF]` | `[TAKEAWAY]` |
| E7 Adversarial | `[STATUS]` | `[CONF]` | `[TAKEAWAY]` |
| E8 Quant+soup | `[STATUS]` | `[CONF]` | `[TAKEAWAY]` |
| E9 Verifier | `[STATUS]` | `[CONF]` | `[TAKEAWAY]` |

**Top recommendations:**
1. `[RECOMMENDATION_1 — WHAT TO DO NEXT]`
2. `[RECOMMENDATION_2]`
3. `[RECOMMENDATION_3]`

**Gates triggered:** `[COUNT]` gates were triggered during the lap. See the Gates Triggered section below for timing, cause, owner, and resolution.

---

## 2. Hardware & Setup

### 2.1 Compute Environment

| Component | Specification | Notes |
|---|---|---|
| Host machine(s) | `[HOSTNAME / CLOUD INSTANCE]` | `[AVAILABILITY NOTES]` |
| GPU(s) | `[MODEL, COUNT, VRAM EACH]` | `[INTERCONNECT: NVLINK/PCIe]` |
| CPU | `[MODEL, CORES]` | `[RELEVANT FOR PREFILL/ROUTING]` |
| System memory | `[RAM SIZE]` | `[SWAP / NUMA NOTES]` |
| Storage | `[TYPE, SIZE, THROUGHPUT]` | `[DATASET/CHECKPOINT LOCATION]` |
| OS / kernel | `[DISTRO, VERSION]` | `[CONTAINERIZED?]` |
| Driver / CUDA / ROCm | `[VERSIONS]` | `[COMPATIBILITY NOTES]` |
| Runtime stack | `[FRAMEWORK, KERNEL LIBS, COMPILER]` | `[E.G., PyTorch, Triton, TensorRT-LLM]` |

### 2.2 Model & Data Configuration

| Item | Value | Rationale |
|---|---|---|
| Base model | `[MODEL_NAME / CHECKPOINT]` | `[WHY SELECTED]` |
| Parameter count | `[PARAMS]` | `[DENSE/MOE DETAILS]` |
| Context lengths tested | `[E.G., 32K, 128K, 256K]` | `[MEMORY/QUALITY TRADEOFF]` |
| Precision modes | `[BF16 / FP8 / NVFP4 / MIXED]` | `[KERNEL AVAILABILITY]` |
| Training/eval datasets | `[DATASET_NAMES]` | `[LICENSE/SPLIT NOTES]` |
| Prompt/task suites | `[SUITE_NAMES]` | `[DIFFICULTY COVERAGE]` |

### 2.3 Experimental Controls

- **Baseline definition:** `[WHAT COUNTS AS THE BASELINE RUN]`
- **Random seeds:** `[SEED POLICY]`
- **Budget cap:** `[GPU-HOURS / WALL-CLOCK / TOKEN BUDGET]`
- **Stop conditions:** `[QUALITY FLOOR, COST CEILING, INSTABILITY RULE]`
- **Logging:** `[METRICS, TRACES, ARTIFACT LOCATIONS]`

---

## Gates Triggered

This section records every gate that fired during the 20-hour lap. Gates are decision points intended to prevent sunk-cost continuation, unsafe deployment assumptions, or misleading conclusions.

| Gate ID | Time (UTC) | Experiment | Trigger condition | Severity | Owner | Action taken | Outcome | Follow-up |
|---|---|---|---|---|---|---|---|---|
| `G-[###]` | `[HH:MM]` | `[E#]` | `[METRIC/THRESHOLD/EVENT]` | `[LOW/MED/HIGH/CRITICAL]` | `[NAME]` | `[PAUSE / PIVOT / FALLBACK / ESCALATE]` | `[RESOLVED / OPEN]` | `[NEXT STEP]` |
| `G-[###]` | `[HH:MM]` | `[E#]` | `[TRIGGER]` | `[SEV]` | `[OWNER]` | `[ACTION]` | `[OUTCOME]` | `[FOLLOW-UP]` |
| `G-[###]` | `[HH:MM]` | `[E#]` | `[TRIGGER]` | `[SEV]` | `[OWNER]` | `[ACTION]` | `[OUTCOME]` | `[FOLLOW-UP]` |

**Gate summary:**
- Total triggered: `[N]`
- Critical: `[N]`
- Pivots caused by gates: `[N]`
- Open at report time: `[N]`

---

## 3. E1 — Loop-Count Scaling Law

### Question
How does task quality, latency, and cost change as loop count increases, and is there a stable scaling relationship that generalizes across task families?

### Method
- Loop counts swept: `[E.G., 1, 2, 4, 8, 16, N]`
- Task families: `[REASONING / CODING / RETRIEVAL / TOOL USE]`
- Fixed factors: `[MODEL, CONTEXT, PROMPT TEMPLATE, DECODE SETTINGS]`
- Primary metrics: `[ACCURACY, PASS@K, LATENCY, TOKENS, $/TASK]`

### Results

| Loop count | Quality metric `[NAME]` | p50 latency (s) | p95 latency (s) | Tokens/task | Cost/task | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[BASELINE]` |
| 2 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[OBSERVATION]` |
| 4 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[OBSERVATION]` |
| 8 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[OBSERVATION]` |
| 16 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[DIMINISHING RETURNS?]` |

### Interpretation
`[Summarize whether gains are monotonic, saturating, task-dependent, or unstable. Note any knee point where marginal quality gain no longer justifies added latency/cost.]`

### Decision
- **Adopt:** `[RECOMMENDED LOOP RANGE]`
- **Avoid:** `[ANTI-PATTERN]`
- **Open question:** `[WHAT REMAINS UNPROVEN]`

---

## 4. E2 — 256K Context Feasibility

### Question
Is 256K context operationally feasible under target quality, latency, memory, and reliability constraints?

### Method
- Context lengths compared: `[32K / 64K / 128K / 256K]`
- Workloads: `[LONG DOC QA / MULTI-FILE REASONING / NEEDLE TASKS / AGENT TRACE]`
- Measurements: `[VRAM PEAK, TTFT, TOKENS/SEC, OOM RATE, QUALITY]`

### Results

| Context | Peak VRAM (GB) | TTFT (s) | Decode tok/s | Quality `[METRIC]` | OOM/failure rate | Feasible? |
|---:|---:|---:|---:|---:|---:|---|
| 32K | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[Y/N]` |
| 64K | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[Y/N]` |
| 128K | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[Y/N]` |
| 256K | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[Y/N]` |

### Interpretation
`[Describe memory bottlenecks, attention/kernel behavior, cache pressure, and where quality actually benefits from longer context versus where retrieval or chunking is sufficient.]`

### Decision
- **Production stance:** `[FEASIBLE / FEASIBLE WITH GUARDS / NOT FEASIBLE NOW]`
- **Required guards:** `[MAX CONTEXT, ROUTING RULE, MEMORY HEADROOM]`

---

## 5. E3 — NVFP4 Reality

### Question
What are the practical gains and risks of NVFP4 in this stack, compared with higher-precision baselines?

### Method
- Baselines: `[BF16 / FP8 / OTHER]`
- Candidate: `[NVFP4 CONFIGURATION]`
- Evaluation: `[QUALITY DELTA, THROUGHPUT, MEMORY, KERNEL STABILITY, CALIBRATION NEEDS]`

### Results

| Mode | Quality `[METRIC]` | Throughput | Memory footprint | Stability | Deployment readiness | Notes |
|---|---:|---:|---:|---|---|---|
| BF16 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[HIGH/MED/LOW]` | `[BASELINE]` |
| FP8 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[LEVEL]` | `[OBSERVATION]` |
| NVFP4 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[LEVEL]` | `[OBSERVATION]` |

### Interpretation
`[State whether NVFP4 is a real win, a narrow win, or an operational distraction. Call out any accuracy cliffs, calibration sensitivity, unsupported operators, or benchmark-only gains.]`

### Decision
- **Recommendation:** `[ADOPT / PILOT / DEFER / REJECT]`
- **Conditions:** `[WORKLOADS OR HARDWARE WHERE IT MAKES SENSE]`

---

## 6. E4 — Difficulty Router

### Question
Can a difficulty-aware router reduce cost or latency while preserving acceptable quality by selecting loop count, context, or model path per task?

### Method
- Routing inputs: `[PROMPT FEATURES / CLASSIFIER / HEURISTICS / CONFIDENCE]`
- Routing outputs: `[LOOP COUNT / MODEL / CONTEXT / TOOL PATH]`
- Comparator: `[STATIC POLICY BASELINE]`

### Results

| Policy | Avg quality | Avg latency | Avg cost | Misroute rate | Escalation rate | Net assessment |
|---|---:|---:|---:|---:|---:|---|
| Static baseline | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[N/A]` | `[N/A]` | `[BASELINE]` |
| Router v1 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[ASSESSMENT]` |
| Router v2 | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[ASSESSMENT]` |

### Interpretation
`[Explain where routing helped, where it overthought easy tasks, and where hard tasks were under-served. Include failure modes such as confidence miscalibration or distribution shift.]`

### Decision
- **Ship status:** `[EXPERIMENTAL / GUARDED PILOT / NOT READY]`
- **Required fallback:** `[DEFAULT PATH WHEN ROUTER IS UNCERTAIN]`

---

## 7. E5 — Loop Regression Taxonomy

### Question
What failure patterns appear when looping makes outputs worse, and how should they be classified for debugging and mitigation?

### Method
- Corpus analyzed: `[RUN IDS / TRACE COUNT / TASK TYPES]`
- Labeling approach: `[MANUAL REVIEW / HEURISTICS / MIXED]`
- Regression definition: `[QUALITY DROP, FACTUAL DRIFT, FORMAT LOSS, COST BLOWUP]`

### Taxonomy

| Class | Description | Typical signal | Frequency | Severity | Mitigation |
|---|---|---|---|---:|---|
| `[T1: Drift]` | `[LATER LOOPS MOVE AWAY FROM CORRECT ANSWER]` | `[SIGNAL]` | `[N/%]` | `[SEV]` | `[MITIGATION]` |
| `[T2: Over-correction]` | `[MODEL FIXES NON-ISSUES]` | `[SIGNAL]` | `[N/%]` | `[SEV]` | `[MITIGATION]` |
| `[T3: Verbosity/cost spiral]` | `[MORE LOOPS WITHOUT QUALITY GAIN]` | `[SIGNAL]` | `[N/%]` | `[SEV]` | `[MITIGATION]` |
| `[T4: Format degradation]` | `[STRUCTURE OR CONSTRAINTS LOST]` | `[SIGNAL]` | `[N/%]` | `[SEV]` | `[MITIGATION]` |
| `[T5: Other]` | `[DESCRIPTION]` | `[SIGNAL]` | `[N/%]` | `[SEV]` | `[MITIGATION]` |

### Interpretation
`[Identify dominant classes and whether they correlate with task difficulty, prompt ambiguity, loop count, or verifier weakness.]`

### Decision
- **Highest-priority fix:** `[TAXONOMY CLASS + INTERVENTION]`
- **Instrumentation needed:** `[TRACE FIELDS / LABELS / ALERTS]`

---

## 8. E6 — Distillation

### Question
Can distillation preserve the useful behavior of the more expensive looped or routed setup in a cheaper student?

### Method
- Teacher setup: `[MODEL/POLICY]`
- Student setup: `[MODEL SIZE / ARCHITECTURE / TRAINING RECIPE]`
- Distillation target: `[FINAL ANSWERS / RATIONALES / TRACES / ROUTER DECISIONS]`
- Evaluation: `[QUALITY RETENTION, COST, LATENCY, GENERALIZATION]`

### Results

| Model/policy | Quality `[METRIC]` | Retention vs teacher | Latency | Cost | Notes |
|---|---:|---:|---:|---:|---|
| Teacher | `[VALUE]` | `100%` | `[VALUE]` | `[VALUE]` | `[REFERENCE]` |
| Student A | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[OBSERVATION]` |
| Student B | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[OBSERVATION]` |

### Interpretation
`[Discuss what behaviors transferred well, what was lost, and whether distillation reduced dependence on expensive loops or merely copied surface patterns.]`

### Decision
- **Viability:** `[PROMISING / MIXED / NOT WORTH IT NOW]`
- **Next step:** `[DATA, TARGETS, OR ARCHITECTURE CHANGE]`

---

## 9. E7 — Adversarial

### Question
How robust are the selected configurations to adversarial prompts, misleading context, or stress conditions relevant to deployment?

### Method
- Threats tested: `[PROMPT INJECTION / CONTRADICTORY CONTEXT / EDGE CASES / RESOURCE STRESS]`
- Configurations: `[BASELINE, LOOPED, ROUTED, QUANTIZED, VERIFIED]`
- Metrics: `[ATTACK SUCCESS RATE, QUALITY UNDER STRESS, RECOVERY RATE]`

### Results

| Configuration | Attack/Stress type | Success or pass rate | Failure mode | Severity | Notes |
|---|---|---:|---|---|---|
| `[CONFIG]` | `[TYPE]` | `[VALUE]` | `[MODE]` | `[SEV]` | `[NOTE]` |
| `[CONFIG]` | `[TYPE]` | `[VALUE]` | `[MODE]` | `[SEV]` | `[NOTE]` |
| `[CONFIG]` | `[TYPE]` | `[VALUE]` | `[MODE]` | `[SEV]` | `[NOTE]` |

### Interpretation
`[Highlight whether added loops/verifiers improve robustness or create new attack surfaces. Distinguish expected hard cases from unacceptable regressions.]`

### Decision
- **Risk posture:** `[ACCEPTABLE / NEEDS MITIGATION / BLOCKING]`
- **Required mitigations:** `[FILTERS, VERIFIER RULES, PROMPT GUARDS, MONITORING]`

---

## 10. E8 — Quant+Soup

### Question
Does combining quantization with model souping (or checkpoint merging) recover quality lost to compression without erasing efficiency gains?

### Method
- Quantization modes: `[MODE_LIST]`
- Soup/merge candidates: `[CHECKPOINTS / WEIGHTS / MERGE METHOD]`
- Comparisons: `[UNQUANTIZED, QUANTIZED, QUANTIZED+SOUP]`

### Results

| Configuration | Quality `[METRIC]` | Delta vs full precision | Throughput/memory gain | Stability | Notes |
|---|---:|---:|---:|---|---|
| Full precision | `[VALUE]` | `0.0` | `[BASELINE]` | `[VALUE]` | `[REFERENCE]` |
| Quantized | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[OBSERVATION]` |
| Quantized + soup | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[OBSERVATION]` |

### Interpretation
`[Assess whether souping meaningfully closes the gap or adds variance/operational complexity. Note reproducibility and sensitivity to merge choices.]`

### Decision
- **Recommendation:** `[USE / EXPERIMENT FURTHER / DROP]`
- **Best-known recipe:** `[PRECISE PLACEHOLDER FOR FINAL RECIPE]`

---

## 11. E9 — Verifier

### Question
Can a verifier improve final answer quality or catch loop regressions at acceptable compute cost?

### Method
- Verifier type: `[RULE-BASED / MODEL-BASED / HYBRID]`
- Verification target: `[FACTS, FORMAT, CODE EXECUTION, CONSTRAINTS]`
- Integration point: `[PER LOOP / FINAL ONLY / ROUTER-GATED]`

### Results

| Setup | Precision | Recall / catch rate | False positive rate | Added latency | Added cost | Net value |
|---|---:|---:|---:|---:|---:|---|
| No verifier | `[N/A]` | `[BASELINE MISS RATE]` | `[N/A]` | `0` | `0` | `[BASELINE]` |
| Verifier A | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[ASSESSMENT]` |
| Verifier B | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[ASSESSMENT]` |

### Interpretation
`[Describe which error classes the verifier catches, which it misses, and whether false positives trigger harmful extra loops or user-visible delays.]`

### Decision
- **Deployment mode:** `[OFF / FINAL-ONLY / SELECTIVE / ALWAYS-ON]`
- **Guardrail:** `[WHEN TO TRUST OR OVERRIDE VERIFIER]`

---

## 12. Cross-Cutting Findings

### Patterns Observed Across Experiments
- `[PATTERN_1 — E.G., QUALITY GAINS ARE TASK-FAMILY DEPENDENT]`
- `[PATTERN_2 — E.G., MEMORY, NOT RAW THROUGHPUT, IS THE MAIN 256K LIMITER]`
- `[PATTERN_3 — E.G., ROUTING AND VERIFICATION INTERACT: BAD ROUTES CREATE BAD VERIFIER LOAD]`

### Tradeoff Matrix

| Lever | Quality impact | Cost impact | Latency impact | Operational complexity | Confidence |
|---|---|---|---|---|---|
| More loops | `[+ / 0 / - / MIXED]` | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[CONF]` |
| Longer context | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[CONF]` |
| Lower precision | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[CONF]` |
| Difficulty routing | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[CONF]` |
| Verification | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[IMPACT]` | `[CONF]` |

### Risks to Carry Forward
- `[RISK_1]`
- `[RISK_2]`
- `[RISK_3]`

---

## 13. Negative Results & Pivots

Negative results are first-class outputs of this lap. Record them clearly to avoid repeating dead ends.

| Item | Original hypothesis | What happened | Evidence | Pivot / decision | Time cost |
|---|---|---|---|---|---|
| `[NR-1]` | `[EXPECTED OUTCOME]` | `[ACTUAL OUTCOME]` | `[RUNS / METRICS / LOGS]` | `[STOP / CHANGE SCOPE / REFRAME]` | `[HOURS]` |
| `[NR-2]` | `[HYPOTHESIS]` | `[RESULT]` | `[EVIDENCE]` | `[PIVOT]` | `[HOURS]` |
| `[NR-3]` | `[HYPOTHESIS]` | `[RESULT]` | `[EVIDENCE]` | `[PIVOT]` | `[HOURS]` |

**Notable pivots:**
1. `[PIVOT_1 — WHAT CHANGED AND WHY]`
2. `[PIVOT_2]`

**What we would not repeat:** `[SHORT LIST OF APPROACHES THAT ARE NOT WORTH RETRYING WITHOUT A MATERIAL CHANGE IN ASSUMPTIONS]`

---

## 14. Conclusions

### What We Learned
`[Synthesize the strongest conclusions supported by evidence. Separate robust findings from tentative observations.]`

### Recommended Next Actions

| Priority | Action | Owner | Expected impact | Effort | Dependencies |
|---|---|---|---|---|---|
| P0 | `[ACTION]` | `[OWNER]` | `[IMPACT]` | `[S/M/L]` | `[DEPENDENCY]` |
| P1 | `[ACTION]` | `[OWNER]` | `[IMPACT]` | `[S/M/L]` | `[DEPENDENCY]` |
| P2 | `[ACTION]` | `[OWNER]` | `[IMPACT]` | `[S/M/L]` | `[DEPENDENCY]` |

### Final Position
- **Safe to build on now:** `[LIST]`
- **Promising but unproven:** `[LIST]`
- **Do not pursue without new evidence:** `[LIST]`

---

## 15. Reproducibility

### Artifacts
| Artifact | Location | Identifier / hash | Notes |
|---|---|---|---|
| Report source | `[PATH/URL]` | `[HASH/COMMIT]` | `[TEMPLATE OR FINAL]` |
| Model checkpoints | `[PATH/URL]` | `[IDS]` | `[ACCESS NOTES]` |
| Datasets / task suites | `[PATH/URL]` | `[VERSION]` | `[LICENSE/REDACTION]` |
| Run logs / traces | `[PATH/URL]` | `[RUN RANGE]` | `[RETENTION POLICY]` |
| Config snapshots | `[PATH/URL]` | `[HASH/COMMIT]` | `[ENV PARITY]` |

### Minimal Reproduction Checklist
- [ ] Confirm hardware and driver/runtime versions match Section 2.
- [ ] Restore model, dataset, and config identifiers listed above.
- [ ] Re-run baseline before any experimental variant.
- [ ] Use the same seed policy and budget/stop conditions.
- [ ] Capture metrics with the same schema and aggregation rules.
- [ ] Compare against the baseline table, not memory or anecdote.

### Known Reproducibility Gaps
- `[GAP_1 — E.G., NONDETERMINISTIC KERNEL OR MISSING TRACE]`
- `[GAP_2]`
- `[GAP_3]`

---

## 16. Appendix

### A. Metric Definitions
| Metric | Definition | Direction | Aggregation |
|---|---|---|---|
| `[METRIC]` | `[DEFINITION]` | `[HIGHER/LOWER IS BETTER]` | `[MEAN / MEDIAN / p95]` |
| `[METRIC]` | `[DEFINITION]` | `[DIRECTION]` | `[AGGREGATION]` |

### B. Run Index
| Run ID | Experiment | Config summary | Start (UTC) | End (UTC) | Status | Artifact link |
|---|---|---|---|---|---|---|
| `[RUN_ID]` | `[E#]` | `[SUMMARY]` | `[TIME]` | `[TIME]` | `[OK/FAILED/PARTIAL]` | `[LINK]` |
| `[RUN_ID]` | `[E#]` | `[SUMMARY]` | `[TIME]` | `[TIME]` | `[STATUS]` | `[LINK]` |

### C. Open Questions
1. `[OPEN_QUESTION_1]`
2. `[OPEN_QUESTION_2]`
3. `[OPEN_QUESTION_3]`

### D. Change Log
| Version | Date | Author | Summary of changes |
|---|---|---|---|
| `[v0.1]` | `[YYYY-MM-DD]` | `[NAME]` | `[INITIAL TEMPLATE / DRAFT / FINAL REVISION]` |

---

*End of report template.*