# Semantic Actuation Contagion Control

Research materials for *Semantic Actuation Contagion Control in Temporal Robot-Agent Networks*, by Weian Guo, Yao Xiao and Wuzhao Li.

This collection provides simulation and analysis code, calibration prompts, historical keyword dictionaries, fitted transfer matrices, initialization rules, complete simulation parameters and hardware configurations. It is a selected reproducibility collection, not the full internal project archive. Manuscript PDFs, private correspondence, access credentials, model weights and large intermediate step traces are not included.

## Contents

| Directory | Materials |
|---|---|
| `calibration` | Calibration input cases, historical dictionaries, fitted injection and forwarding matrices, observations and functional-analysis source |
| `prediction` | Heldout input cases, result tables and prediction-analysis source |
| `matrix` | Complete declared simulation configurations, statistical plan, trial-level analysis inputs and all paired comparisons |
| `joint_action` | All three matched-gradient action sets, their observations, configurations, action records, frozen source and standalone reconstruction |
| `hardware` | Deployment configuration, native receiver calibration, terminal trial counts and protected-controller source |

The legacy keyword dictionary documents the earlier estimator. Revised functional calibration uses executed operations rather than keyword matching. Model weights and software dependencies must be obtained from their original distributors.

## Download the complete selected collection

The repository browser contains the small source files, configuration files and guides. Larger observation tables, calibration input cases and retained action records are supplied in the [tagged release archive](https://github.com/ahshicr/semantic-actuation-contagion-control/releases/tag/v2026.10.08). Download `Selected_Research_Materials_v2026.10.08.zip` and extract it into a new directory before running the commands below. A source-only clone does not contain all reconstruction inputs.

`material_manifest.json` indexes the complete selected archive, not just the files displayed in the repository browser. Both use the same tagged source version.

## Quick verification

Use Python with NumPy, pandas, SciPy and threadpoolctl for statistical reconstruction. Recorded statistical package versions are in `requirements-statistics.txt`. PyTorch is needed only for the optional CPU trajectory replay.

```text
python verify_materials.py
python matrix/reconstruct_statistics.py --out fresh_matrix_statistics
python joint_action/reconstruct_joint_action.py --data joint_action --out fresh_action_statistics --statistics-only
```

Each output directory must be new. These commands do not access a model service, networked devices, SSH or serial ports. The first verifies supplied checksums. The other commands reconstruct the reported statistics from the supplied observations.

For full matched-action CPU trajectory replay, omit `--statistics-only` and choose another fresh output directory. It regenerates motion inputs from the retained configuration, checks every summary and action mask, and then reconstructs all forty contrasts. This is an offline replay, not a new experiment or an additional set of independent environments.

The archived generation modules are supplied for inspection and a separately planned fresh simulation. They should not be confused with the lightweight statistical entries above. Local-model generation and new hardware operation require separate runtime and deployment preparation.

## Interpretation and units

The main matrix contains 96,768 policy-trial rows, 8,960 policy-summary rows and 14,680 paired comparisons. Policy rows are not independent environments. The complete primary and secondary Holm families contain 2,000 and 12,680 contrasts respectively. All favorable, adverse and nonsignificant comparisons are retained.

The matched action-set study uses 96 existing environments in each of four scenarios, with 64 nodes per environment. Three policies produce 1,152 trajectories from 384 reused environments. They use the same nonlinear prediction and gradient calculation rule, a forecast horizon of five, common modeled action costs and the normalized total budget cap `B/N = 0.06`. This does not impose a separate cap of 0.06 on every node. Gradient values evolve along each policy's own closed-loop trajectory. All arms omit the path multiplier.

Forty two-sided Wilcoxon-Pratt contrasts form a separate complete Holm family. Mean intervals are pointwise paired whole-environment bootstrap intervals, not simultaneous intervals. The additional comparison plan was fixed before its outcomes were generated, with the earlier matrix results already known. No new independent environments are claimed.

Costs are expenditures computed from the declared action prices. They are not measured joules, power consumption or monetary device costs. Simulation scores, local terminal inference and safe serial replies are separate measurements.

## Provenance and privacy

`material_manifest.json` maps every selected material to its retained source and records both original and public SHA-256 hashes. Scientific numeric CSV observations are unchanged. Private addresses, user paths and connection details were removed from string metadata. Endpoint aliases are deterministic and do not reveal live access addresses. Public configuration checksums reflect those metadata edits.

The verification records describe tests on the recorded runtime. Statistical reconstruction does not imply identical token generation or floating-point trajectories on every machine. Intermediate motion snapshots can be regenerated and are not duplicated here.

## Citation and version

Use the tagged release for the exact version referenced in a manuscript, rather than a changing branch. Author and title metadata are provided in `CITATION.cff`. No publication acceptance, DOI or journal submission is asserted by this repository.

## Licensing

First-party software and software documentation are licensed under the [MIT License](LICENSE). Author-generated research data, prompts, observations and scientific configurations are licensed under [Creative Commons Attribution 4.0 International](DATA_LICENSE.md). Attribution is to Weian Guo, Yao Xiao and Wuzhao Li, using the title and tagged version recorded above.

These licenses govern reuse. They do not change experimental observations, mathematical claims or the correspondence between the selected materials and the manuscript. Third-party models and dependencies retain their own licenses and are not redistributed here.
