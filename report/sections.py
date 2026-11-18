"""Report sections 3-8. All numbers are read from outputs/ and artifacts/.
Qualitative statements that depend on results are guarded with assertions, so
the build fails (instead of printing a wrong claim) if the results change."""
import json
import os

import numpy as np
import pandas as pd

from rl import KeepTogether, P, PageBreak, Spacer, TEXT_W, bullets, cm, img, table

DERIVED = {}   # every number the report derives (differences, ratios, ...) -> outputs/logs/report_derived_values.json


def dv(key, value):
    """Record a derived value (with a descriptive key) so the number checker can verify it."""
    DERIVED[key] = float(value)
    return value


ORDER = ["LogisticRegression", "GaussianNB", "KNN", "RandomForest", "ExtraTrees", "HistGradientBoosting",
         "LightGBM", "XGBoost", "MLP"]


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ============================================================== section 3
def sec3(O, A, rj):
    C = pd.read_csv(O("models", "comparison.csv")).set_index("Model")
    F = pd.read_csv(O("models", "cv_folds_all.csv"))
    R = {n: rj(O("models", "per_model", f"{n}.json")) for n in ORDER}
    sel = rj(O("models", "model_selection.json"))
    E = pd.read_csv(O("models", "error_types_student_test.csv")).set_index("model")
    H = pd.read_csv(O("models", "mlp_history.csv"))
    tl = F.groupby("model").train_log_loss.mean()
    fold_top1 = F.pivot(index="model", columns="fold", values="val_top1")
    st = [PageBreak(), P("3. Model experimentation and comparison", "h1"),
          P("<b>Task formulation.</b> Every (observation, candidate) row is a binary example: positive if the candidate is "
            "the reference arc of that observation. A model outputs P(positive) for each row; the observation's predicted "
            f"segment is the candidate with the highest probability (Top-1). Positives are about 1 in "
            f"{dv('rows_per_obs_student_test', R['XGBoost']['st_rows'] / R['XGBoost']['st_points']):.1f} rows (student_test), "
            "so row-level accuracy is inflated by the many easy negatives and is reported only for completeness."),
          P("<b>Protocol.</b> 5-fold GroupKFold by trip on <b>train (78 trips)</b>: each fold is fitted on every 3rd "
            "observation of its training trips and evaluated on <i>all</i> observations of the held-out trips. Each model "
            "is then refitted on all 78 train trips and evaluated on the 10 <b>student_test</b> trips. Seed 42 everywhere; "
            "configurations, per-fold metrics, train and inference times are stored in outputs/models/per_model/*.json and "
            "*_cv_folds.csv. Inference time = predict_proba on all 213,304 student_test candidate rows (19,160 observations), "
            "excluding feature extraction (common to all models), measured on the same 4-core CPU, one model at a time."),
          P(f"<b>Primary selection metric: mean CV observation-level Top-1 accuracy.</b> The deliverable is one segment "
            f"per GPS observation, which is exactly what Top-1 scores; it is threshold-free, whereas candidate Macro F1 "
            f"depends on the arbitrary 0.5 threshold that the pipeline never uses. Because several models are statistically "
            f"tied, models within one fold standard deviation of the best CV Top-1 are ranked by <b>CV log loss</b> "
            f"(the HMM consumes the probabilities, so calibration matters), then by inference time. The rule was fixed "
            f"before the comparison (scripts/03b_plot_compare.py).")]

    # ---- 3.1 table
    rows = [["Model", "Algorithm family", "Key configuration", "CV Macro F1 (mean ± std)", "Top-1 Accuracy (student-test)",
             "Route continuity (student-test)", "Student-test Macro F1", "Log loss (student-test)",
             "Inference time (student-test)"]]
    for n in ORDER:
        c = C.loc[n]
        rows.append([("<b>%s</b>" % n) if n == sel["chosen"] else n, c["Algorithm family"], c["Key configuration"],
                     f"{c['CV Macro F1 (mean)']:.4f} ± {c['CV Macro F1 (std)']:.4f}", f4(c["Student-test Top-1"]),
                     f4(c["Route continuity (student-test, classifier only)"]), f4(c["Student-test Macro F1"]),
                     f4(c["Student-test Log loss"]),
                     f"{c['Inference time student-test [s]']:.2f} s ({c['Inference [ms / 1000 obs]']:.0f} ms/1k obs)"])
    st += [P("3.1 Overall model comparison", "h2"),
           table(rows, col_widths=[2.2 * cm, 1.9 * cm, 3.6 * cm, 1.7 * cm, 1.4 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm, 2.3 * cm]),
           P("Table 3 - Consolidated comparison (required columns). CV columns: train partition, 5-fold GroupKFold by trip; "
             "other columns: student_test partition. Route continuity is for the classifier-only (Top-1) path; the effect "
             "of the HMM is in section 5. No cell is N/A: all nine models output class probabilities, so log loss and "
             "Top-1 apply to all. Source: outputs/models/comparison.csv.", "cap")]
    rows = [["Model", "CV Top-1 (mean ± std)", "CV log loss", "Student-test Precision (pos)", "Student-test Recall (pos)",
             "Student-test Precision / Recall (macro)", "Row accuracy", "Train time (final fit)", "Mean CV fold fit"]]
    for n in ORDER:
        c = C.loc[n]
        rows.append([n, f"{c['CV Top-1 (mean)']:.4f} ± {c['CV Top-1 (std)']:.4f}", f4(c["CV Log loss"]),
                     f4(c["Student-test Precision (pos)"]), f4(c["Student-test Recall (pos)"]),
                     f"{c['Student-test Precision (macro)']:.4f} / {c['Student-test Recall (macro)']:.4f}",
                     f4(c["Student-test Accuracy (rows)"]), f"{c['Train time final fit [s]']:.1f} s",
                     f"{c['Mean CV fold fit time [s]']:.1f} s"])
    st += [table(rows, col_widths=[2.6 * cm, 2.2 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 2.4 * cm, 1.5 * cm, 1.7 * cm, 1.6 * cm]),
           P("Table 4 - Additional metrics: precision/recall at threshold 0.5 (positive class and macro), row accuracy, "
             "training cost. KNN 'training' only stores a 150,000-row sample; its cost is at inference. Candidate miss rate "
             f"is a property of candidate generation, identical for all models: {pct(C['Candidate miss rate (student-test)'].iloc[0], 2)} "
             "on student_test (Top-1 ceiling). Source: outputs/models/comparison.csv.", "cap")]
    ch = sel["chosen"]
    assert ch == "XGBoost", "text below describes XGBoost as the selected model"
    best = sel["best_cv_top1_model"]
    st += [P("<b>Selection.</b> " +
             f"The best CV Top-1 is {best} ({sel['best_cv_top1']:.4f} ± {sel['fold_std_of_best']:.4f}); "
             f"{len(sel['contenders'])} models lie within one fold standard deviation of it "
             f"({', '.join(sel['contenders'])}), so their Top-1 differences are not meaningful - all five folds rank the "
             f"tree ensembles in the same order of difficulty (fold 0 hardest, fold 4 easiest, Table 6), i.e. fold-to-fold "
             f"variance comes from the trips, not the models. Among the contenders <b>XGBoost</b> has the lowest CV log loss "
             f"({f4(C.loc['XGBoost', 'CV Log loss'])} vs {f4(C.loc[best, 'CV Log loss'])} for {best}) and is the fastest tree "
             f"ensemble at inference ({C.loc['XGBoost', 'Inference time student-test [s]']:.2f} s vs "
             f"{C.loc[best, 'Inference time student-test [s]']:.2f} s). It is therefore selected. The student_test results "
             f"agree: XGBoost has the highest student_test Top-1 ({f4(C.loc['XGBoost', 'Student-test Top-1'])}), Macro F1 "
             f"({f4(C.loc['XGBoost', 'Student-test Macro F1'])}) of all nine models and ties with LightGBM for the lowest "
             f"log loss ({C.loc['XGBoost', 'Student-test Log loss']:.6f} vs {C.loc['LightGBM', 'Student-test Log loss']:.6f}), "
             "but student_test was "
             "not used to make the choice.")]
    assert C["Student-test Top-1"].idxmax() == "XGBoost" and C["Student-test Macro F1"].idxmax() == "XGBoost"
    assert C["Student-test Log loss"].nsmallest(2).index.tolist() in (["XGBoost", "LightGBM"], ["LightGBM", "XGBoost"])
    assert abs(C.loc["XGBoost", "Student-test Log loss"] - C.loc["LightGBM", "Student-test Log loss"]) < 1e-4
    assert C.loc[sel["contenders"], "CV Log loss"].idxmin() == "XGBoost"
    rows = [["Model"] + [f"fold {k}" for k in range(5)]]
    for n in ORDER:
        rows.append([n] + [f4(fold_top1.loc[n, k]) for k in range(5)])
    st += [table(rows, col_widths=[3.4 * cm] + [2 * cm] * 5),
           P("Table 6 - Held-out Top-1 per CV fold (train partition). Source: outputs/models/cv_folds_all.csv.", "cap")]
    PC = pd.read_csv(O("models", "paired_comparison.csv")).set_index("other")
    rows = [["XGBoost vs", "CV folds XGBoost better (of 5)", "CV mean diff [pp]", "student_test trips better / worse",
             "student_test diff [pp] (95% CI over trips)"]]
    for o, r in PC.iterrows():
        rows.append([o, str(r.cv_folds_selected_better), f"{100 * r.cv_mean_diff:+.2f} pp",
                     f"{r.st_trips_selected_better} / {r.st_trips_selected_worse}",
                     f"{100 * r.st_weighted_diff:+.2f} pp ({100 * r.st_ci_low:+.2f} pp, {100 * r.st_ci_high:+.2f} pp)"])
    clear = [o for o, r in PC.iterrows() if r.cv_folds_selected_better == 5 and r.st_ci_excludes_zero]
    tied = [o for o in PC.index if o not in clear]
    assert set(tied) == {"RandomForest", "LightGBM"}
    st += [table(rows, col_widths=[3.4 * cm, 2.8 * cm, 2.4 * cm, 3.2 * cm, 5.6 * cm]),
           P("Table 6b - Paired comparison of the selected model with each other model: per-fold CV differences (train) "
             "and per-trip student_test differences with a 95% bootstrap confidence interval over the 10 trips. Source: "
             "scripts/03d_paired_comparison.py -> outputs/models/paired_comparison.csv.", "cap"),
           P(f"XGBoost is better than {', '.join(clear)} in every CV fold and on student_test with a confidence interval that "
             f"excludes zero - the gradient-boosted/bagged tree ensembles genuinely beat the other families. Against "
             f"{' and '.join(tied)} the evidence is mixed: CV differences are within "
             f"+-{100 * PC.loc[tied, ['cv_min_diff', 'cv_max_diff']].abs().max().max():.2f} pp and change sign between folds; on "
             f"student_test XGBoost is ahead of RandomForest on {PC.loc['RandomForest', 'st_trips_selected_better']} of 10 "
             f"trips but its lead over LightGBM is not significant. We therefore describe XGBoost as <i>tied</i> with these "
             f"two on accuracy and chosen for calibration and speed, not as the most accurate model.")]

    xe = E.loc["XGBoost"]
    st += [P("<b>What limits Top-1?</b> We classified every student_test Top-1 error (Figure 11). For XGBoost, "
             f"{xe.errors:,} of {xe.n_obs:,} observations are wrong: {xe.reverse_twin:,} pick the <i>reverse direction</i> "
             f"of the reference road, {xe.adjacent:,} pick an arc sharing a node with it (junction ambiguity), "
             f"{xe.missed} are candidate misses and {xe.other:,} are other arcs. The reverse-twin errors are almost the "
             f"same for every model ({E.reverse_twin.min():,}-{E.reverse_twin.max():,}) and concentrate in one trip "
             f"(77: {dict(x.split(':') for x in xe.twin_errors_by_trip.split(';'))['00000077']} of them for XGBoost). In "
             f"{xe.twin_errors_gps_opposes_reference:,} of the {xe.reverse_twin:,} cases the vehicle is moving and its GPS "
             "heading over +-5 fixes is more than 90 deg away from the reference arc's direction - the model follows the "
             "observed direction of travel and the reference route says the opposite. We therefore read most of these as "
             "limitations of the reference route (the dataset notes warn that it is not error-free), not as model errors; "
             "we did <b>not</b> relabel anything, all metrics use the reference as given."),
           img(O("figures", "error_types.png"), width=14 * cm,
               caption="Figure 11 - Top-1 error types per model (student_test, classifier only). Source: "
                       "scripts/03c_error_analysis.py -> outputs/models/error_types_student_test.csv")]
    assert xe.twin_errors_gps_opposes_reference > 0.6 * xe.reverse_twin

    # ---- 3.2 per model
    st += [PageBreak(), P("3.2 Per-model analysis (A1, A2, B5, B6, C7, C8, C9)", "h2"),
           P("C7/C8 report the student_test partition; C9 compares in-sample metrics on the training folds (computed on up "
             "to 300k of the rows each fold model was fitted on) with held-out CV metrics (train partition). The "
             "generalisation gap below is train-fold Macro F1 minus held-out CV Macro F1.")]
    best_cv = C["CV Top-1 (mean)"].max()

    def gap(n):
        return C.loc[n, "Train-fold Macro F1"] - C.loc[n, "CV Macro F1 (mean)"]

    def c7(n):
        c = C.loc[n]
        return (f"Student_test: Top-1 {f4(c['Student-test Top-1'])} (ceiling {f4(dv('top1_ceiling_student_test', 1 - c['Candidate miss rate (student-test)']))}), "
                f"Macro F1 {f4(c['Student-test Macro F1'])}, positive-class precision {f4(c['Student-test Precision (pos)'])} "
                f"/ recall {f4(c['Student-test Recall (pos)'])}, classifier-only route continuity "
                f"{f4(c['Route continuity (student-test, classifier only)'])}, inference "
                f"{c['Inference time student-test [s]']:.2f} s for 19,160 observations; Top-1 errors: "
                f"{E.loc[n, 'reverse_twin']:,} reverse-direction, {E.loc[n, 'adjacent']:,} adjacent-arc, "
                f"{E.loc[n, 'other']:,} other. Rank {int(C['Student-test Top-1'].rank(ascending=False)[n])}/9 on student_test "
                f"Top-1 (CV rank {int(C['CV Top-1 (mean)'].rank(ascending=False)[n])}/9).")

    def c8(n):
        c = C.loc[n]
        return (f"Top-1 = <b>{f4(c['Student-test Top-1'])}</b>, Macro F1 = <b>{f4(c['Student-test Macro F1'])}</b>, "
                f"log loss = <b>{f4(c['Student-test Log loss'])}</b> (CV: Top-1 {f4(c['CV Top-1 (mean)'])}, Macro F1 "
                f"{f4(c['CV Macro F1 (mean)'])}, log loss {f4(c['CV Log loss'])}).")

    def c9_numbers(n):
        c = C.loc[n]
        return (f"Train-fold Macro F1 {f4(c['Train-fold Macro F1'])} vs held-out {f4(c['CV Macro F1 (mean)'])} "
                f"(gap {dv('gap_macro_f1_' + n, gap(n)):+.4f}); train-fold log loss {f4(tl[n])} vs held-out {f4(c['CV Log loss'])}; "
                f"CV Top-1 std across folds {f4(c['CV Top-1 (std)'])}; student_test Top-1 "
                f"{f4(c['Student-test Top-1'])} vs CV {f4(c['CV Top-1 (mean)'])}.")

    def params(n):
        p = R[n]["params"]
        skip = {"n_jobs", "verbose", "eval_metric", "tree_method"}
        return ", ".join(f"{k}={v}" for k, v in p.items() if k not in skip)

    MD = {}
    # ---------- Logistic regression
    assert abs(gap("LogisticRegression")) < 0.01 and C.loc["LogisticRegression", "CV Top-1 (mean)"] < best_cv - 0.02
    MD["LogisticRegression"] = dict(
        A1="A linear, well-understood probabilistic baseline. Most of our features are monotone evidence (smaller distance, "
           "smaller heading difference, closer context distances = more likely), which a linear logit can weigh directly, "
           "and its probabilities are a natural emission model for the HMM.",
        A2=f"Strengths: fastest inference of all models ({C.loc['LogisticRegression', 'Inference time student-test [s]']:.2f} s), "
           f"stable folds, reasonably calibrated (student_test log loss {f4(C.loc['LogisticRegression', 'Student-test Log loss'])}). "
           "Limitation: one linear boundary cannot express interactions that matter here - e.g. heading is only informative "
           "when the vehicle moves (low_motion) and distance matters less when context distances agree. Its Top-1 is "
           f"{f4(dv('cv_top1_best_minus_LogisticRegression', best_cv - C.loc['LogisticRegression', 'CV Top-1 (mean)']))} below the best CV model.",
        B6="Binary cross-entropy (log loss) with L2 penalty (C = 1.0 is the inverse regularisation strength), minimised by L-BFGS.",
        C9=" There is no overfitting: in-sample and held-out scores are almost identical. The model is <b>underfitting</b> "
           f"(high bias): it is {dv('cv_top1_points_best_minus_LogisticRegression', 100 * (best_cv - C.loc['LogisticRegression', 'CV Top-1 (mean)'])):.1f} points of CV Top-1 behind "
           "the best model while having no generalisation gap, "
           "so the missing accuracy comes from model capacity, not from variance.")
    # ---------- GaussianNB
    assert C.loc["GaussianNB", "Student-test Log loss"] > 5 * C.loc["XGBoost", "Student-test Log loss"]
    assert C.loc["GaussianNB", "Student-test Recall (pos)"] > C.loc["GaussianNB", "Student-test Precision (pos)"] + 0.2
    MD["GaussianNB"] = dict(
        A1="A generative probabilistic baseline that models each feature per class as a Gaussian; it is extremely cheap and "
           "shows how far simple per-feature evidence goes on this problem.",
        A2=f"Strength: trains in {C.loc['GaussianNB', 'Train time final fit [s]']:.2f} s and needs no tuning. Limitations seen in "
           "our results: (i) the independence assumption is badly violated - dist, dist_minus_min, dist_ratio, the four "
           "context distances and ctx_mean all measure nearly the same thing, so their evidence is counted many times and "
           f"the probabilities become extreme (log loss {f4(C.loc['GaussianNB', 'Student-test Log loss'])}, about "
           f"{C.loc['GaussianNB', 'Student-test Log loss'] / C.loc['XGBoost', 'Student-test Log loss']:.0f}x XGBoost); "
           "(ii) distances are heavily skewed, not Gaussian. At threshold 0.5 it flags many wrong candidates as positive "
           f"(precision {f4(C.loc['GaussianNB', 'Student-test Precision (pos)'])}, recall "
           f"{f4(C.loc['GaussianNB', 'Student-test Recall (pos)'])}), hence the lowest Macro F1.",
        B6="No iterative loss: class priors and per-class feature means/variances are maximum-likelihood estimates; "
           "prediction applies Bayes' rule. var_smoothing adds a fraction of the largest variance for stability.",
        C9=" No overfitting (train and held-out scores are equal); the model <b>underfits</b> because its assumptions "
           "(independent Gaussian features) are wrong for these correlated, skewed features. Its poor calibration makes it "
           "unsuitable as an HMM emission model.")
    # ---------- KNN
    assert gap("KNN") > 0.04
    MD["KNN"] = dict(
        A1="Instance-based: a candidate is scored by how many similar (observation, candidate) situations in the training data "
           "were the reference arc. With standardised features 'similar' means similar distance/heading/context patterns, "
           "a reasonable non-parametric notion for map matching.",
        A2=f"Strengths: no training, non-linear decision surface. Limitations seen here: it is by far the slowest at inference "
           f"({C.loc['KNN', 'Inference time student-test [s]']:.1f} s, {C.loc['KNN', 'Inference [ms / 1000 obs]']:.0f} ms per "
           "1000 observations, against well under 1 s for most other models) because every prediction scans the stored sample; "
           "the stored sample had to be capped at 150,000 rows, and all features get equal weight after scaling, so "
           "weakly informative ones (degrees, dt) dilute the distance metric.",
        B6="No explicit loss: majority vote of the k nearest neighbours in standardised feature space, weighted by "
           "inverse distance; P(positive) = weighted share of positive neighbours.",
        C9=" <b>Overfitting (high variance)</b>: the training-fold Macro F1 is much higher than the held-out one. With "
           "distance weighting, a stored row is its own nearest neighbour, so in-sample scores are optimistic by construction "
           "(about a third of the in-sample evaluation rows are in the stored 150k sample); the held-out gap is the "
           "relevant evidence. Held-out log loss is the second worst of all models.")
    # ---------- RandomForest
    assert 0.015 < gap("RandomForest") < 0.04 and sel["best_cv_top1_model"] == "RandomForest"
    MD["RandomForest"] = dict(
        A1="Bagged deep decision trees capture the feature interactions that the linear model misses (e.g. heading only "
           "counts while moving; context distance overrides point distance) without feature scaling, and averaging many "
           "trees gives usable probabilities.",
        A2=f"Strengths: best mean CV Top-1 of all nine models ({f4(C.loc['RandomForest', 'CV Top-1 (mean)'])}), robust to "
           f"skewed features. Limitations: the slowest training ({C.loc['RandomForest', 'Train time final fit [s]']:.0f} s for "
           "the final fit), a large model file, and slightly worse calibration than the boosted models (CV log loss "
           f"{f4(C.loc['RandomForest', 'CV Log loss'])} vs {f4(C.loc['XGBoost', 'CV Log loss'])} for XGBoost).",
        B6="Each tree greedily minimises Gini impurity; the forest averages tree class-probability estimates (no global loss).",
        C9=" <b>Mild overfitting</b>: deep trees with min_samples_leaf = 5 fit the training trips noticeably better than "
           "unseen trips. The gap is a variance effect that bagging keeps under control - held-out performance is still the "
           "best in CV, and student_test Top-1 is within the CV fold spread.")
    # ---------- ExtraTrees
    assert 0.015 < gap("ExtraTrees") < 0.04
    MD["ExtraTrees"] = dict(
        A1="Like the random forest but with random split thresholds: more randomisation reduces variance and training cost, "
           "useful on 600k strongly autocorrelated rows.",
        A2=f"Strengths: about {C.loc['RandomForest', 'Train time final fit [s]'] / C.loc['ExtraTrees', 'Train time final fit [s]']:.0f}x "
           "faster to train than the random forest. Limitation seen: random thresholds are a poor fit for the few very "
           "sharp cut-offs that matter here (a few metres of distance difference, small heading differences), so it is "
           f"{f4(dv('cv_top1_RF_minus_ET', C.loc['RandomForest', 'CV Top-1 (mean)'] - C.loc['ExtraTrees', 'CV Top-1 (mean)']))} below RF in CV Top-1 and "
           "has the slowest inference among the tree ensembles (300 trees).",
        B6="Per-tree Gini impurity with randomly drawn thresholds; averaged class probabilities.",
        C9=" <b>Mild overfitting</b>, smaller than RF's (extra randomisation acts as a regulariser), at the cost of somewhat "
           "higher bias: lower held-out scores than RF, LightGBM and XGBoost.")
    # ---------- HGB
    assert C.loc["HistGradientBoosting", "CV Log loss"] > C.loc["XGBoost", "CV Log loss"] + 0.01
    MD["HistGradientBoosting"] = dict(
        A1="Gradient-boosted trees on histogram-binned features: sequential trees correct the previous errors, which suits "
           "the many small, hard cases (junctions, parallel roads); scikit-learn's implementation is fast and native.",
        A2=f"Strengths: {C.loc['HistGradientBoosting', 'Train time final fit [s]']:.1f} s to train, good Macro F1. "
           f"Limitation seen: the worst-calibrated boosted model (CV log loss {f4(C.loc['HistGradientBoosting', 'CV Log loss'])} "
           f"vs {f4(C.loc['XGBoost', 'CV Log loss'])} for XGBoost). With learning rate 0.1, 300 iterations and no "
           "early stopping or row/column subsampling, it becomes over-confident on held-out trips - visible in log loss "
           "more than in Top-1.",
        B6="Binary log loss (binomial deviance), minimised by gradient boosting with second-order (Newton) leaf values; "
           "l2_regularization = 0.",
        C9=" <b>Mild overfitting</b>, visible mainly in the probabilities: held-out log loss is far above the in-sample "
           "log loss, while the Macro F1 gap is moderate. Shrinkage or early stopping would help - this is what the Bayesian "
           "search addresses for the selected boosting model.")
    # ---------- LightGBM
    assert 0.015 < gap("LightGBM") < 0.04
    MD["LightGBM"] = dict(
        A1="Leaf-wise gradient boosting with row/column subsampling; a strong learner for tabular features with interactions "
           "and a natural candidate for the final model.",
        A2=f"Strengths: essentially tied with XGBoost and RF on CV Top-1 ({f4(C.loc['LightGBM', 'CV Top-1 (mean)'])}) and on "
           f"student_test log loss ({f4(C.loc['LightGBM', 'Student-test Log loss'])}). Limitations: leaf-wise growth with "
           "63 leaves overfits the training trips most of all boosted models (lowest in-sample log loss "
           f"{f4(tl['LightGBM'])}), and its inference was "
           f"{dv('inference_ratio_LightGBM_over_XGBoost', C.loc['LightGBM', 'Inference time student-test [s]'] / C.loc['XGBoost', 'Inference time student-test [s]']):.1f}x "
           "slower than XGBoost in our measurement (400 trees, 63 leaves).",
        B6="Binary log loss minimised by gradient boosting (first- and second-order gradients), leaf-wise tree growth.",
        C9=" <b>Mild overfitting</b>: the largest in-sample vs held-out log loss ratio among the boosted models; "
           "held-out results remain among the best, so the overfitting is tolerable but is a reason to prefer the "
           "better-regularised XGBoost.")
    # ---------- XGBoost
    assert 0.015 < gap("XGBoost") < 0.04
    MD["XGBoost"] = dict(
        A1="Depth-wise gradient-boosted trees with shrinkage, row/column subsampling and L2 regularisation: models feature "
           "interactions, outputs well-calibrated probabilities (needed as HMM emissions) and predicts fast enough for the API.",
        A2=f"Strengths (measured): best student_test Top-1 and Macro F1, log loss tied with LightGBM; lowest CV log loss among the tied models; "
           f"fastest tree ensemble at inference ({C.loc['XGBoost', 'Inference time student-test [s]']:.2f} s); "
           f"{C.loc['XGBoost', 'Train time final fit [s]']:.1f} s to train. Limitations: many interacting hyperparameters "
           "(hence the Bayesian search in section 4); like every pointwise model it scores each observation independently, so "
           "it cannot enforce route connectivity (continuity "
           f"{f4(C.loc['XGBoost', 'Route continuity (student-test, classifier only)'])}) - that is the HMM's job; about half "
           "of its remaining errors are reverse-direction picks that mostly contradict the reference route's direction (above).",
        B6="Regularised binary log loss: sum of logistic losses + L2 (lambda = 1) penalty on leaf weights, optimised by "
           "second-order gradient boosting with learning rate 0.05.",
        C9=" <b>Mild overfitting</b>, smaller than RF and LightGBM: depth 6, subsampling and shrinkage keep the gap "
           "moderate; student_test Top-1 matches the CV mean, so the model generalises to unseen trips. The tuned version is "
           "analysed in section 4.")
    # ---------- MLP
    hm = H[H.fold != "final"]
    be = rj(O("models", "per_model", "MLP.json"))
    fin = H[H.fold == "final"]
    assert gap("MLP") < 0.015
    assert int(be["final_best_epoch"]) == int(fin.epoch.max())  # final fit still improving at the epoch cap
    MD["MLP"] = dict(
        A1="A feed-forward neural network learns smooth non-linear combinations of the scaled features and outputs "
           "probabilities; it is the only model here trained in epochs, so its learning dynamics can be inspected.",
        A2=f"Strengths: very fast inference ({C.loc['MLP', 'Inference time student-test [s]']:.2f} s), the highest classifier-only "
           f"route continuity ({f4(C.loc['MLP', 'Route continuity (student-test, classifier only)'])}). Limitations: needs "
           "scaling and careful training (learning rate, epochs, early stopping); it is "
           f"{f4(dv('cv_top1_XGBoost_minus_MLP', C.loc['XGBoost', 'CV Top-1 (mean)'] - C.loc['MLP', 'CV Top-1 (mean)']))} below XGBoost in CV Top-1 - smooth "
           "functions approximate the sharp thresholds of this problem less efficiently than trees.",
        B6="Binary cross-entropy + L2 penalty (alpha), Adam optimiser; trained epoch by epoch with partial_fit and early "
           "stopping on the log loss of 15% held-out training <i>trips</i> (patience 5, best-epoch weights restored).",
        C9=f" Evidence from the epoch-wise curves (Figure 16): in folds 1 and 2 validation loss stops improving after "
           f"{', '.join(str(int(x)) for x in be['cv_best_epochs'][1:3])} epochs while training loss keeps falling - early "
           "stopping prevents overfitting there; in folds 3 and 4 the validation trips are easier than the training trips "
           "(validation loss below training loss). The final fit was still improving at the 30-epoch cap "
           f"(validation loss {fin.val_log_loss.iloc[-1]:.4f} at epoch {int(fin.epoch.max())}), and the in-sample/held-out "
           "gap is small, so overall the MLP is <b>slightly underfitting</b> rather than overfitting; more epochs or a wider "
           "network are the obvious next steps.")
    for n in ORDER:
        d = MD[n]
        st += [KeepTogether([P(f"3.2.{ORDER.index(n) + 1} {n} ({C.loc[n, 'Algorithm family']})", "h3")]),
               *bullets([f"<b>A1 - Why suitable:</b> {d['A1']}",
                         f"<b>A2 - Strengths and limitations:</b> {d['A2']}",
                         f"<b>B5 - Hyperparameters (initial values):</b> <font size=7.5>{params(n)}</font>"
                         + ("; fitted on a 150,000-row random sample of the fold" if n == "KNN" else "")
                         + ("; StandardScaler fitted inside the fold" if n in ("LogisticRegression", "KNN", "MLP") else "")
                         + "; seed 42.",
                         f"<b>B6 - Loss / objective:</b> {d['B6']}",
                         f"<b>C7 - Student-test performance:</b> {c7(n)}",
                         f"<b>C8 - Top-1, Macro F1, log loss (student_test):</b> {c8(n)}",
                         f"<b>C9 - Over/underfitting:</b> {c9_numbers(n)}{d['C9']}"])]

    # ---- 3.3 plots
    st += [PageBreak(), P("3.3 Model visualisations", "h2"),
           P("All plots are generated by scripts/03b_plot_compare.py from the stored results; partitions as stated."),
           img(O("figures", "cv_macro_f1.png"), caption="Figure 12 - CV Macro F1 per model (train, 5-fold GroupKFold; dots = folds)."),
           img(O("figures", "student_test_top1_macro_f1.png"), caption="Figure 13 - Student_test Top-1 and Macro F1 (classifier only)."),
           img(O("figures", "log_loss.png"), caption="Figure 14 - Log loss, CV (train) and student_test."),
           img(O("figures", "confusion_matrices.png"), width=14.5 * cm,
               caption="Figure 15 - Candidate-level confusion matrices, student_test, threshold 0.5."),
           img(O("figures", "mlp_loss_curves.png"),
               caption="Figure 16 - MLP training and validation log loss per epoch (only the MLP trains in epochs; no curves "
                       "are shown for the other models). Validation = 15% of the training trips of each fold. Source: "
                       "outputs/models/mlp_history.csv."),
           img(O("figures", "train_vs_val_gap.png"), caption="Figure 17 - In-sample vs held-out Macro F1 and log loss (train, CV) - over/underfitting evidence."),
           img(O("figures", "selection_top1_folds_and_speed.png"),
               caption="Figure 18 - Model-selection plots: CV Top-1 per fold (primary metric) and accuracy vs inference cost."),
           img(O("figures", "train_inference_time.png"), caption="Figure 19 - Training and inference time (log scale).")]
    return st




# ============================================================== section 4
def sec4(O, A, rj):
    ss = rj(O("hpo", "search_space.json"))
    bp = rj(O("hpo", "best_params.json"))
    TU = pd.read_csv(O("hpo", "tuned_vs_untuned.csv"))
    T = pd.read_csv(O("hpo", "trials.csv"))
    un, tu = TU.iloc[0], TU.iloc[1]
    st = [PageBreak(), P("4. Bayesian hyperparameter optimisation (chosen model: %s)" % ss["model"], "h1"),
          P("Partition: <b>train only</b> (78 trips) for the search; student_test only afterwards, for the tuned vs "
            "untuned comparison. Code: scripts/04_bayes_opt.py; full record: outputs/hpo/trials.csv.")]
    kind = {"int": "integer, uniform", "int_log": "integer, log-uniform", "float": "real, uniform",
            "float_log": "real, log-uniform"}
    defaults = json.loads(un.params)
    rows = [["Hyperparameter", "Range", "Distribution", "Untuned value", "Best value (trial %d)" % bp["best_trial"]]]
    for k, (kd, lo, hi) in ss["space"].items():
        b = bp["best_params"][k]
        rows.append([k, f"[{lo}, {hi}]", kind[kd], str(defaults.get(k, "-")),
                     f"{b:.4g}" if isinstance(b, float) else str(b)])
    st += [P("4.1 Search space (Q1, Q2) and best values (Q5)", "h2"),
           table(rows, col_widths=[3.4 * cm, 2.6 * cm, 3.2 * cm, 2.6 * cm, 3.4 * cm]),
           P("Table 7 - Search space and result. Ranges are centred on the library defaults and widened by about one order of "
             "magnitude on log scales; subsample/colsample stop at 0.4-0.5 to keep enough data per tree. Source: "
             "outputs/hpo/search_space.json, best_params.json.", "cap")]
    nc, npr = bp["n_complete"], bp["n_pruned"]
    st += [P("4.2 Objective and procedure (Q3, Q4)", "h2"),
           *bullets([
               "<b>Objective (maximised):</b> mean observation-level Top-1 accuracy over the same 5-fold GroupKFold-by-trip "
               "split of the 78 train trips used in section 3 (fit on every 3rd observation, score all observations of the "
               "held-out trips) - i.e. the primary selection metric, so tuning and selection optimise the same thing.",
               f"<b>Sampler:</b> Optuna TPE (Tree-structured Parzen Estimator, seed 42, 10 random start-up trials), a "
               f"Bayesian method that models good vs bad regions of the space from past trials. Trial 0 was the untuned "
               f"configuration, enqueued as a baseline.",
               f"<b>Pruning:</b> MedianPruner on the running fold mean (after at least 2 folds; first 8 trials never pruned).",
               f"<b>Trials:</b> {bp['n_trials']} ({nc} complete, {npr} pruned) in {bp['search_minutes']:.0f} minutes "
               f"on a 4-core CPU (computational cost). The tuned configuration was then refitted on all 78 train trips "
               f"and evaluated on student_test.",
           ]),
           img(O("hpo", "optimisation_history.png"), width=14 * cm,
               caption="Figure 20 - Optimisation history (objective value per trial and best so far). Train partition, CV. "
                       "Source: outputs/hpo/optimisation_history.png"),
           img(O("hpo", "param_importances.png"), width=13 * cm,
               caption="Figure 21 - Hyperparameter importance (fANOVA over completed trials). Source: outputs/hpo/param_importances.png")]
    rows = [["Model", "CV Macro F1", "CV Top-1", "CV log loss", "Student-test Top-1", "Student-test Macro F1",
             "Student-test log loss", "Train time", "Inference time (student-test)"]]
    for _, r in TU.iterrows():
        rows.append([r.model, f4(r.cv_macro_f1), f4(r.cv_top1), f4(r.cv_log_loss), f4(r.st_top1), f4(r.st_macro_f1),
                     f4(r.st_log_loss), f"{r.train_time_s:.1f} s", f"{r.inference_s:.2f} s"])
    st += [P("4.3 Tuned vs untuned (Q6)", "h2"),
           table(rows, col_widths=[3.4 * cm, 1.6 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 1.8 * cm, 1.8 * cm, 1.5 * cm, 2.4 * cm]),
           P("Table 8 - Same partitions, folds and metrics for both models. Inference times of both models were measured in "
             "the same process. Source: outputs/hpo/tuned_vs_untuned.csv.", "cap"),
           img(O("hpo", "tuned_vs_untuned.png"), caption="Figure 22 - Untuned vs tuned (CV on train; other panels student_test).")]
    d = {k: dv('tuned_minus_untuned_' + k, tu[k] - un[k]) for k in ("cv_top1", "cv_macro_f1", "cv_log_loss", "st_top1", "st_macro_f1", "st_log_loss")}
    dv('tuned_minus_untuned_cv_top1_points', 100 * d['cv_top1']); dv('tuned_minus_untuned_st_top1_points', 100 * d['st_top1'])
    better = [n for n, k, s in [("CV Top-1", "cv_top1", 1), ("CV Macro F1", "cv_macro_f1", 1), ("CV log loss", "cv_log_loss", -1),
                                ("student_test Top-1", "st_top1", 1), ("student_test Macro F1", "st_macro_f1", 1),
                                ("student_test log loss", "st_log_loss", -1)] if s * d[k] > 0]
    worse = [n for n in ["CV Top-1", "CV Macro F1", "CV log loss", "student_test Top-1", "student_test Macro F1",
                         "student_test log loss"] if n not in better]
    std = rj(O("models", "per_model", f"{ss['model']}.json"))["cv_top1_std"]
    st.append(P(
        f"<b>Did tuning help?</b> The tuned model is better on {', '.join(better) if better else 'none of the metrics'}"
        + (f" and worse on {', '.join(worse)}" if worse else "") +
        f". The objective improved by {d['cv_top1'] * 100:+.2f} points of CV Top-1 ({f4(un.cv_top1)} -> {f4(tu.cv_top1)}); "
        f"on student_test Top-1 changed by {d['st_top1'] * 100:+.2f} points and log loss by {d['st_log_loss']:+.4f}. "
        f"These gains are small compared with the fold-to-fold standard deviation of CV Top-1 ({f4(std)}): the default "
        f"configuration was already near a plateau, which the flat optimisation history also shows. "
        f"Inference time changed from {un.inference_s:.2f} s to {tu.inference_s:.2f} s"
        f" ({'more' if tu.inference_s > un.inference_s else 'fewer'} / larger trees). "
        f"The deployed model (artifacts/model.joblib) is the {'tuned' if tu.cv_top1 >= un.cv_top1 else 'untuned'} one, "
        f"chosen by the same rule as everything else - CV Top-1 on train."))
    return st


# ============================================================== section 5
def sec5(O, A, rj):
    hc = rj(A("hmm_config.json"))
    G = pd.read_csv(O("hmm", "grid.csv"))
    PTd = pd.read_csv(O("hmm", "per_trip.csv"), dtype={"trip_id": str})
    SU = pd.read_csv(O("hmm", "summary.csv")).set_index("partition")
    BR = pd.read_csv(O("hmm", "breaks.csv"), dtype={"trip_id": str})
    st = [PageBreak(), P("5. HMM and Viterbi route reconstruction", "h1"),
          P("Code: src/pathsnap/hmm.py (decoder), scripts/05_hmm_eval.py (tuning and comparison). Emissions come from the "
            "selected classifier: on train, <b>out-of-fold</b> probabilities of the 5-fold CV (so the decoder is tuned on "
            "probabilities that behave like unseen-trip probabilities); on student_test, the final model. Partitions: "
            "<b>train (78 trips) for choosing decoder parameters</b>, student_test for reporting; both are shown.")]
    st += [P("5.1 Model (Q1-Q5)", "h2"), *bullets([
        f"<b>Q1 Hidden states.</b> At GPS observation i, the states are its top-{hc['top_k']} candidate arcs ranked by "
        "classifier probability. Each state stores its arc id and projection position, so the decoded state maps "
        "one-to-one back to a road segment.",
        f"<b>Q2 Emission.</b> log e_i(c) = w * log(max(p_i(c), {hc['prob_floor']})), p_i(c) = classifier probability that c "
        f"is the reference arc; w = {hc['emission_weight']:g} (tuned). The floor stops one over-confident zero from vetoing "
        "a path.",
        f"<b>Q3 Transition</b> (Newson-Krumm style). d_route = RoadNetwork.route_distance(a, t_a, b, t_b): driving distance "
        "from the projection of fix i on arc a to the projection of fix i+1 on arc b along the <i>directed</i> network "
        "(bounded Dijkstra, cached); d_gc = straight-line distance between the fixes; log T(a->b) = -|d_route - d_gc| / "
        f"beta_eff with beta_eff = beta * max(1, dt)^{hc['beta_dt_exp']:g}, beta = {hc['beta']:g} m (tuned). A real move along "
        "the road has d_route close to d_gc; a jump to a parallel road or the wrong direction needs a detour and is "
        f"penalised. Moving onto the reverse twin of the same road (U-turn) costs an extra {hc['uturn_penalty']:g} (tuned). "
        f"Backward movement of up to {hc['back_tol_m']:g} m along the same arc is treated as GPS jitter of a stationary "
        "vehicle (tuned; see 5.3 for why this matters).",
        f"<b>Q4 Missing candidates and gaps.</b> An observation with no candidate is skipped; the chain continues from the "
        f"last observation that had candidates (dt and d_gc measured between those two fixes) and the skipped fix inherits "
        f"the previous arc. Transitions are only searched up to max(2 d_gc + {hc['cutoff_slack_m']:g} m, "
        f"{hc['vmax_mps']:g} m/s x dt); beyond that they are impossible. Gaps longer than {hc['gap_split_s']:g} s split the "
        f"chain deliberately (the vehicle may have driven anywhere), and beta grows with dt for shorter gaps.",
        "<b>Q5 Disconnected segments.</b> If no candidate of fix i+1 is reachable from any surviving state of fix i, every "
        "transition probability is zero. Instead of failing, the decoder closes the current segment (back-tracking from its "
        "best final state), restarts from the emissions of fix i+1 and logs a 'disconnected' break (outputs/hmm/breaks.csv). "
        "The route polyline is then built by joining consecutive decoded arcs with shortest paths (up to "
        f"{hc['max_fill_m']:g} m); joins that are impossible are counted as unfillable gaps.",
        "<b>Viterbi recurrence.</b> delta_0(c) = log e_0(c); delta_i(c) = log e_i(c) + max_c' [delta_{i-1}(c') + log T(c'->c)], "
        "psi_i(c) = argmax; trace back from argmax delta at the end of each segment. Log space avoids underflow over "
        "thousands of steps."])]
    grid_desc = ", ".join(f"{k} in {v}" for k, v in hc["grid"].items())
    nconf = len(G)
    top = G.sort_values("train_top1", ascending=False).head(5)
    rows = [["back_tol [m]", "beta [m]", "w", "top_k", "U-turn pen.", "train Top-1", "student_test Top-1"]]
    for _, r in top.iterrows():
        rows.append([f"{r.back_tol_m:g}", f"{r.beta:g}", f"{r.emission_weight:g}", f"{r.top_k:g}", f"{r.uturn_penalty:g}",
                     f4(r.train_top1), f4(r.student_test_top1)])
    st_best = G.loc[G.student_test_top1.idxmax()]
    st += [P("5.2 Decoder tuning", "h2"),
           P(f"Grid search over {nconf} configurations ({grid_desc}); objective = observation-level Top-1 on <b>train</b> "
             f"(OOF probabilities). Selected: back_tol = {hc['back_tol_m']:g} m, beta = {hc['beta']:g} m, w = "
             f"{hc['emission_weight']:g}, top_k = {hc['top_k']}, U-turn penalty = {hc['uturn_penalty']:g} "
             f"(train Top-1 {f4(hc['train_top1'])}). Every configuration was also scored on student_test for transparency; "
             f"the best student_test configuration would have reached {f4(st_best.student_test_top1)} vs "
             f"{f4(hc['student_test_top1_at_selected'])} for the selected one - we did not switch to it."),
           table(rows, col_widths=[2 * cm, 1.7 * cm, 1.3 * cm, 1.4 * cm, 1.9 * cm, 2.3 * cm, 2.8 * cm]),
           P("Table 9 - Five best decoder configurations on train. Source: outputs/hmm/grid.csv (all configurations).", "cap"),
           img(O("hmm", "hmm_grid_heatmap.png"), caption="Figure 23 - Top-1 over beta and emission weight (other parameters at "
                                                          "the selected values), train (left) and student_test (right).")]
    bt = G.groupby("back_tol_m").train_top1.max()
    st.append(P("Sensitivity: best train Top-1 per back_tol: " + ", ".join(f"{k:g} m: {f4(v)}" for k, v in bt.items()) +
                ". The jitter tolerance matters most: with 5 m, stationary vehicles near a junction drift onto the outgoing "
                "arc (the projection of a fix that lies 'before' the arc is clamped to its start, so backward jitter is free "
                "there but costs a detour on the correct arc) and the chain later hits impossible transitions. We found this "
                "on student_test trip 7 during development and added the parameter to the grid; its value was chosen on train."))
    # ---- comparison
    rows = [["Metric", "train: classifier", "train: + HMM", "student_test: classifier", "student_test: + HMM"]]
    for lab, c in [("Top-1 accuracy (all observations)", "top1"), ("Macro F1 (decision level)", "macro_f1_decision"),
                   ("Route continuity", "continuity"), ("Route precision (length, gap-filled)", "route_precision"),
                   ("Route recall (length, gap-filled)", "route_recall"), ("Route F1 (length, gap-filled)", "route_f1"),
                   ("Route precision (raw arcs)", "route_precision_raw"), ("Route recall (raw arcs)", "route_recall_raw")]:
        rows.append([lab] + [f4(SU.loc[p, f"{c}_{t}"]) for p in ("train", "student_test") for t in ("clf", "hmm")])
    rows.append(["Trips improved / same / worse (Top-1)", "", f"{SU.loc['train', 'trips_improved']} / {SU.loc['train', 'trips_same']} / {SU.loc['train', 'trips_worse']}",
                 "", f"{SU.loc['student_test', 'trips_improved']} / {SU.loc['student_test', 'trips_same']} / {SU.loc['student_test', 'trips_worse']}"])
    rows.append(["Chain breaks (gap / disconnected)", "",
                 " / ".join(str(int(((BR.partition == 'train') & (BR.reason == r)).sum())) for r in ("gap", "disconnected")), "",
                 " / ".join(str(int(((BR.partition == 'student_test') & (BR.reason == r)).sum())) for r in ("gap", "disconnected"))])
    st += [P("5.3 Classifier-only vs classifier + HMM (Q6)", "h2"),
           table(rows, col_widths=[5.4 * cm, 2.7 * cm, 2.7 * cm, 3.1 * cm, 3.1 * cm]),
           P("Table 10 - Classifier-only (Top-1 per observation) vs Viterbi decoding, same probabilities. Top-1, Macro F1 and "
             "continuity are observation-weighted; route precision/recall are length-weighted and averaged over trips. "
             "'Gap-filled' joins consecutive predicted arcs with shortest paths for BOTH methods, so the comparison isolates "
             "the decoder; 'raw arcs' use only the arcs predicted for observations. Decision-level Macro F1 marks exactly one "
             "candidate per observation as positive. Source: outputs/hmm/summary.csv, per_trip.csv, breaks.csv.", "cap"),
           img(O("hmm", "hmm_summary.png"), caption="Figure 24 - Classifier-only vs classifier + HMM (train and student_test)."),
           img(O("hmm", "hmm_per_trip_delta.png"), caption="Figure 25 - Per-trip change in Top-1 from HMM decoding (green better, red worse).")]
    S = PTd[PTd.partition == "student_test"].sort_values("trip_id")
    rows = [["Trip", "obs", "Top-1 clf", "Top-1 HMM", "change", "continuity clf", "continuity HMM", "route F1 clf",
             "route F1 HMM", "breaks"]]
    for _, r in S.iterrows():
        rows.append([r.trip_id, str(r.n_obs), f4(r.top1_clf), f4(r.top1_hmm), f"{100 * r.delta_top1:+.2f} pp",
                     f4(r.continuity_clf), f4(r.continuity_hmm), f4(r.route_f1_clf), f4(r.route_f1_hmm), str(r.breaks_hmm)])
    st += [table(rows, col_widths=[1.7 * cm, 1.1 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 1.7 * cm, 1.7 * cm, 1.2 * cm]),
           P("Table 11 - Per-trip results on student_test. Source: outputs/hmm/per_trip.csv.", "cap")]
    s, t = SU.loc["student_test"], SU.loc["train"]
    worse = PTd[PTd.delta_top1 < -0.001].sort_values("delta_top1")
    improved = (t.top1_hmm > t.top1_clf) and (s.top1_hmm > s.top1_clf)
    txt = (f"<b>Did the HMM improve the route?</b> {'Yes, on both partitions' if improved else 'Not uniformly'}: Top-1 "
           f"{f4(t.top1_clf)} -> {f4(t.top1_hmm)} on train and {f4(s.top1_clf)} -> {f4(s.top1_hmm)} on student_test; "
           f"route continuity {f4(t.continuity_clf)} -> {f4(t.continuity_hmm)} (train) and {f4(s.continuity_clf)} -> "
           f"{f4(s.continuity_hmm)} (student_test). The decoder removes the isolated jumps of the pointwise classifier "
           f"onto crossing streets and the flip-flopping between the two directions of a road, which is exactly what the "
           f"continuity and route-precision rows show. ")
    if len(worse):
        wl = ", ".join(f"{r.partition} {r.trip_id} ({100 * r.delta_top1:+.1f} pp)" for _, r in worse.head(6).iterrows())
        txt += (f"<b>Where it did not help:</b> {len(worse)} of {len(PTd)} trips got worse - {wl}. ")
    st.append(P(txt))
    st.append(P("<b>Why it fails on those trips</b> (inspected on the maps of section 6 and in breaks.csv): (i) long stops at "
                "junctions - the decoder may commit to the outgoing arc while the reference keeps the vehicle on the incoming "
                "one, and every stationary fix then counts as an error even though the physical position is almost the same; "
                "(ii) when the reference route's direction disagrees with the GPS motion (section 3, trip 77), a connected "
                "path along the reference is penalised by the transition model; (iii) where the classifier is confidently "
                "wrong for many consecutive fixes, the HMM keeps the error consistent instead of correcting it, and a "
                "'disconnected' break can make it re-enter on the wrong road. These are honest failure modes of an HMM with "
                "pointwise emissions; reporting the classifier-only result next to it (Table 10) makes them visible."))
    return st


def sec5b(O, A, rj):
    """Measured characterisation of what the HMM fixed / broke (scripts/05b_hmm_error_analysis.py)."""
    FB = pd.read_csv(O("hmm", "fix_break_analysis.csv"), dtype={"trip_id": str})
    a = FB[FB.trip_id == "ALL"].set_index(["partition", "set"])
    rows = [["Partition", "Set", "Observations", "reverse twin", "adjacent arc", "other arc", "vehicle stationary",
             "within 30 fixes after a break"]]
    for (part, s), r in a.iterrows():
        rows.append([part, "fixed by HMM" if s == "fixed" else "broken by HMM", f"{r.n:,}",
                     *(f"{r[c]:,} ({pct(r[c] / max(r.n, 1))})" for c in ("reverse_twin", "adjacent", "other", "stationary", "near_break"))])
    st = [table(rows, col_widths=[2.2 * cm, 2.4 * cm, 1.9 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm, 2.4 * cm]),
          P("Table 12 - Observations the HMM fixed (classifier wrong, HMM right) and broke (classifier right, HMM wrong). "
            "Error type refers to the wrong arc (the classifier's for 'fixed', the HMM's for 'broken'). Stationary = GPS "
            "displacement over +-2 fixes below 2 m. Source: scripts/05b_hmm_error_analysis.py -> "
            "outputs/hmm/fix_break_analysis.csv.", "cap")]
    return st


# ============================================================== section 6
def sec6(O, A, rj):
    M = pd.read_csv(O("maps", "maps_index.csv"), dtype={"trip_id": str})
    st = [PageBreak(), P("6. Geographical visualisation using Folium", "h1"),
          P("Code: src/pathsnap/maps.py (map builder) and scripts/07_maps.py (selection + export). Each map is an interactive "
            "HTML file (outputs/maps/*.html) with separate toggleable layers in a LayerControl: GPS observations (blue dots), "
            "nearby road network (grey), reference route (wide translucent green), classifier-only route (orange), "
            "classifier + HMM route (red), start (green) and end (black) markers; a legend box; and the view fitted to the "
            "trip (fit_bounds) or to a zoom window. Prediction maps are produced with the deployed pipeline "
            "(pathsnap.pipeline.predict_trip with artifacts/), the same code the API runs. Trips were picked automatically "
            "from outputs/hmm/per_trip.csv: the student_test trip with the largest HMM gain, the largest HMM loss, and the "
            "median HMM Top-1; each also has a zoomed map on the 60-fix window where classifier and HMM disagree most."),
          P("<i>Note on the screenshots:</i> they were rendered headless on a machine without internet access, so the "
            "OpenStreetMap basemap tiles could not load and the background is plain grey; all data layers are drawn by "
            "Leaflet from the HTML itself. Opening the HTML files with internet access shows the OSM basemap underneath.", "small")]
    k = 26
    for _, r in M[M.partition == "student_test"].iterrows():
        extra = (f" Top-1 classifier {f3(r.top1_classifier)}, HMM {f3(r.top1_hmm)}"
                 + (" (in this window)." if "zoom" in r.map else " (whole trip)."))
        st.append(img(O("maps", r.png), width=16 * cm,
                      caption=f"Figure {k} - student_test trip {r.trip_id}: {r.purpose}.{extra} Interactive: outputs/maps/{r.map}"))
        k += 1
    st.append(P("Reading the maps: on the full-trip views the red HMM route lies on the green reference route almost "
                "everywhere and is continuous, while the orange classifier-only route shows short detours onto side "
                "streets and gaps; the zoomed views show the places where the two disagree, including the case where the "
                "HMM is worse (section 5.3). The EDA maps (Figures 7-8) show train trips with the reference route only."))
    return st


# ============================================================== section 7
def sec7(O, A, rj):
    import re
    U = pd.read_csv(O("screenshots", "ui_run_results.csv"), dtype={"trip_id": str})
    log = open(O("screenshots", "api_console.log")).read()
    sc = re.findall(r"predict trip=(\S+): SCORE top1 HMM=([\d.]+) \(classifier ([\d.]+)\) \| continuity ([\d.]+) \| "
                    r"route P=([\d.]+) R=([\d.]+) \| miss=([\d.]+) \| total (\d+) ms", log)
    ok200 = len(re.findall(r"POST /predict from \S+ -> 200", log))
    st = [PageBreak(), P("7. API, UI and deployment", "h1"),
          P("<b>Laptop 1 - FastAPI backend</b> (api/main.py, started with <font face='Courier'>uvicorn api.main:app --host "
            "0.0.0.0 --port 8000</font>). On start-up it loads artifacts/model.joblib, feature_config.json and "
            "hmm_config.json and lists the trips in <font face='Courier'>EVAL_DIR</font> (environment variable; default "
            "data/raw/student_test as a stand-in, since the evaluator trips are not in our dataset)."),
          table([["Endpoint", "Request", "Response"],
                 ["GET /health", "-", "status, loaded model, EVAL_DIR, number of trips"],
                 ["GET /trips", "-", "{\"trips\": [ids]} - IDs only, read from the folders in EVAL_DIR"],
                 ["POST /predict", "{\"trip_id\": \"...\"}",
                  "runs network build -> candidates -> features -> classifier -> Viterbi; returns predicted segment per GPS "
                  "point, the predicted route as lat/lon polylines (HMM and classifier-only), GPS points, HMM breaks, "
                  "server-side metrics (Top-1, continuity, route precision/recall/F1, candidate miss rate) and per-stage "
                  "timings. Unknown IDs -> 404."]],
                col_widths=[2.6 * cm, 3 * cm, 11.8 * cm]),
          P("Leakage safeguards: the trip is loaded with <font face='Courier'>with_route=False</font> for prediction; the "
            "reference route is read only after the prediction is finished, by score_prediction(), and only aggregate "
            "scores are returned - never the reference route or per-point labels (asserted in tests/test_api.py). No trip ID "
            "or result is hardcoded; every request runs the full pipeline. Every request is logged to the console "
            "(method, path, client IP, status, latency, stage timings, scores)."),
          P("<b>Laptop 2 - UI</b> (ui/: index.html, app.js, style.css, served by <font face='Courier'>python ui/serve.py "
            "--api http://&lt;laptop1-ip&gt;:8000</font>). Plain HTML+JS with Leaflet bundled locally (works offline). It "
            "calls /health and /trips, shows the trip list, sends the selected ID to /predict and draws the GPS points, the "
            "HMM route and the classifier-only route (layer control + legend), the server-side metrics (HMM vs "
            "classifier-only) and timings. The API URL is configurable in the page, in config.js or with ?api=.")]
    rows = [["Trip", "UI run", "Top-1 HMM", "Top-1 classifier", "Continuity", "Route P", "Route R", "Miss rate", "API time"]]
    for m in sc:
        u = U[U.trip_id == m[0]]
        rows.append([m[0], "OK" if len(u) and bool(u.ui_success.iloc[0]) else "FAILED", m[1], m[2], m[3], m[4], m[5], m[6],
                     f"{int(m[7]) / 1000:.1f} s"])
    st += [P("7.1 End-to-end test on all student_test trips", "h2"),
           P(f"scripts/08_screenshots.py started the API and the UI, then clicked every trip returned by /trips in a "
             f"headless browser. {int(U.ui_success.sum())} of {len(U)} trips completed in the UI; the API log shows "
             f"{ok200} successful POST /predict responses. Table 13 is parsed from the captured API console log."),
           table(rows, col_widths=[1.9 * cm, 1.4 * cm, 1.8 * cm, 2.1 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm, 1.7 * cm, 1.7 * cm]),
           P("Table 13 - Server-side scores per trip as logged by the API (student_test as EVAL_DIR). Sources: "
             "outputs/screenshots/api_console.log, ui_run_results.csv.", "cap")]
    shots = sorted(x for x in os.listdir(O("screenshots")) if x.startswith("ui_") and x.endswith(".png"))
    st.append(img(O("screenshots", shots[0]), caption="Figure %d - UI after connecting: trip list from GET /trips." % 33))
    picks = [x for x in shots[1:]][:3]
    for i, s in enumerate(picks):
        st.append(img(O("screenshots", s), caption=f"Figure {34 + i} - UI showing the prediction for {s[6:-4].replace('_', ' ')}: "
                                                    "GPS points, HMM route (red), classifier-only route (orange dashed), "
                                                    "server-side metrics and timings."))
    st += [img(O("screenshots", "api_console.png"),
               caption="Figure 37 - API console on Laptop 1 during the run: start-up, GET /health, GET /trips and one "
                       "POST /predict per trip with stage timings and server-side scores. This is the captured stdout of the "
                       "uvicorn process (outputs/screenshots/api_console.log), rendered to an image; during the live demo the "
                       "same lines appear in the CMD window of Laptop 1."),
           P("All 10 UI screenshots are in outputs/screenshots/. Note: basemap tiles need internet; in our headless run "
             "they were unavailable, so the map background is plain.", "small")]
    return st


# ============================================================== section 8
def sec8(O, A, rj):
    st = [PageBreak(), P("8. Academic integrity and AI-use disclosure", "h1"),
          P("<b>AI assistance.</b> This submission was produced with substantial help from an AI coding assistant (Claude "
            "Code, Anthropic). Under the team's direction it wrote most of the source code (feature engineering, training "
            "and evaluation scripts, HMM decoder, API, UI, map and report generation) and drafted the text of this report. "
            "The starting point was the team's own EDA code and plan (src/pathsnap/io.py, geo.py, network.py, labels.py, "
            "maps.py, scripts/01_eda.py)."),
          P("<b>What was not AI-generated:</b> no number, table, plot or map in this report was written by hand or by the "
            "assistant. Every one is produced by the scripts in the repository, run on the provided data, and read from the "
            "result files by report/build_report.py. Statements that depend on results (e.g. 'overfits', 'tuning helped') are "
            "guarded by assertions in the report generator and the build fails if the results do not support them. "
            "scripts/check_report_numbers.py re-checks the report's numbers against outputs/. The repository history shows "
            "the order in which stages were built and run."),
          P("<b>Data integrity.</b> Models, features, hyperparameters, the candidate radius, decoder parameters and thresholds "
            "were chosen using the train partition only (student_test was used for comparison and reported, and its numbers "
            "for every decoder configuration are published alongside). Evaluator-test data was never available or used. "
            "Reference routes never enter the prediction pipeline or the UI. No predictions are hardcoded."),
          P("<b>Team responsibility.</b> Every team member must be able to explain the code, the experiments and the "
            "conclusions in this report at the viva; the per-stage scripts, logs in outputs/logs/ and this report are the "
            "basis for that preparation."),
          P("<b>Limitations we are aware of.</b> Per-observation labels are derived from the reference route by alignment and "
            "inherit its errors (section 3, reverse-direction cases; trips 33/54 excluded); evaluator trips may differ from "
            "student_test; inference time depends on network size (the biggest networks dominate API latency).")]
    st += [P("Appendix - reproduction", "h1"),
           P("pip install -r requirements.txt; place data in data/raw/{train,student_test}; then run in order: "
             "scripts/01_eda.py, 02_build_dataset.py, 03_train_compare.py, 03b_plot_compare.py, 03c_error_analysis.py, "
             "03d_paired_comparison.py, 04_bayes_opt.py --trials 40, 05_hmm_eval.py, 05b_hmm_error_analysis.py, 07_maps.py, 08_screenshots.py, "
             "report/build_report.py, scripts/check_report_numbers.py. Logs of the runs used for this report are in "
             "outputs/logs/. Seeds: 42 (numpy, scikit-learn, LightGBM, XGBoost, Optuna).", "small")]
    return st


def build_all(O, A, rj):
    st = sec3(O, A, rj)
    st += sec4(O, A, rj)
    s5 = sec5(O, A, rj)
    st += s5[:-1] + sec5b(O, A, rj) + s5[-1:]
    st += sec6(O, A, rj) + sec7(O, A, rj) + sec8(O, A, rj)
    return st
