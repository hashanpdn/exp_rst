"""End-to-end smoke tests: tiny synthetic runs through the full pipeline.

Verifies: config plumbing, all partitions, all assignment methods, all
weighting presets, all DP modes, the auto selector, CSV/JSON outputs, and
that the accountant reports a finite eps for DP runs.
"""
import json
import os

import pytest

from hdpba_hfl.train import main

BASE = [
    "--dataset", "synthetic", "--num_clients", "20", "--num_edges", "4",
    "--global_rounds", "2", "--intermediate_rounds", "1",
    "--local_epochs", "1", "--batch_size", "32", "--device", "cpu",
    "--partition", "pathological", "--classes_per_client", "2",
]


def run(tmp_path, name, *extra):
    argv = BASE + ["--outdir", str(tmp_path), "--name", name, "--seed", "7",
                   *extra]
    return main(argv)


@pytest.mark.parametrize("method", ["bl1", "bl2", "escs", "esca", "auto"])
def test_methods_run(tmp_path, method):
    s = run(tmp_path, f"m_{method}", "--method", method,
            "--assignment", "random", "--dp_mode", "baseline")
    assert 0.0 <= s["final_acc"] <= 1.0
    assert os.path.exists(os.path.join(tmp_path, f"m_{method}_seed7",
                                       "metrics.csv"))


@pytest.mark.parametrize("assignment", ["random", "oracle", "nonprivate",
                                        "rr", "hdpba"])
def test_assignments_run(tmp_path, assignment):
    s = run(tmp_path, f"a_{assignment}", "--method", "escs",
            "--assignment", assignment, "--dp_mode", "baseline")
    assert s["assignment_method"] in (assignment, "greedy", "hdpba",
                                      "nonprivate_greedy", "oracle", "random",
                                      "rr_presence")
    assert s["d_res"] >= 0


def test_rr_assignment_charges_budget(tmp_path):
    s = run(tmp_path, "rr_budget", "--method", "escs", "--assignment", "rr",
            "--eps_assign", "1.0", "--dp_mode", "baseline")
    assert any("rr_presence" in e for e in s["privacy_events"])


def test_cg_ng_mode_runs(tmp_path):
    s = run(tmp_path, "cgng", "--method", "escs", "--assignment", "random",
            "--dp_mode", "cg-ng", "--clip", "1.0", "--sigma", "0.01")
    assert 0.0 <= s["final_acc"] <= 1.0


@pytest.mark.parametrize("dp_mode", ["baseline", "cg-np", "cp-np"])
def test_dp_modes_and_accounting(tmp_path, dp_mode):
    s = run(tmp_path, f"dp_{dp_mode}", "--method", "escs",
            "--assignment", "hdpba", "--dp_mode", dp_mode,
            "--sigma", "0.5", "--clip", "1.0")
    if dp_mode == "baseline":
        # only assignment events present
        assert all("training" not in e for e in s["privacy_events"])
    else:
        assert any("training" in e for e in s["privacy_events"])
        assert s["eps_total"] < float("inf")


def test_auto_selector_records_choice(tmp_path):
    s = run(tmp_path, "auto_sel", "--method", "auto", "--assignment", "hdpba",
            "--dp_mode", "baseline", "--tau", "0.15")
    assert s["selected_central_rule"] in ("avg_samplesize", "accuracy")


@pytest.mark.parametrize("partition", ["iid", "dirichlet", "powerlaw",
                                       "compound"])
def test_partitions_run(tmp_path, partition):
    s = run(tmp_path, f"p_{partition}", "--method", "escs",
            "--assignment", "random", "--dp_mode", "baseline",
            "--partition", partition)
    assert 0.0 <= s["final_acc"] <= 1.0


def test_summary_is_complete(tmp_path):
    s = run(tmp_path, "full", "--method", "auto", "--assignment", "hdpba",
            "--dp_mode", "cp-np", "--sigma", "0.5")
    path = os.path.join(tmp_path, "full_seed7", "summary.json")
    with open(path) as f:
        j = json.load(f)
    for key in ("final_acc", "best_acc", "acc_auc", "eps_total", "d_res",
                "selected_central_rule", "privacy_events", "config"):
        assert key in j


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
