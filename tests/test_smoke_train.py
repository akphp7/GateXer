"""End-to-end smoke test on synthetic data (no real grid data needed)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "gatexer")


def test_train_one_epoch_on_synthetic_data(tmp_path):
    subprocess.run([sys.executable, os.path.join(SRC, "make_synth.py")], cwd=tmp_path, check=True, capture_output=True)
    out = tmp_path / "run.json"
    subprocess.run([sys.executable, os.path.join(SRC, "train.py"), "--csv", str(tmp_path / "synth.csv"),
                    "--exog", "temp,cloud,wspeed", "--preset", "full", "--decomp", "25", "--ple", "0", "--net", "0", "--ramp", "0",
                    "--epochs", "1", "--bs", "128", "--max_train_windows", "600", "--out", str(out)],
                   check=True, capture_output=True)
    r = json.load(open(out))
    assert len(r["metrics"]["nmae"]) == 3 and all(v > 0 for v in r["metrics"]["nmae"])


def test_baselines_run(tmp_path):
    subprocess.run([sys.executable, os.path.join(SRC, "make_synth.py")], cwd=tmp_path, check=True, capture_output=True)
    for preset in ("snaive", "dlinear", "patchtst", "itransformer", "timexer"):
        out = tmp_path / f"{preset}.json"
        subprocess.run([sys.executable, os.path.join(SRC, "train.py"), "--csv", str(tmp_path / "synth.csv"), "--exog", "temp,cloud,wspeed",
                        "--preset", preset, "--epochs", "1", "--bs", "128", "--max_train_windows", "400", "--out", str(out)],
                       check=True, capture_output=True)
        assert out.exists()


def test_run_experiments_dry_run():
    r = subprocess.run([sys.executable, os.path.join(SRC, "run_experiments.py"), "--stage", "main", "--dry", "--datasets", "opsd_DE",
                        "--presets", "gx,timexer", "--seeds", "0", "--data_dir", "data/final", "--out", "runs_dry_test"],
                       capture_output=True, text=True)
    assert r.returncode == 0
