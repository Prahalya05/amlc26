"""M13: build the final submission zip (README 'Final Submission Package').

  python m13_package.py TEAM_NAME [OUTPUT_DIR]     (OUTPUT_DIR default: <ROOT>/submission/output_v5g)
-> <ROOT>/submission/<TEAM_NAME>_submission.zip with output/, code/business_entity_resolution/{src,README.md,
   requirements.txt}, Documentation_template.md. Nothing is changed in the pipeline or the output folders.
"""
import os
import sys
import time
import zipfile

ROOT = os.environ.get("AMLC_ROOT", r"C:\amlc")
PIPE = os.path.join(ROOT, "pipeline")
SRC = ["common.py", "m1_prepare.py", "m3_split.py", "keys.py", "keys2.py", "skeleton.py", "romanize.py", "records.py",
       "blocking.py", "blocking_v3.py", "features2.py", "m4_block_eval.py", "m5_train.py", "m6_decide.py",
       "m6b_expected_f.py", "m7_infer.py", "m9_cands.py", "m10_v5.py", "m12_dense_merge.py", "block_cfg.json"]
KAGGLE = ["kaggle_dense.py", "README_kaggle.md"]
MODELS = ["models/lgb_model.txt", "models/feature_list.txt", "models/decision.json", "models/v3/lgb_model.txt",
          "models/v3/lgb_fold0.txt", "models/v3/old_fold0_model.txt", "models/v3/cand_cfg.json",
          "models/v3/decision.json", "models/v5/merge_v5d.json"]


def main(team, out_dir):
    t0 = time.time()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert os.path.exists(os.path.join(out_dir, f)), f
    z = os.path.join(ROOT, "submission", f"{team}_submission.zip")
    assert not os.path.exists(z), f"{z} exists"
    base = "code/business_entity_resolution"
    with zipfile.ZipFile(z + ".tmp", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            zf.write(os.path.join(out_dir, f), f"output/{f}")
        for f in SRC:
            zf.write(os.path.join(PIPE, f), f"{base}/src/{f}")
        for f in KAGGLE:
            zf.write(os.path.join(ROOT, "kaggle", f), f"{base}/src/kaggle/{f}")
        for f in MODELS:
            zf.write(os.path.join(PIPE, f), f"{base}/models_trained/{f[len('models/'):]}")
        zf.write(os.path.join(PIPE, "package", "README.md"), f"{base}/README.md")
        zf.write(os.path.join(PIPE, "package", "requirements.txt"), f"{base}/requirements.txt")
        zf.write(os.path.join(PIPE, "package", "Documentation_template.md"), "Documentation_template.md")
    os.replace(z + ".tmp", z)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    print(f"{z}: {os.path.getsize(z) / 2**20:,.0f} MB, {len(names)} files, CRC check {'OK' if bad is None else bad}, "
          f"{time.time() - t0:.0f}s")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "submission", "output_v5g"))
