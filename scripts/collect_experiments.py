"""
experiments/ 以下の全 run（config.json を持つディレクトリ）を走査し、
主要な実験設定と結果を 1 run = 1 行の CSV にまとめるスクリプト。

読み取る情報:
    config.json             : 学習スクリプトが保存した設定（model / training / data）
    experiment_summary.json : 学習終了時の final_metrics（最終エポックの val 指標と best_f1_macro）
    logs/ (TensorBoard)     : val_acc / val_f1 の最良値とそのエポック（logs が残っている run のみ）
    train_log.json          : AFS StyleExtractor 学習 run の最終エポック損失

使い方:
    python scripts/collect_experiments.py \
        --roots experiments experiments_other \
        --out_csv document/experiment_runs.csv
"""

import argparse
import csv
import glob
import json
import os


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def read_tb_best(run_dir):
    """TensorBoard ログから val_acc / val_f1 の最良値と到達エポックを返す（無ければ空 dict）"""
    logs = os.path.join(run_dir, "logs")
    if not os.path.isdir(logs):
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logs, size_guidance={"scalars": 0})
    ea.Reload()
    res = {}
    for tag in ("val_acc", "val_f1"):
        if tag not in ea.Tags().get("scalars", []):
            continue
        ev = ea.Scalars(tag)
        best = max(ev, key=lambda e: e.value)
        res[f"tb_{tag}_best"] = round(best.value, 4)
        res[f"tb_{tag}_best_epoch"] = best.step
        res["tb_epochs_logged"] = len(ev)
    return res


def first(c, *keys):
    for k in keys:
        if c.get(k) not in (None, ""):
            return c[k]
    return ""


def infer_model(c):
    if "latent_dir" in c:
        # train_style_extractor.py（論文再現）と train_fer_extractor.py（FER特化）の判別
        return "AFS-extractor(FER)" if "lambda_expr" in c else "AFS-extractor(paper)"
    m = first(c, "model.model", "model.type")
    if m:
        return m
    if "model.backbone" in c:
        return f"ResNet({c['model.backbone']})"
    if "model.model_type" in c:
        return f"LatentCNN({c['model.model_type']})"
    if "model.freeze_transformer" in c:
        return f"HybridLatentViT({c.get('model.model_size')})"
    if "model.img_size" in c:
        return "ImageViT"
    if "model.depth" in c:
        return "LatentViT"
    return "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", default=["experiments", "experiments_other"])
    p.add_argument("--out_csv", default="document/experiment_runs.csv")
    args = p.parse_args()

    rows = []
    for root in args.roots:
        for cfg_path in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            run_dir = os.path.dirname(cfg_path)
            c = flatten(json.load(open(cfg_path)))
            row = {
                "run_dir": run_dir,
                "model": infer_model(c),
                "train_dir": first(c, "data.train_dir", "latent_dir"),
                "val_dir": first(c, "data.val_dir", "val_latent_dir"),
                "use_pretrained": first(c, "model.use_pretrained"),
                "n_parameters": first(c, "model.n_parameters"),
                "depth": first(c, "model.depth"),
                "dropout": first(c, "model.dropout"),
                "epochs": first(c, "training.epochs", "epochs"),
                "lr": first(c, "training.lr", "lr"),
                "batch_size": first(c, "training.batch_size", "batch_size"),
                "data_fraction": first(c, "training.data_fraction"),
                "use_class_weights": first(c, "training.use_class_weights"),
                "label_smoothing": first(c, "training.label_smoothing"),
                "mixup": first(c, "training.mixup"),
                "use_augmentation": first(c, "training.use_augmentation"),
                "scheduler": first(c, "training.scheduler"),
                "seed": first(c, "training.seed"),
                "modules": " ".join(k.split(".")[-1] for k in
                                    ("model.use_lwn", "model.use_lwn_residual", "model.use_spe", "model.use_leam")
                                    if c.get(k)),
                "projection": first(c, "training.svm_projection", "training.pca_projection")
                if first(c, "training.svm_basis", "training.pca_basis") else "",
                "basis": first(c, "training.svm_basis", "training.pca_basis"),
                "style_extractor": first(c, "training.style_extractor_path"),
            }
            summ = os.path.join(run_dir, "experiment_summary.json")
            if os.path.exists(summ):
                s = json.load(open(summ))
                fm = s.get("final_metrics") or {}
                for k in ("accuracy", "f1_macro", "best_f1_macro"):
                    if isinstance(fm.get(k), float):
                        row[f"final_{k}"] = round(fm[k], 4)
                if s.get("duration_seconds"):
                    row["duration_min"] = round(s["duration_seconds"] / 60, 1)
            tlog = os.path.join(run_dir, "train_log.json")
            if os.path.exists(tlog):
                last = json.load(open(tlog))[-1]
                row["extractor_last_epoch"] = json.dumps({k: round(v, 4) for k, v in last.items()})
            row.update(read_tb_best(run_dir))
            rows.append(row)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {args.out_csv}")


if __name__ == "__main__":
    main()
