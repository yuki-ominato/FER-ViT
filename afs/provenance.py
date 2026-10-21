"""
StyleExtractor チェックポイントと latent ディレクトリの「作成元（provenance）」を記録・参照するヘルパー。

記録の流れ:
    train_style_extractor.py / train_fer_extractor.py
        → チェックポイントに ckpt['provenance'] を埋め込む（抽出器の種類・引数・git commit）
    generate_latents.py / extract_style_latents.py
        → 出力ディレクトリに provenance.json を書く（どのエンコーダ／どの抽出器から作ったか）
    train_latent_vit*.py
        → config.json の data に train/val ディレクトリの provenance.json の中身を写す

provenance を持たない旧形式のチェックポイントは、run ディレクトリの config.json から
抽出器の種類を推定する（`lambda_expr` を持てば FER 特化版、無ければ論文再現版）。
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys
from typing import Optional

import torch

PROVENANCE_FILE = "provenance.json"
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 抽出器の種類（train スクリプトとの対応）
EXTRACTOR_PAPER = "paper"   # train/train_style_extractor.py + afs/losses.py (AFSLoss)
EXTRACTOR_FER = "fer"       # train/train_fer_extractor.py   + afs/fer_losses.py (AFSFERLoss)


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def git_info() -> dict:
    """リポジトリの commit と未コミット変更の有無を返す（git が無い環境では unknown）"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_extractor_provenance(extractor_type: str, args, run_dir: str) -> dict:
    """抽出器チェックポイントに埋め込む provenance を作る"""
    return {
        "kind": "style_extractor",
        "extractor_type": extractor_type,
        "script": os.path.basename(sys.argv[0]),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "run_dir": os.path.abspath(run_dir),
        "created_at": now(),
        "git": git_info(),
    }


def describe_extractor_checkpoint(ckpt_path: str, ckpt: Optional[dict] = None) -> dict:
    """
    チェックポイントの作成元を返す。
    新形式は ckpt['provenance'] をそのまま使い、旧形式は run ディレクトリの config.json から推定する。
    """
    if ckpt is None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "provenance" in ckpt:
        desc = dict(ckpt["provenance"])
        desc["provenance_source"] = "checkpoint"
    else:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
        cfg_path = os.path.join(run_dir, "config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
        if cfg is None:
            extractor_type = "unknown"
        elif "extractor_type" in cfg:
            extractor_type = cfg["extractor_type"]
        elif "lambda_expr" in cfg or "fer_image_ckpt" in cfg:
            extractor_type = EXTRACTOR_FER
        else:
            extractor_type = EXTRACTOR_PAPER
        desc = {
            "kind": "style_extractor",
            "extractor_type": extractor_type,
            "args": cfg,
            "run_dir": run_dir,
            "provenance_source": "inferred_from_config_json (legacy checkpoint)"
            if cfg is not None else "none (legacy checkpoint without config.json)",
        }

    desc["checkpoint_path"] = os.path.abspath(ckpt_path)
    desc["checkpoint_epoch"] = ckpt.get("epoch")
    desc["checkpoint_sha256"] = sha256_file(ckpt_path)
    return desc


def read_latent_provenance(latent_dir: str) -> Optional[dict]:
    """latent ディレクトリの provenance.json を返す（無ければ None）"""
    path = os.path.join(latent_dir, PROVENANCE_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_latent_provenance(latent_dir: str, provenance: dict) -> None:
    os.makedirs(latent_dir, exist_ok=True)
    with open(os.path.join(latent_dir, PROVENANCE_FILE), "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)


def check_output_dir(latent_dir: str, identity: dict) -> None:
    """
    出力先に別の作成元の .pt が混ざらないことを確認する。
    identity: 同一の作成元とみなすためのキー（抽出器 sha256・mode・入力ディレクトリなど）。
    既存の provenance.json の該当キーが一致する場合だけ再実行（上書き）を許可する。
    """
    if not os.path.isdir(latent_dir):
        return
    has_pt = any(f.endswith(".pt") for f in os.listdir(latent_dir))
    if not has_pt:
        return
    existing = read_latent_provenance(latent_dir)
    if existing is None:
        raise RuntimeError(
            f"{latent_dir} には作成元の記録 ({PROVENANCE_FILE}) が無い .pt が既にあります。"
            f"作成元が混ざるのを防ぐため、新しい出力ディレクトリを指定してください。")
    mismatched = {k: (existing.get(k), v) for k, v in identity.items() if existing.get(k) != v}
    if mismatched:
        raise RuntimeError(
            f"{latent_dir} は別の作成元で作られた latent を含んでいます: {mismatched}\n"
            f"新しい出力ディレクトリを指定してください。")
