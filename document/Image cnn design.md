# 2D Image CNN ベースライン 実装設計書

## 0. このドキュメントについて

提案手法(Latent + ViT)とのベースライン比較に追加する「2D Image CNN」の実装設計書。
本リポジトリ(FER-ViT)に存在する `ImageViT` / `LatentCNN` の実装パターンを踏襲しつつ、

- ImageNet事前学習 / スクラッチ学習をオプションで切り替え可能
- データセットを FER2013 / RAF-DB でオプションで切り替え可能

を満たすことを目的とする。実装はリポジトリ外のデータディレクトリ構造を参照できる環境(Cursor上のClaude Code)で行うことを前提とし、本書では「確定している設計」と「実装前に実データを見て確認・確定すべき事項」を明確に分離して記載する。

---

## 1. 背景・目的（再掲）

比較対象は以下の4手法。

| # | 手法 | 入力表現 | 事前学習 |
|---|------|---------|---------|
| 1 | PreTrained ImageViT | 画像(pixel) | ImageNet |
| 2 | Scratch ImageViT | 画像(pixel) | なし |
| 3 | 1D LatentCNN | StyleGAN W+ latent | なし(latent空間にImageNet事前学習は存在しないため) |
| 4 | **2D Image CNN（新規）** | 画像(pixel) | **オプションで切替** |

目的は「latent表現 vs pixel表現」「事前学習あり vs なし」という軸を、アーキテクチャの強さ・学習テクニックの差に邪魔されずに比較すること。そのため2D Image CNNは、FER分野のSOTA(アンサンブル・特殊データ拡張等を含む)を目指すのではなく、**広く使われ妥当性が確認しやすい標準的なCNN(ResNetファミリー)** を採用する。

事前学習あり/なしを公平に比較するには、**事前学習済み重みが実在するアーキテクチャ**を使う必要がある。そのため、`LatentCNN`のような自作の小型ResBlockではなく、torchvisionが提供する標準ResNet実装（ImageNet学習済み重みあり）をバックボーンとして採用する。

---

## 2. 全体方針

```
data/
  image_dataset.py         既存: FER2013用 ImageFERDataset（流用）
  raf_db_dataset.py         新規: RAF-DB用 Dataset
  dataset_factory.py        新規: dataset名 → Dataset を返すファクトリ

models_fer_vit/
  image_cnn.py               新規: ImageCNN2D（torchvision ResNetラッパー）

train/
  train_image_cnn.py         新規: 学習スクリプト（train_image_vit.pyを踏襲）

eval/
  evaluate_image_cnn.py      新規: 評価スクリプト（evaluate_image_vit.pyを踏襲）

utils/
  experiment_logger.py       既存に軽微な変更（model_type引数追加）
```

既存の `ImageViT` 系パイプラインと**学習ループ・ログ仕様・チェックポイント形式を完全に統一**することで、4手法の比較表が成立するようにする。具体的には `train_loss/train_acc/train_f1/val_loss/val_acc/val_f1` の6指標、`ExperimentLogger`によるTensorBoardログ・チェックポイント保存・混同行列出力の流儀を完全に踏襲する。

---

## 3. データセット設計

### 3.1 共通インターフェース

`data/image_dataset.py` の `ImageFERDataset` と同じ「契約」をRAF-DB側も満たす。

- `__getitem__` は `(image: Tensor[C,H,W], label: int)` を返す
- ラベルIDは以下に統一（既存 `ImageFERDataset.CLASS_TO_LABEL` と同一にする。これにより7クラスの並びがFER2013/RAF-DBで揃い、混同行列やF1計算をそのまま使い回せる）

```python
CLASS_TO_LABEL = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
    "neutral": 4, "sad": 5, "surprise": 6,
}
```

- 変換(transform)は既存の `get_train_transforms(img_size)` / `get_val_transforms(img_size)`（`data/image_dataset.py`内）をそのまま再利用し、新規実装しない。これにより画像正規化・データ拡張の条件もFER2013/RAF-DB/ImageViTで完全に揃う。

### 3.2 FER2013

既存の `ImageFERDataset` をそのまま利用する。変更不要。

### 3.3 RAF-DB（新規実装が必要・要確認事項あり）

RAF-DBの一般的な配布形式は以下だが、**実際にユーザー環境にダウンロードされているディレクトリ構造・ファイル名規則は本書執筆時点で未確認**。Cursor環境で実ディレクトリを確認したうえで、以下の仮設計を実データに合わせて調整すること。

**仮定する構造（一般的なRAF-DB basic emotionの配布形式）:**
```
raf_db_root/
  Image/aligned/
    train_00001_aligned.jpg
    test_0001_aligned.jpg
    ...
  EmoLabel/
    list_patition_label.txt   # 例: "train_00001.jpg 5"
```

**ラベル対応（要確認）:** RAF-DBのbasic 7感情は数値ラベル1〜7で配布されることが多く、一般的には
`1:Surprise 2:Fear 3:Disgust 4:Happiness 5:Sadness 6:Anger 7:Neutral` という割当てが使われる。これを §3.1 の `CLASS_TO_LABEL` （0-indexed, angry=0...）へ変換するマッピングテーブルを実装するが、**実際のラベルファイルの中身を確認してから数値対応を確定すること**（配布元やバージョンにより並びが異なる場合があるため、思い込みで実装しない）。

**実装すべきクラス:** `data/raf_db_dataset.py` に `RafDbDataset(Dataset)` を実装。コンストラクタ引数は `ImageFERDataset` と揃える（`data_root`, `transform`, `img_size`）。加えて `split: Literal["train","test"]` を必須引数とする（RAF-DBはtrain/test両方が単一ディレクトリ配下にまとまっている配布が多いため）。

**実装前にCursor側で確認すべき事項（チェックリスト）:**
1. RAF-DBの実体パスとディレクトリ階層
2. ラベルファイルの実フォーマット（区切り文字、ファイル名にtrain/test接頭辞があるか、拡張子）
3. 数値ラベル→感情名の実際の対応関係（README等の付属ドキュメントで確認）
4. 画像が顔アライメント済みか（`aligned`画像を使うか、未加工画像を使うか）
5. basic emotion(7クラス)のみ使うか、compound emotion(11クラス等)のサブセットが混在していないか
6. train/val/testの分割方法（RAF-DBは公式にはtrain/testのみでvalがない場合が多く、その場合はtrainを内部分割してvalを作る必要がある。FER2013側の分割方針——train/val/testをどう作っているか——と揃えること）

これらが未確認のままコーディングを始めると、ラベル対応の誤りに気づかないまま実験が進むリスクが高い。**実装の最初のステップとして、実ディレクトリを `ls`/`head` 等で確認し、上記6項目を設計書に追記してから実装に入ること。**

### 3.4 データセットファクトリ

`data/dataset_factory.py` に以下を新規実装し、学習スクリプトからはこの関数経由でのみデータセットを取得する（学習スクリプト内に分岐ロジックを直書きしない）。

```python
def get_image_datasets(
    dataset_name: str,        # "fer2013" | "rafdb"
    data_root: str,
    img_size: int,
    use_augmentation: bool,
) -> tuple[Dataset, Dataset]:  # (train_ds, val_ds)
    ...
```

`data_root` 配下の構造はデータセットごとに異なるため、内部で `dataset_name` に応じて `ImageFERDataset(train/val 別ディレクトリ)` または `RafDbDataset(split指定)` を呼び分ける。

---

## 4. モデル設計

### 4.1 アーキテクチャ選定

- バックボーン: `torchvision.models.resnet18`（デフォルト）。`resnet34` / `resnet50` も `--backbone` 引数で選択可能にし、パラメータ規模を他モデル（ImageViT-small: 約22M、LatentCNN系）に近づけたい場合は `resnet34`（約21.8M）を選べるようにしておく。
- SOTAを狙ったアーキテクチャ探索（EfficientNet, ConvNeXt等）はスコープ外。理由は前回回答のとおり、「latent vs pixel」の比較軸を汚染しないため。
- 入力は3チャンネルRGB・`img_size=224`に統一する。`ImageFERDataset`は`.convert('RGB')`によりFER2013のグレースケール画像も3chに揃えているため、RAF-DB（元々RGB）と入力仕様の差異は生じない。

### 4.2 クラス設計（`models_fer_vit/image_cnn.py`）

```python
class ImageCNN2D(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",      # "resnet18" | "resnet34" | "resnet50"
        pretrained: bool = False,
        num_classes: int = 7,
        dropout: float = 0.0,             # 必要なら fc 直前に Dropout を挿入
    ):
        """
        torchvision の ResNet を読み込み、最終 fc 層を num_classes 用に置換する。
        pretrained=True の場合は ImageNet1K の重みをロードしたうえで、
        fc 層のみランダム初期化（＝全層ファインチューニング前提。
        train_hybrid_latent_vit.py の Adapter/Freeze 戦略のような
        部分凍結オプションは本設計のスコープ外とし、まずは
        ImageViTのuse_pretrained実装と同等の「フルファインチューニング」で揃える）
        """
```

- `pretrained=True/False` の切り替えは `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)` のように実装する（torchvisionのバージョンによってAPIが`pretrained=bool`の旧式と`weights=Enum`の新式に分かれるため、`requirements.txt`記載の`torchvision>=0.15.0`であれば新式APIが使用可能。実装時にインストール済みバージョンを確認すること）。
- `forward(x) -> logits (B, num_classes)` のシンプルなインターフェースとし、`ImageViT.forward`と同じ呼び出し方で `train_epoch`/`evaluate` 関数を共通化できるようにする。
- パラメータ数を `model_config['n_parameters']` としてログに残す点も既存踏襲（`train_image_vit.py`と同様に `sum(p.numel() for p in model.parameters() if p.requires_grad)`）。

---

## 5. 学習スクリプト設計（`train/train_image_cnn.py`）

`train/train_image_vit.py` の構造をそのまま踏襲する。差分のみ記載。

### 5.1 追加・変更するCLI引数

| 引数 | 型/選択肢 | 説明 |
|---|---|---|
| `--dataset` | `{fer2013, rafdb}` | 使用データセット（必須） |
| `--data_root` | str | データセットルート（`dataset_factory.get_image_datasets`に渡す。FER2013の場合は既存の`--train_dir`/`--val_dir`方式と非互換になるため、後方互換の扱いを5.2に記載） |
| `--backbone` | `{resnet18, resnet34, resnet50}` | デフォルト `resnet18` |
| `--use_pretrained` | flag | ImageNet事前学習重みを使うか（既存`ImageViT`の引数名と統一） |
| `--img_size` | int | デフォルト224（既存と同様） |

`--model_size`, `--patch_size`, `--embed_dim`, `--depth`, `--heads`, `--mlp_dim` 等のViT固有引数は不要なので削除する。`--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--optimizer`, `--scheduler`, `--label_smoothing`, `--grad_clip`, `--use_class_weights`, `--use_augmentation`, `--data_fraction`, `--seed`, `--num_workers`は既存と同名・同意味で流用し、比較表のハイパーパラメータ列を揃えやすくする。

### 5.2 `--train_dir`/`--val_dir` との関係（要設計判断）

既存の`ImageFERDataset`は「train用ディレクトリ」「val用ディレクトリ」を別々に受け取る設計（クラス名サブディレクトリ構成）。RAF-DBは単一ルート＋split引数で扱う想定（§3.3）。この非対称性を吸収するため、`train_image_cnn.py`では

- `--dataset fer2013` のとき: `--train_dir` / `--val_dir` を必須とする（既存と同じ使い方）
- `--dataset rafdb` のとき: `--data_root` を必須とし、内部で train/val split を行う

という分岐をCLI引数のバリデーションとして実装する。`dataset_factory.get_image_datasets`のシグネチャは§3.4の通りシンプルに保ちつつ、`train_image_cnn.py`側でデータセットごとの引数解釈を吸収する。

### 5.3 学習ループ

`train_epoch` / `evaluate` 関数は `train_image_vit.py` のものをほぼそのまま再利用可能（モデルのforwardシグネチャが同じため）。`set_seed`, `create_subset_dataset`, `calculate_class_weights` も流用する（`dataset.samples`属性に依存しているため、`RafDbDataset`も`self.samples = [(path, label), ...]`の形式を必ず持たせること。これは§3.3の実装要件に追記する）。

### 5.4 実験ログとの連携（既存への小さな変更が必要）

`utils/experiment_logger.py` の `create_experiment_name()` は現状 `is_latent: bool` のみでモデル名を `latent_vit_...` / `image_vit_...` に固定生成しており、`image_cnn`用の分岐がない。以下のいずれかの対応が必要：

- (推奨) `create_experiment_name`に`model_type: str = None`引数を追加し、指定があればそれを優先してモデル名プレフィックスに使う（後方互換: 既存呼び出し元は無改修で動く）
- `train_image_cnn.py`内で`create_experiment_name`を呼ばず、独自に実験名を組み立てる（簡易だが命名規則が分裂するため非推奨）

`model_config`/`training_config`/`data_config`の3分割構成、`logger.log_config`, `logger.log_metrics`, `logger.log_confusion_matrix`, `logger.log_experiment_summary`の呼び出し順序は既存と完全に揃える。`data_config`には`dataset_name`, `data_root`(またはtrain_dir/val_dir), `train_samples`, `val_samples`を含める。`model_config`には`backbone`, `use_pretrained`, `n_parameters`を含める。

---

## 6. 評価スクリプト設計（`eval/evaluate_image_cnn.py`）

`eval/evaluate_image_vit.py`を踏襲し、`load_model`内のモデル復元部分のみ`ImageCNN2D`に差し替える。チェックポイントの`config['model']`から`backbone`/`use_pretrained`/`num_classes`を読み出して再構築する。

---

## 7. 受け入れ条件

- [ ] `--dataset fer2013 --use_pretrained` / `--dataset fer2013`（スクラッチ）/ `--dataset rafdb --use_pretrained` / `--dataset rafdb`（スクラッチ）の4パターンが同一スクリプトで実行できる
- [ ] 4パターンとも`ExperimentLogger`に`train/val`の`loss/acc/f1`が記録され、既存3手法（PreTrained/Scratch ImageViT, LatentCNN）の実験ログと同じ形式で並べて比較できる
- [ ] チェックポイントから`evaluate_image_cnn.py`でF1・混同行列が再現できる
- [ ] FER2013/RAF-DBのクラスラベルIDが完全に一致しており（angry=0...surprise=6）、`classification_report`の`target_names`がそのまま使い回せる
- [ ] パラメータ数(`n_parameters`)が他の3手法と概ね近い桁数で記録され、比較表の「パラメータ効率」列に転記可能

---

## 8. 実装に着手する前に最初に行うべきこと（Cursor側タスク）

1. ユーザー環境のRAF-DB実データを `ls -R`等で確認し、§3.3のディレクトリ構造仮定・ラベル対応表を実データに合わせて修正・確定する
2. 既存FER2013の`--train_dir`/`--val_dir`が実際にどのパスを指しているか（`data/generate_latents.py`等で使われているパスと整合するか）を確認する
3. インストール済み`torchvision`のバージョンを確認し、`weights=` APIか`pretrained=`旧APIかを確定する
4. 上記が確定した時点で、本設計書の該当セクション（特に§3.3, §5.2）を更新してから実装に入る