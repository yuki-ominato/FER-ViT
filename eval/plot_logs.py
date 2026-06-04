import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_learning_curves(file_paths, custom_labels=None):
    """
    複数のCSVファイルを読み込み、Accuracy vs Epochのグラフを描画する関数
    custom_labels: ファイルに対応する凡例名のリスト（オプション）
    """
    plt.figure(figsize=(10, 6))

    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"警告: ファイルが見つかりません: {file_path}")
            continue

        try:
            # CSVの読み込み
            df = pd.read_csv(file_path)

            # カラム名の前後の空白を除去
            df.columns = [col.strip() for col in df.columns]

            # 必要なカラムが存在するか確認
            if 'Step' not in df.columns or 'Value' not in df.columns:
                print(f"スキップ: {file_path} に 'Step' または 'Value' カラムが含まれていません。")
                continue

            # 凡例名の決定
            # ラベルが指定されており、かつ現在のインデックスに対応するラベルがある場合に使用
            if custom_labels and i < len(custom_labels):
                label_name = custom_labels[i]
            else:
                # ラベルがない場合はファイル名を使用
                label_name = os.path.basename(file_path)

            # グラフにプロット
            plt.plot(df['Step'], df['Value'], marker='.', label=label_name)

        except Exception as e:
            print(f"エラー: {file_path} の処理中にエラーが発生しました: {e}")

    # グラフの装飾
    # plt.title('Learning Curve (Accuracy vs Epoch)')
    plt.xlabel('Epoch (Step)')
    plt.ylabel('Accuracy')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CSV学習ログをグラフ化するツール（ラベル指定対応版）')
    
    # ファイルパスのリスト引数 (-f または --files)
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                        help='読み込むCSVファイルのパス（スペース区切り）')
    
    # 凡例ラベルのリスト引数 (-l または --labels)
    parser.add_argument('-l', '--labels', type=str, nargs='+',
                        help='グラフの凡例に表示する名前（ファイルの順序に対応・スペース区切り）')

    args = parser.parse_args()

    # ラベルの数がファイル数より少ない場合の警告（処理は続行）
    if args.labels and len(args.files) != len(args.labels):
        print(f"注意: ファイル数({len(args.files)})とラベル数({len(args.labels)})が一致していません。足りない部分はファイル名が使用されます。")

    plot_learning_curves(args.files, args.labels)