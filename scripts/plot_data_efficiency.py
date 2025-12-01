import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_experiments(experiments_dir):
    data = []
    
    # experiment_summary.json を検索
    summary_files = glob.glob(os.path.join(experiments_dir, "fraction/experiment_summary.json"), recursive=True)
    
    for f in summary_files:
        try:
            with open(f, 'r') as file:
                summary = json.load(file)
            
            config = summary.get('config', {})
            training_config = config.get('training', {})
            model_config = config.get('model', {})
            metrics = summary.get('final_metrics', {})
            
            # モデルタイプの判別 (簡易的なロジック)
            if 'latent' in summary.get('experiment_name', ''):
                model_type = "Latent ViT (Proposed)"
            else:
                model_type = "Image ViT (Baseline)"
            
            data.append({
                'Model': model_type,
                'Data Fraction': training_config.get('data_fraction', 1.0) * 100, # %表記に
                'Accuracy': metrics.get('accuracy', 0),
                'F1 Score': metrics.get('f1_macro', 0),
                'Experiment': summary.get('experiment_name', '')
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.DataFrame(data)

def main(args):
    df = parse_experiments(args.experiments_dir)
    
    if df.empty:
        print("No experiment data found.")
        return

    # グラフ描画設定
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # プロット作成 (Accuracy)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Data Fraction', y='Accuracy', hue='Model', marker='o', linewidth=2.5)
    plt.title('Data Efficiency: Accuracy vs Data Amount')
    plt.xlabel('Training Data (%)')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, 'data_efficiency_accuracy.png'), dpi=300)
    print(f"Saved: {os.path.join(args.output_dir, 'data_efficiency_accuracy.png')}")
    
    # プロット作成 (F1 Score)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Data Fraction', y='F1 Score', hue='Model', marker='s', linewidth=2.5)
    plt.title('Data Efficiency: F1 Score vs Data Amount')
    plt.xlabel('Training Data (%)')
    plt.ylabel('Test F1 Score (Macro)')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'data_efficiency_f1.png'), dpi=300)
    print(f"Saved: {os.path.join(args.output_dir, 'data_efficiency_f1.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="experiments", help="Directory containing experiment logs")
    parser.add_argument("--output_dir", default="figures", help="Directory to save plots")
    args = parser.parse_args()
    main(args)