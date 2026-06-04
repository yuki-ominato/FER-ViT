import matplotlib.pyplot as plt
import numpy as np

# データの定義 (画像から読み取った概算値です。正確な数値に置き換えてください)
fractions = [10, 25, 50, 100]
acc_latent = [0.40, 0.45, 0.50, 0.54]     # Blue: Latent ViT (Proposed)
acc_cnn = [0.14, 0.38, 0.44, 0.48]        # Grey: Latent CNN
acc_scratch = [0.23, 0.30, 0.36, 0.46]    # Orange: Scratch Image ViT
acc_pretrained = [0.47, 0.58, 0.66, 0.70] # Yellow: Pre-trained Image ViT

plt.figure(figsize=(10, 6))

# プロット
plt.plot(fractions, acc_pretrained, 'o-', color='#f1c40f', label='Image ViT (Pre-trained on ImageNet)', linewidth=2, markersize=8)
plt.plot(fractions, acc_latent, 'o-', color='#2980b9', label='Latent ViT (Proposed)', linewidth=3, markersize=8) # 提案手法を太く
plt.plot(fractions, acc_cnn, 's--', color='#7f8c8d', label='Latent CNN', linewidth=2, markersize=8)
plt.plot(fractions, acc_scratch, '^--', color='#e67e22', label='Image ViT (Scratch)', linewidth=2, markersize=8)

# 装飾
# plt.title('Data Efficiency Comparison: Accuracy vs Training Data Fraction', fontsize=14)
plt.xlabel('Training Data Fraction (%)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.ylim(0, 0.8)
plt.xticks(fractions, [f'{x}%' for x in fractions])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

# 強調したいポイントに矢印を入れる（オプション）
# plt.annotate('Robust with 10% Data', xy=(10, 0.40), xytext=(20, 0.30),
#              arrowprops=dict(facecolor='black', shrink=0.1))

plt.tight_layout()
plt.savefig('data_efficiency_final.png', dpi=300)
plt.show()