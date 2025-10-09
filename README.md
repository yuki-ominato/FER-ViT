# FER-ViT
ViTによる感情認識モデル開発

# 潜在コード生成コマンド
(fer-vit) yuki@DESKTOP-8Q4U3NL:~/research2/fer-vit$ python scripts/generate_latents.py     --data_root /home/yuki/research2/dataset/fer2013/train_smoke     --latent_out /home/yuki/research2/fer-vit/latents/train_smoke     --encoder_model /home/yuki/research2/fer-vit/pretrained_models/psp_ffhq_encode.pt     --encoder_type psp     --batch_size 2
