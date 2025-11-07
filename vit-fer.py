import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# ハイパーパラメータ
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', DEVICE)

# データ前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # FER2013は1ch画像もあるため
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# データセットパス
train_dir = '../dataset/fer2013/train'
test_dir = '../dataset/fer2013/val'

# ImageFolderでデータセット作成
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ViTモデル（torchvision）
model = models.vit_b_16(weights='IMAGENET1K_V1')
# 出力層を7クラスに変更
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 学習ループ
train_losses = []
test_accuracies = []

# データ保存場所
learning_process_path = "learning_process/"
model_paramater_path = "model/"

# フォルダがなければ作成
os.makedirs(learning_process_path, exist_ok=True)
os.makedirs(model_paramater_path, exist_ok=True)

def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        avg_loss = total_loss/len(train_loader)
        train_losses.append(avg_loss)
        # 各エポックごとにテスト精度を計算
        acc = test(epoch)
        test_accuracies.append(acc)
        print(f'Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}, Test Accuracy: {acc:.2f}%')

    # モデル保存
    model_save_path = os.path.join(model_paramater_path, 'model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'モデルを {model_save_path} として保存しました')

    # 学習曲線の保存・表示
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Learning Curve')
    curve_save_path = os.path.join(learning_process_path, 'loss_acc.png')
    plt.savefig(curve_save_path)
    # plt.show()
    print(f'学習曲線を {curve_save_path} として保存しました')

# テストループ
def test(epoch=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    if epoch is None:
        print(f'Test Accuracy: {acc:.2f}%')
    return acc

if __name__ == '__main__':
    train()
    # 最終テスト精度も表示
    test()
