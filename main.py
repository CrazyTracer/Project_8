import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
from lime import lime_image

# -----------------------
# Конфигурация
# -----------------------
DATA_DIR = "data/afsis"
IMG_DIR = os.path.join(DATA_DIR, "images")
TARGET = "pH"     # можно заменить на N, P или K
ID_COL = "uid"
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

# -----------------------
# Загрузка данных
# -----------------------
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

X_tab = df.drop(columns=[TARGET])
y = df[TARGET].values

X_train_tab, X_val_tab, y_train, y_val = train_test_split(
    X_tab, y, test_size=0.2, random_state=RANDOM_STATE
)

num_cols = X_train_tab.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train_tab_scaled = scaler.fit_transform(X_train_tab[num_cols])
X_val_tab_scaled = scaler.transform(X_val_tab[num_cols])

X_train_tab_t = torch.tensor(X_train_tab_scaled, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
X_val_tab_t = torch.tensor(X_val_tab_scaled, dtype=torch.float32).to(DEVICE)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)

# -----------------------
# Модель табличных данных (FT-Transformer — упрощённый вариант)
# -----------------------
class FTTransformer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model_tab = FTTransformer(X_train_tab_scaled.shape[1]).to(DEVICE)
opt_tab = optim.Adam(model_tab.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -----------------------
# Обучение табличной модели
# -----------------------
for epoch in range(EPOCHS):
    model_tab.train()
    opt_tab.zero_grad()
    preds = model_tab(X_train_tab_t)
    loss = loss_fn(preds, y_train_t)
    loss.backward()
    opt_tab.step()
    print(f"[Tabular] Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

# -----------------------
# Dataset для изображений
# -----------------------
class SoilImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.df.loc[idx, ID_COL]
        img_path = os.path.join(self.img_dir, f"{uid}.jpg")
        img = Image.open(img_path).convert("RGB")
        y = self.df.loc[idx, TARGET]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_img_ds = SoilImageDataset(df, IMG_DIR, transform)
train_img_loader = DataLoader(train_img_ds, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------
# Модель изображений (ResNet18)
# -----------------------
model_img = models.resnet18(weights="IMAGENET1K_V1")
model_img.fc = nn.Linear(model_img.fc.in_features, 1)
model_img = model_img.to(DEVICE)
opt_img = optim.Adam(model_img.parameters(), lr=1e-4)

# -----------------------
# Обучение CNN
# -----------------------
for epoch in range(EPOCHS):
    model_img.train()
    epoch_loss = 0
    for imgs, targets in train_img_loader:
        imgs, targets = imgs.to(DEVICE), targets.view(-1,1).to(DEVICE)
        opt_img.zero_grad()
        preds = model_img(imgs)
        loss = loss_fn(preds, targets)
        loss.backward()
        opt_img.step()
        epoch_loss += loss.item()
    print(f"[Images] Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_img_loader):.4f}")

# -----------------------
# Мультимодальная модель
# -----------------------
class MultiModalModel(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Identity()

        self.tabular = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img)
        tab_feat = self.tabular(tab)
        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(x)

model_mm = MultiModalModel(X_train_tab_scaled.shape[1]).to(DEVICE)
opt_mm = optim.Adam(model_mm.parameters(), lr=1e-4)

# -----------------------
# Метрики
# -----------------------
def eval_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

# -----------------------
# SHAP для табличной модели
# -----------------------
model_tab_cpu = model_tab.cpu().eval()
explainer = shap.Explainer(model_tab_cpu, X_train_tab_scaled[:200])
shap_values = explainer(X_val_tab_scaled[:50])
shap.summary_plot(shap_values, X_val_tab[num_cols].iloc[:50])

# -----------------------
# LIME для изображений (пример)
# -----------------------
explainer_img = lime_image.LimeImageExplainer()
sample_img_path = os.path.join(IMG_DIR, f\"{df.iloc[0][ID_COL]}.jpg\")
sample_img = np.array(Image.open(sample_img_path).resize((224,224)))

def predict_img(batch):
    batch_t = torch.tensor(batch).permute(0,3,1,2).float().to(DEVICE)
    with torch.no_grad():
        preds = model_img(batch_t).cpu().numpy()
    return preds

explanation = explainer_img.explain_instance(
    sample_img,
    classifier_fn=predict_img,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

print(\"Готово. Модели обучены, SHAP и LIME рассчитаны.\")
