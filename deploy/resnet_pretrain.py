import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import h5py
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torch import nn
from torchvision.models import resnet50
import os

# SimCLR图像增强
class SimCLRTransform:
    def __init__(self):
        self.base_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

# Dataset改成返回多相机视角图片对（每个视角两个增强图）
class MultiCameraHDF5Dataset(Dataset):
    def __init__(self, hdf5_path, cameras=['left_wrist', 'right_wrist', 'top'], transform=None):
        self.hdf5_path = hdf5_path
        self.cameras = cameras
        self.transform = transform

        # 先打开文件获取长度，默认所有相机视角长度相同
        with h5py.File(hdf5_path, 'r') as f:
            self.length = f[f'observations/images/{self.cameras[0]}'].shape[0]

    def __len__(self):
        return self.length

    def _load_image(self, f, cam, idx):
        img = f[f'observations/images/{cam}'][idx]
        if img.ndim == 3 and img.shape[2] == 3:
            pass
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        img = Image.fromarray(img)
        return img

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            # 针对每个相机，获得两个增强图
            views = []
            for cam in self.cameras:
                img = self._load_image(f, cam, idx)
                x1, x2 = self.transform(img)
                views.append((x1, x2))
        # 返回格式：[(x1_cam1, x2_cam1), (x1_cam2, x2_cam2), ...]
        return views

# SimCLR模型和投影头（保持不变）
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.projector = ProjectionHead(2048, 128)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

# NT-Xent损失保持不变
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature

    batch_size = z1.size(0)
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim = sim[~mask].view(sim.shape[0], -1)

    loss = -torch.log(torch.exp(sim) / torch.exp(sim).sum(dim=1, keepdim=True))
    loss = (loss * labels).sum(dim=1).mean()
    return loss

# 单个文件训练，支持多相机
def train_on_single_file_multi_camera(model, optimizer, data_path, transform, batch_size, device, cameras):
    dataset = MultiCameraHDF5Dataset(data_path, cameras=cameras, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model.train()
    total_loss = 0
    for batch in loader:
        # batch是 list，每个元素是list: [(x1_cam1,x2_cam1), (x1_cam2,x2_cam2), ...]
        # 转成batch的格式：对每个视角分别stack成张量
        batch_size_actual = len(batch)
        cam_num = len(cameras)
        # 每个视角两张图，shape (B, C, H, W)
        views_x1 = [torch.stack([batch[i][cam][0] for i in range(batch_size_actual)]).to(device) for cam in range(cam_num)]
        views_x2 = [torch.stack([batch[i][cam][1] for i in range(batch_size_actual)]).to(device) for cam in range(cam_num)]

        # 多相机loss求平均
        loss = 0
        for x1, x2 in zip(views_x1, views_x2):
            z1, z2 = model(x1), model(x2)
            loss += nt_xent_loss(z1, z2)
        loss /= cam_num

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    return avg_loss

def train_simclr_multi_files_multi_cameras(hdf5_files, save_path, epochs=100, batch_size=8, lr=1e-3, device='cuda', cameras=['left_wrist', 'right_wrist', 'top'], pretrained_path=None):
    transform = SimCLRTransform()
    model = SimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.encoder.load_state_dict(torch.load(pretrained_path))

    for epoch in range(epochs):
        epoch_loss = 0
        for file in hdf5_files:
            print(f"Epoch {epoch+1} training on file {file}")
            loss = train_on_single_file_multi_camera(model, optimizer, file, transform, batch_size, device, cameras)
            print(f"File {file} Loss: {loss:.4f}")
            epoch_loss += loss
        avg_loss = epoch_loss / len(hdf5_files)
        print(f"Epoch {epoch+1} average Loss: {avg_loss:.4f}")

        # 构造带 loss 的文件名
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        save_dir = os.path.dirname(save_path)
        save_name = f"{base_name}_epoch{epoch+1:03d}_loss{avg_loss:.4f}.pth"
        full_save_path = os.path.join(save_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.encoder.state_dict(), full_save_path)
        print(f"Model saved to {full_save_path}")
def get_all_hdf5_files(folder_path):
    hdf5_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.hdf5'):
                full_path = os.path.join(root, file)
                hdf5_files.append(full_path)
    return hdf5_files

if __name__ == "__main__":
    folder = '/workspace/exchange/5-9/duikong'#HDF5路径
    hdf5_files = get_all_hdf5_files(folder)
    cameras = ['left_wrist',  'top']#相机数量

    train_simclr_multi_files_multi_cameras(#训练参数
        hdf5_files=hdf5_files,
        save_path='saved_models/simclr_encoder.pth',
        epochs=100,
        batch_size=8,
        lr=1e-3,
        device='cuda',
        cameras=cameras,
        pretrained_path=None
    )
