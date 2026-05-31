import sys
import os
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

# 创建输出目录
os.makedirs('outputs', exist_ok=True)

# 定义网络
class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = x.view(x.size(0), 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)
    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out)
        out = out.view(out.size(0), 1, 28, 28)
        return out

# 加载数据
print("Loading Fashion MNIST dataset...")

# 尝试多个下载源
import urllib.request
import gzip
import os

def download_fashion_mnist(data_root='./FashionMNIST/FashionMNIST'):
    """下载Fashion MNIST数据集"""
    os.makedirs(os.path.join(data_root, 'raw'), exist_ok=True)

    # 使用国内镜像源
    urls = [
        'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
    ]
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    for filename in files:
        filepath = os.path.join(data_root, 'raw', filename.replace('.gz', ''))
        if os.path.exists(filepath):
            print(f"  {filename} already exists, skipping")
            continue

        gz_path = os.path.join(data_root, 'raw', filename)
        if not os.path.exists(gz_path):
            downloaded = False
            for base_url in urls:
                print(f"  Downloading {filename} from {base_url}...")
                try:
                    urllib.request.urlretrieve(base_url + filename, gz_path)
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  Failed: {e}")
            if not downloaded:
                return False

        # 解压
        print(f"  Extracting {filename}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(filepath, 'wb') as f_out:
                f_out.write(f_in.read())

    return True

if not download_fashion_mnist():
    print("Failed to download Fashion MNIST. Using MNIST instead...")
    dataset = torchvision.datasets.MNIST(root='./MNIST/',
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.5,), (0.5,))]),
                           download=True)
else:
    dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.5,), (0.5,))]),
                           download=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
print(f"Dataset loaded: {len(dataset)} samples")

def train_gan(epochs=10, lr_d=0.0002, lr_g=0.0002, optimizer_type='Adam', experiment_name='exp1'):
    """训练GAN并记录损失和生成图像"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'\n{"="*60}')
    print(f'Experiment: {experiment_name}')
    print(f'Optimizer: {optimizer_type}, LR_D: {lr_d}, LR_G: {lr_g}')
    print(f'Device: {device}')
    print(f'{"="*60}')

    # 初始化网络
    D = Discriminator().to(device)
    G = Generator().to(device)

    # 设置优化器
    if optimizer_type == 'Adam':
        optimizerD = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    else:  # SGD
        optimizerD = torch.optim.SGD(D.parameters(), lr=lr_d)
        optimizerG = torch.optim.SGD(G.parameters(), lr=lr_g)

    criterion = nn.BCELoss()

    # 固定噪声用于可视化
    fixed_noise = torch.randn(64, 100, device=device)

    # 记录损失
    G_losses = []
    D_losses = []
    D_x_history = []
    D_G_z_history = []

    # 保存生成图像
    img_list = []

    print('Starting Training...')
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) 更新判别器 D
            ###########################
            D.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=device)

            output = D(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, 100, device=device)
            fake = G(noise)
            label.fill_(0.)
            output = D(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) 更新生成器 G
            ###########################
            G.zero_grad()
            label.fill_(1.)
            output = D(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # 记录损失
            if i % 50 == 0:
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                D_x_history.append(D_x)
                D_G_z_history.append(D_G_z2)

            if i % 200 == 0:
                print(f'[{epoch+1}/{epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

        # 每个epoch结束后保存生成图像
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, nrow=8, normalize=True))
        print(f'Epoch {epoch+1}/{epochs} completed')

    print(f'Training complete for {experiment_name}')
    return G_losses, D_losses, D_x_history, D_G_z_history, img_list, G, D

# 运行实验1：Adam优化器，lr=0.0002
print("\n" + "="*60)
print("EXPERIMENT 1: Adam optimizer, lr=0.0002")
print("="*60)
G_losses1, D_losses1, D_x1, D_G_z1, img_list1, G1, D1 = train_gan(
    epochs=10, lr_d=0.0002, lr_g=0.0002, optimizer_type='Adam', experiment_name='Adam_lr0.0002'
)

# 保存生成图像
for idx, img in enumerate(img_list1):
    vutils.save_image(img, f'outputs/adam_lr0.0002_epoch{idx}.png')
print("Saved Adam lr=0.0002 results")

# 运行实验2：SGD优化器，lr=0.03
print("\n" + "="*60)
print("EXPERIMENT 2: SGD optimizer, lr=0.03")
print("="*60)
G_losses2, D_losses2, D_x2, D_G_z2, img_list2, G2, D2 = train_gan(
    epochs=10, lr_d=0.03, lr_g=0.03, optimizer_type='SGD', experiment_name='SGD_lr0.03'
)

# 保存生成图像
for idx, img in enumerate(img_list2):
    vutils.save_image(img, f'outputs/sgd_lr0.03_epoch{idx}.png')
print("Saved SGD lr=0.03 results")

# 运行实验3：Adam优化器，lr=0.001
print("\n" + "="*60)
print("EXPERIMENT 3: Adam optimizer, lr=0.001")
print("="*60)
G_losses3, D_losses3, D_x3, D_G_z3, img_list3, G3, D3 = train_gan(
    epochs=10, lr_d=0.001, lr_g=0.001, optimizer_type='Adam', experiment_name='Adam_lr0.001'
)

# 保存生成图像
for idx, img in enumerate(img_list3):
    vutils.save_image(img, f'outputs/adam_lr0.001_epoch{idx}.png')
print("Saved Adam lr=0.001 results")

# 可视化结果
print("\n" + "="*60)
print("Generating visualization...")
print("="*60)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 绘制损失曲线对比
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(G_losses1, label='G_loss (Adam lr=0.0002)', alpha=0.7)
ax1.plot(D_losses1, label='D_loss (Adam lr=0.0002)', alpha=0.7)
ax1.plot(G_losses2, label='G_loss (SGD lr=0.03)', alpha=0.7)
ax1.plot(D_losses2, label='D_loss (SGD lr=0.03)', alpha=0.7)
ax1.plot(G_losses3, label='G_loss (Adam lr=0.001)', alpha=0.7)
ax1.plot(D_losses3, label='D_loss (Adam lr=0.001)', alpha=0.7)
ax1.set_xlabel('Iterations (x50)')
ax1.set_ylabel('Loss')
ax1.set_title('Generator and Discriminator Loss During Training')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 绘制D(x)和D(G(z))对比
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(D_x1, label='D(x) Adam lr=0.0002', alpha=0.7)
ax2.plot(D_G_z1, label='D(G(z)) Adam lr=0.0002', alpha=0.7)
ax2.plot(D_x2, label='D(x) SGD lr=0.03', alpha=0.7)
ax2.plot(D_G_z2, label='D(G(z)) SGD lr=0.03', alpha=0.7)
ax2.plot(D_x3, label='D(x) Adam lr=0.001', alpha=0.7)
ax2.plot(D_G_z3, label='D(G(z)) Adam lr=0.001', alpha=0.7)
ax2.set_xlabel('Iterations (x50)')
ax2.set_ylabel('Probability')
ax2.set_title('D(x) and D(G(z)) During Training')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 显示最后一个epoch的生成图像
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(np.transpose(img_list1[-1], (1, 2, 0)))
ax3.set_title('Adam lr=0.0002 (Final)')
ax3.axis('off')

ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(np.transpose(img_list2[-1], (1, 2, 0)))
ax4.set_title('SGD lr=0.03 (Final)')
ax4.axis('off')

ax5 = fig.add_subplot(gs[1, 2])
ax5.imshow(np.transpose(img_list3[-1], (1, 2, 0)))
ax5.set_title('Adam lr=0.001 (Final)')
ax5.axis('off')

plt.savefig('outputs/experiment_comparison.png', dpi=150, bbox_inches='tight')
print("Saved experiment_comparison.png")

# 显示Adam lr=0.0002实验在不同epoch的生成结果
fig2, axes = plt.subplots(2, 5, figsize=(20, 8))
fig2.suptitle('Generated Images at Different Epochs (Adam lr=0.0002)', fontsize=16)

for idx, ax in enumerate(axes.flat):
    if idx < len(img_list1):
        ax.imshow(np.transpose(img_list1[idx], (1, 2, 0)))
        ax.set_title(f'Epoch {idx+1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/epoch_progression.png', dpi=150, bbox_inches='tight')
print("Saved epoch_progression.png")

# 保存损失数据到文件
np.savez('outputs/loss_data.npz',
         G_losses1=G_losses1, D_losses1=D_losses1,
         G_losses2=G_losses2, D_losses2=D_losses2,
         G_losses3=G_losses3, D_losses3=D_losses3,
         D_x1=D_x1, D_G_z1=D_G_z1,
         D_x2=D_x2, D_G_z2=D_G_z2,
         D_x3=D_x3, D_G_z3=D_G_z3)

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETED!")
print("Results saved to outputs/ directory")
print("="*60)
