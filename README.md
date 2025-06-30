# 模仿学习算法和移动ALOHA的共训练

#### 项目网站: https://mobile-aloha.github.io/

本仓库包含ACT、扩散策略(Diffusion Policy)和VINN的实现，以及两个模拟环境：Transfer Cube和Bimanual Insertion。您可以在模拟环境或真实环境中训练和评估这些模型。对于真实环境，您还需要安装[Mobile ALOHA](https://github.com/MarkFzp/mobile-aloha)。该仓库是从[ACT仓库](https://github.com/tonyzhaozh/act)派生的。

### 更新：
您可以在[这里](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link)找到所有的模拟/人工演示数据。

### 仓库结构
- ``train.py`` 训练和评估ACT
- ``policy.py`` ACT策略的适配器
- ``detr`` ACT的模型定义，修改自DETR
- ``utils.py`` 数据加载和辅助函数等工具
- ``deploy/hdf5_file_edit_utils.py`` 从.hdf5数据集中保存视频等方法

### 环境安装
```
#安装git
apt-get install git
#克隆项目
git clone https://github.com/misaka-maples/ACT_plus_plus.git
# 安装依赖项
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython 
pip install diffusers
pip install wandb
pip install torch==2.0.1+cu118 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e #文件根目录
cd ./detr \
pip install -e .\
cd ../diffusion_policy\
pip install -e .\
cd ../egl_probe \
pip install -e .\
```

```
修改train.py中路径到episode.hdf5文件，训练 python train.py,出现进度条既配置完成，推理部署修改eval.py中ckpt文件路径和hdf5路径，运行python eval.py
```
