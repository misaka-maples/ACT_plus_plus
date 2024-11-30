# 模仿学习算法和移动ALOHA的共训练

#### 项目网站: https://mobile-aloha.github.io/

本仓库包含ACT、扩散策略(Diffusion Policy)和VINN的实现，以及两个模拟环境：Transfer Cube和Bimanual Insertion。您可以在模拟环境或真实环境中训练和评估这些模型。对于真实环境，您还需要安装[Mobile ALOHA](https://github.com/MarkFzp/mobile-aloha)。该仓库是从[ACT仓库](https://github.com/tonyzhaozh/act)派生的。

### 更新：
您可以在[这里](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link)找到所有的模拟/人工演示数据。

### 仓库结构
- ``imitate_episodes.py`` 训练和评估ACT
- ``policy.py`` ACT策略的适配器
- ``detr`` ACT的模型定义，修改自DETR
- ``sim_env.py`` 基于Mujoco和DM_Control的关节空间控制环境
- ``ee_sim_env.py`` 基于Mujoco和DM_Control的末端执行器(EE)空间控制环境
- ``scripted_policy.py`` 用于模拟环境的脚本化策略
- ``constants.py`` 在文件间共享的常量
- ``utils.py`` 数据加载和辅助函数等工具
- ``visualize_episodes.py`` 从.hdf5数据集中保存视频

### 软件安装

```
conda create -n aloha python=3.8.10
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install 

```

- 还需要安装[robomimic](https://github.com/ARISE-Initiative/robomimic/tree/r2d2)（注意r2d2分支），通过 `pip install -e .`
- 需要安装detr、mobile_aloha、robomimic，执行：`python setup.py install`

### 硬件安装

#### 启动相机流程

```bash
# 进入项目文件夹
cd ~/Documents/xzx_projects/ros2_ws
# 启动 ros2 相机节点
. ./install/setup.zsh
# 开启相机
ros2 launch orbbec_camera multi_camera.launch.py
```
启动后可开启另一个终端，打开 `rviz2` 来检查相机是否正常开启
```bash
ctrl + t
# 在新终端中输入
rviz2
```


### 示例用法

要设置一个新的终端，运行：

```bash
conda activate aloha
cd <act repo的路径>
```

### 模拟实验

下面的示例中使用``sim_transfer_cube_scripted``任务。另一个选择是``sim_insertion_scripted``。
要生成50个脚本化数据的回合，运行：

```bash
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir <数据保存目录> --num_episodes 50
```

可以添加``--onscreen_render``标志来查看实时渲染。
要在收集完数据后可视化模拟的回合，运行：

```bash
python3 visualize_episodes.py --dataset_dir <数据保存目录> --episode_idx 0
```

注意：要可视化来自mobile-aloha硬件的数据，使用来自[mobile-aloha](https://github.com/MarkFzp/mobile-aloha)的visualize_episodes.py。

要训练ACT：

```bash
# Transfer Cube任务
python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir <ckpt目录> --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0
```

要评估策略，运行相同的命令，但添加``--eval``。这将加载最佳验证检查点。

对于Transfer Cube任务，成功率应该在90%左右，对于插入任务则在50%左右。

要启用时间集成(temporal ensembling)，添加``--temporal_agg``

每次回合的录像将保存在``<ckpt_dir>``中。

您还可以添加``--onscreen_render``以在评估期间查看实时渲染。

部署ACT：需要在根目录创建一个results文件，来存储训练得到的模型文件

对于真实数据，由于建模更难，请至少训练5000个epochs，或者在损失平稳后训练3-4倍的时间。

更多信息，请参考[tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)。

### [ACT调优建议](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
如果您的ACT策略很不稳定或在回合中途停顿，只需训练更长时间！成功率和流畅性通常会在损失平稳后进一步提升。

