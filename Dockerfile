FROM python:3.10.12
LABEL authors="maples"
WORKDIR /ACT_plus_plus
COPY . /ACT_plus_plus

# 安装 Git 和其他必要的依赖
RUN apt-get update && \
    apt-get install -y git vim tmux libgl1-mesa-glx sudo && \
    apt-get clean
# 设置 pip 源
RUN echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /etc/pip.conf
#RUN #pip cache purge
# 安装 PyTorch 特定源
#RUN pip install  \
#    "https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp38-cp38-linux_x86_64.whl"

RUN pip install --upgrade pip
#    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
RUN pip install .
RUN cd ./robomimic
RUN pip install .
CMD ["/bin/bash"]


