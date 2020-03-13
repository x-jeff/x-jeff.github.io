---
layout:     post
title:      【Conda】常用的conda命令
subtitle:   基本命令，包管理，虚拟环境，镜像源
date:       2019-12-26
author:     x-jeff
header-img: blogimg/20191226.jpg
catalog: true
tags:
    - Conda
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.环境

本机环境：`Mac OS`

# 2.基本命令

1. 获取conda版本号：`conda --version`
2. 获取帮助：
	* `conda --help`
	* `conda -h`
3. 查看某一命令的使用方法（以`upgrade `为例）：
	* `conda upgrade --help`
	* `conda upgrade -h`
4. 查看conda配置：`conda config --show`

# 3.包管理

1. 查看已经安装的包：`conda list`
2. 更新所有的包：`conda upgrade --all`
3. 更新指定包：`conda upgrade <pkg name>`
4. 卸载指定包：`conda uninstall <pkg name>`
5. 搜索包：`conda search <pkg name>`
6. 安装指定包：`conda install <pkg name>`

# 4.conda虚拟环境

1. 查看已有的conda虚拟环境：
	* `conda info -e`
	* `conda env list`
2. 创建conda虚拟环境：`conda create -n <env name> python=<python version> <pkg1 pkg2>`
3. 进入虚拟环境：`source activate <env name>`
4. 退出虚拟环境：`source deactivate`
	* conda 4.8.0版本改为：`conda deactivate`
5. 复制虚拟环境：`conda create -n <new env> --clone <old env>`
6. 删除环境：`conda remove -n <env name> --all`

## 4.1.移植虚拟环境

将虚拟环境从一台主机移植到另一台主机上：

1. 输出已有的虚拟环境配置：`conda env export > environment.yml`
2. 根据配置文件生成一模一样的虚拟环境：`conda env create -f environment.yml`

# 5.conda源

1. 查看当前使用源：`conda config --show-sources`
2. 删除指定源：`conda config --remove channels`
3. 添加指定源：`conda config --add channels`

## 5.1.镜像源

因为官方源在国外，所以在国内访问速度很慢或访问失败，所以可以通过以下方式设置国内镜像源。

👉清华源：

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

👉中科大源：

```
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

👉换回默认源：

```
conda config --remove-key channels
```
