## 环境配置 

环境配置参考：k-planes 代码仓库 https://github.com/sarafridov/K-Planes/issues/3

由于在代码中添加了NerfAcc工具包，需要额外安装tinny-cuda-nn，安装参考nerfacc官网https://www.nerfacc.com/

## 数据集

D-NeRF数据集下载地址：https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0&file_subpath=%2Fdata

DyNeRF数据集下载地址：https://github.com/facebookresearch/Neural_3D_Video

## 训练

我们的配置文件在“configs”和”pre_models“目录中提供， 这些配置文件可能会使用下载数据的位置以及您所需的场景名称和实验名称进行更新。 要训练模型，请运行

```bash
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py
```

请注意，对于 DyNeRF 场景，建议首先以 4 倍下采样运行单次迭代，以预先计算并存储光线重要性权重，然后照常以 2 倍下采样运行。 其他数据集则不需要。

## 可视化/评估

`main.py`脚本还支持渲染新颖的相机轨迹、评估质量指标以及从保存的模型渲染时空分解视频。 这些选项可通过标志`--render-only`、`--validate-only`和`--spacetime-only`访问，并且可以通过`--log-dir`指定保存的模型路径。更多选项参考`opt.py`。

