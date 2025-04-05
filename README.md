# VerbNet

`VerbNet` 是一个用于构建和训练神经网络的 `.NET` 库。它提供了基本的张量操作、神经网络层、损失函数等功能，可用于各种深度学习任务。

## 项目结构
- `VerbNet.Core`：核心库，包含张量操作、神经网络层、损失函数等实现。
- `VerbNet.Demo`：示例项目，展示如何使用 `VerbNet` 构建和训练一个简单的神经网络。

## 功能特性
- **张量操作**：支持基本的张量运算，如加法、减法、乘法、除法、矩阵乘法等。
- **自动求导**：支持自动求导，通过 `Backward` 方法计算梯度。

## TODO
### 功能开发
- [x] 实现 SIMD 算子以提高计算性能，包括 SIMD 加法、减法、乘法和矩阵乘法。
- [x] 实现更多的激活函数，如 ReLU、Sigmoid、Tanh 等。
- [ ] 支持更复杂的损失函数，如交叉熵损失、Huber 损失等。
- [ ] 增加优化器，如 SGD、Adam、RMSProp 等。
- [ ] 增加更多的神经网络层。

## 使用示例
以下是一个简单的示例，展示如何使用 `VerbNet` 构建和训练一个简单的神经网络：

```csharp
using VerbNet.Core;

namespace VerbNet.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // 定义神经网络层
            LayerList layers = new LayerList(
                new Linear(16, 512, true, 0.001f),
                new Linear(512, 512, true, 0.001f),
                new Linear(512, 16, true, 0.001f)
            );

            // 定义损失函数
            MSELoss mse = new MSELoss(true);

            // 生成随机输入和目标数据
            Tensor input = Tensor.Random([64, 16]);
            Tensor target = Tensor.Random([64, 16]);

            // 训练循环
            for (int i = 0; i < 1000000; i++)
            {
                // 清零梯度
                layers.ZeroGrad();

                // 前向传播
                Tensor output = layers.Forward(input);

                // 计算损失
                mse.Forward(output, target);

                // 反向传播
                mse.Backward();

                // 打印损失值
                Console.WriteLine(mse.LossValue);

                // 更新参数
                layers.ApplyGrad();
            }
        }
    }
}
```

## 许可证
该项目采用 Apache License 2.0 许可证。详细信息请参阅 [LICENSE.txt](LICENSE.txt) 文件。
