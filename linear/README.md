

1. **初始化参数：**
   - 假设初始权重 \(w\) 和偏置 \(b\) 是随机初始化的。

   \[ w = \text{随机初始化}, \quad b = \text{随机初始化} \]

2. **前向传播：**
   - 计算模型的输出 \(\hat{y} = wx + b\)。

   \[ \hat{y} = w \cdot x + b \]

3. **计算损失：**
   - 使用均方误差损失函数计算预测值 \(\hat{y}\) 与真实值 \(y\) 之间的误差。

   \[ L = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 \]

   其中 \(n\) 是样本数量，\(\hat{y}_i\) 是模型对第 \(i\) 个样本的预测值，\(y_i\) 是实际值。

4. **计算损失对预测值的梯度：**
   - 计算损失函数对每个样本的预测值的梯度。

   \[ \frac{\partial L}{\partial \hat{y}_i} = \frac{1}{n} (\hat{y}_i - y_i) \]

5. **计算预测值对模型参数的梯度：**
   - 计算模型的预测值对权重 \(w\) 和偏置 \(b\) 的梯度。

   \[ \frac{\partial \hat{y}_i}{\partial w} = x_i \]
   \[ \frac{\partial \hat{y}_i}{\partial b} = 1 \]

6. **使用链式法则计算损失对模型参数的梯度：**
   - 计算损失对权重 \(w\) 和偏置 \(b\) 的梯度。

   \[ \frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial w} \]
   \[ \frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial b} \]

7. **梯度下降更新参数：**
   - 使用计算得到的梯度，通过梯度下降算法更新模型参数。

   \[ w = w - \alpha \frac{\partial L}{\partial w} \]
   \[ b = b - \alpha \frac{\partial L}{\partial b} \]

   其中，\(\alpha\) 是学习率，控制更新的步长。

这样，通过多次迭代（多个 epoch），模型的参数 \(w\) 和 \(b\) 将逐渐调整，使得模型的预测值 \(\hat{y}\) 逼近真实值 \(y\)。这是一个简单的线性回归模型的反向传播过程。
