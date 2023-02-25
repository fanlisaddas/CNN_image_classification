# CNN_image_classification
CNN 图像识别  

使用 FashionMNIST 数据集训练 CNN 模型，在测试中 20 个周期达到 97.5% 的准确率。

使用  PyTorch 框架进行开发。

——————————————————————————————————————————————————

data_load.py：继承 Dataset 类，重写接口加载数据进入神经网络模型。

train_and_test.py：模型训练及测试函数。

model.py：模型结构。

main.py：定义超参数，模型训练和测试。

——————————————————————————————————————————————————

参考：

https://pytorch.org/tutorials
