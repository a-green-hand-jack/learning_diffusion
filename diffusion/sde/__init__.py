"""
我要实现自己的 SDE 类，并且测试。

首先，我要定义一个 SDE 的抽象类，这个里面维护了一些通用的设定。
定义 SDE 的几种实践，包括 VP-SDE, VE-SDE 以及 sub-VP-SDE。
这些 SDE 仅仅维护了 前向过程、后向过程、sorce funcation 以及 drift 和 diffusion 的计算。

其次，我要定义一些采样的策略，包括 euler 采样、pc 采样、ode 采样等。

然后，我要定义一个 SdeTrainer 类，这个类维护了训练的通用逻辑。
这个类里面有一个 SDE 的实例；一个用来训练的模型的实例，并且是采用了 EMA 的模型；对应的 optimizer 实例； scheduler 实例。
通过这个 SdeTrainer 类，我实现了对模型的训练和评估。

随后，我要定义一个 SdeSampler 类，这个类维护了采样的通用逻辑。
这个类里面有一个 SDE 的实例；一个训练好的模型的实例，这个模型就是之前在Trainer里面保存的 EMA模型；一个采样实例。
通过这个 SdeSampler 类，我实现从随机噪音生成数据的过程。
"""
