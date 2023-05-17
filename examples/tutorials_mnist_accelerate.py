# 使用huggingface的accelerate库来加速训练

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch4keras.model import BaseModel, Trainer
from torch4keras.snippets import seed_everything
from torch4keras.callbacks import Checkpoint, Evaluator, EarlyStopping, Summary
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch4keras.model import add_trainer
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(42)
accelerator = Accelerator()
accelerator.print(f'device {str(accelerator.device)} is used!')

# 读取数据
mnist = torchvision.datasets.MNIST(root='./', download=True)
x, y = mnist.train_data.unsqueeze(1), mnist.train_labels
x, y = x.to(accelerator.device), y.to(accelerator.device)
x = x.float() / 255.0    # scale the pixels to [0, 1]
x_train, y_train = x[:40000], y[:40000]
train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=8)
x_test, y_test = x[40000:], y[40000:]
test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=8)

# 方式1: 继承BaseModel（推荐）
class MyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7744, 10)
        )
    def _forward(self, inputs):
        return self.model(inputs)
model = MyModel().to(accelerator.device)

# initialize accelerator and auto move data/model to accelerator.device
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optim.Adam(model.parameters()), train_dataloader, test_dataloader)
model = add_trainer(model)
model.compile(optimizer=optim.Adam(model.parameters()), loss=nn.CrossEntropyLoss(), metrics=['acc'], accelerator=accelerator)


class MyEvaluator(Evaluator):
    # 重构评价函数
    def evaluate(self):
        total, hit = 1e-5, 0
        for X, y in tqdm(test_dataloader, desc='Evaluating:'):
            pred_y = model.predict(X).argmax(dim=-1)
            hit += pred_y.eq(y).sum().item()
            total += y.shape[0]
        return {'test_acc': hit/total}


if __name__ == '__main__':
    evaluator = MyEvaluator(monitor='test_acc', 
                            model_path='./ckpt/best_model.pt', 
                            optimizer_path='./ckpt/best_optimizer.pt', 
                            steps_params_path='./ckpt/best_step_params.pt')
    ckpt = Checkpoint('./ckpt/model_{epoch}_{test_acc:.5f}.pt',
                      optimizer_path='./ckpt/optimizer_{epoch}_{test_acc:.5f}.pt',
                      steps_params_path='./ckpt/steps_params_{epoch}_{test_acc:.5f}.pt')
    early_stop = EarlyStopping(monitor='test_acc', verbose=1)
    model.fit(train_dataloader, steps_per_epoch=100, epochs=5, callbacks=[evaluator, ckpt, early_stop])
