import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch4keras.model import BaseModel, Trainer
from torch4keras.snippets import seed_everything
from torch4keras.callbacks import Checkpoint, Evaluator, EarlyStopping, Summary, Logger, EmailCallback, WandbCallback, Tensorboard
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

seed_everything(42)
steps_per_epoch = 1000
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 读取数据
mnist = torchvision.datasets.MNIST(root='./', download=True)
x, y = mnist.train_data.unsqueeze(1), mnist.train_labels
x, y = x.to(device), y.to(device)
x = x.float() / 255.0    # scale the pixels to [0, 1]
x_train, y_train = x[:40000], y[:40000]
train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=8)
x_test, y_test = x[40000:], y[40000:]
test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=8)

# 方式1: 继承BaseModel
# class MyModel(BaseModel):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(),
#             nn.MaxPool2d(2, 2), 
#             nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(7744, 10)
#         )
#     def forward(self, inputs):
#         return self.model(inputs)
# model = MyModel().to(device)
# optimizer = optim.Adam(model.parameters())
# scheduler = get_linear_schedule_with_warmup(optimizer, steps_per_epoch, steps_per_epoch*epochs)
# model.compile(optimizer=optimizer, scheduler=scheduler, loss=nn.CrossEntropyLoss(), metrics=['acc'])


# 方式2：把nn.Module传入Trainer
net = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7744, 10)
        )
model = Trainer(net.to(device))
optimizer = optim.Adam(net.parameters())
scheduler = get_linear_schedule_with_warmup(optimizer, steps_per_epoch, steps_per_epoch*epochs)
model.compile(optimizer=optimizer, scheduler=scheduler, loss=nn.CrossEntropyLoss(), metrics=['acc'], clip_grad_norm=1.0)

class MyEvaluator(Evaluator):
    # 重构评价函数
    def evaluate(self):
        total, hit = 1e-5, 0
        for X, y in tqdm(test_dataloader, desc='Evaluating'):
            pred_y = model.predict(X).argmax(dim=-1)
            hit += pred_y.eq(y).sum().item()
            total += y.shape[0]
        return {'test_acc': hit/total}
    

if __name__ == '__main__':
    evaluator = MyEvaluator(monitor='test_acc', save_dir='./ckpt/best/')
    ckpt = Checkpoint(save_dir='./ckpt/{epoch}/')
    early_stop = EarlyStopping(monitor='test_acc', verbose=1)
    logger = Logger('./ckpt/log.log')  # log文件
    ts_board = Tensorboard('./ckpt/tensorboard')  # tensorboard
    email = EmailCallback(mail_receivers='tongjilibo@163.com')  # 发送邮件
    wandb = WandbCallback(save_code=True)  # wandb
    hist = model.fit(train_dataloader, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                     callbacks=[Summary(), evaluator, ts_board, logger, ckpt, early_stop])
else:
    model.load_weights('./ckpt/5/model.pt')
    metrics = MyEvaluator().evaluate()
    print(metrics)
