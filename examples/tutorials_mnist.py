import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch4keras.model import BaseModel, Trainer
from torch4keras.snippets import seed_everything
from torch4keras.callbacks import Checkpoint, Evaluator, EarlyStopping, Summary, Logger
from torch4keras.callbacks import EmailCallback, WandbCallback, Tensorboard, SystemStateCallback
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# 参数区
seed_everything(42)
steps_per_epoch = 1000
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
method = 1  # 1 or 2
resume_path = None  # './ckpt/2/'

# 读取数据
mnist = torchvision.datasets.MNIST(root='./', download=True)
x, y = mnist.train_data.unsqueeze(1), mnist.train_labels
x, y = x.to(device), y.to(device)
x = x.float() / 255.0    # scale the pixels to [0, 1]
x_train, y_train = x[:40000], y[:40000]
train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=8)
x_test, y_test = x[40000:], y[40000:]
test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=8)


# 准备模型
net = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7744, 10)
        )

if method == 1:
    # 方式1: 继承BaseModel
    class MyModel(BaseModel):
        def __init__(self, module):
            super().__init__()
            self.model = module
        def forward(self, inputs):
            return self.model(inputs)
    
    model = MyModel(net).to(device)
    optimizer = optim.Adam(model.parameters())
    scheduler = get_linear_schedule_with_warmup(optimizer, steps_per_epoch, steps_per_epoch*epochs)
    model.compile(optimizer=optimizer, scheduler=scheduler, mixed_precision=True, loss=nn.CrossEntropyLoss(), metrics=['acc'])

elif method == 2:
    # 方式2：把nn.Module传入Trainer
    model = Trainer(net.to(device))
    optimizer = optim.Adam(net.parameters())
    scheduler = get_linear_schedule_with_warmup(optimizer, steps_per_epoch, steps_per_epoch*epochs)
    model.compile(optimizer=optimizer, scheduler=scheduler, loss=nn.CrossEntropyLoss(), metrics=['acc'], clip_grad_norm=1.0)

# 断点续训
if resume_path is not None:
    model.resume_from_checkpoint(resume_path)

class MyEvaluator(Evaluator):
    # 重构评价函数
    def evaluate(self):
        total, hit = 1e-5, 0
        for X, y in tqdm(test_dataloader, desc='Evaluating', ncols=80):
            pred_y = model.predict(X).argmax(dim=-1)
            hit += pred_y.eq(y).sum().item()
            total += y.shape[0]
        return {'test_acc': hit/total}
    

if __name__ == '__main__':
    evaluator = MyEvaluator(monitor='test_acc', save_dir='./ckpt/best/')
    ckpt = Checkpoint(save_dir='./ckpt/{epoch}')
    early_stop = EarlyStopping(monitor='test_acc', verbose=1)
    logger = Logger('./ckpt/log.log')  # log文件
    ts_board = Tensorboard('./ckpt/tensorboard')  # tensorboard
    email = EmailCallback(mail_receivers='tongjilibo@163.com')  # 发送邮件
    wandb = WandbCallback(save_code=True)  # wandb
    syscallback = SystemStateCallback('./ckpt/tensorboard/system')
    hist = model.fit(train_dataloader, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                     callbacks=[Summary(), syscallback, evaluator, ts_board, logger, ckpt, early_stop])
else:
    model.load_weights('./ckpt/5/model.pt')
    metrics = MyEvaluator().evaluate()
    print(metrics)
