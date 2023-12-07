## 版本历史

|更新日期| 版本 | 版本说明 |
|------| ----------------- |----------- |
|20231207| 0.1.6     |监控fit过程，有报错则发送邮件提醒; 解决torch2.0的compile冲突问题|
|20230928| 0.1.5     |进度条中显示已经训练的时间|
|20230912|v0.1.4.post2|History增加plot()方法, 增加add_module()方法，修复0.1.4的_argparse_forward的bug, 增加loss2metrics方法|
|20230909|v0.1.4|增加from_pretrained和save_pretrained方法，增加log_warn_once方法，compile()中可设置成员变量，默认move_to_model_device设置为True, 增加JsonConfig，增加_argparse_forward()方便下游继承改写Trainer|
|20230901|v0.1.3|compile()可不传参，interval不一致报warning, 去除部分self.vars, 调整move_to_model_device逻辑，DDP每个epoch重新设置随机数，save_weights()和load_weights()可以按照`pretrained`格式|
|20230821|v0.1.2.post2|代码结构调整，增加trainer.py文件，方便下游集成|
|20230812|v0.1.2|修复DeepSpeedTrainer，修复DDP|
|20230803|v0.1.1|增加指标平滑的SmoothMetricCallback，统一管理指标平滑的问题, 增加SKIP_METRICS，NO_SMOOTH_METRICS，ROUND_PRECISION，默认对指标会进行平滑，修改tensorboard和wandb的callback, 允许跳过nan的指标, Tensorboard可以记录gpu等系统信息|
|20230725|v0.1.0.post2|修复v0.1.0的bug，主要是进度条和log的标签平滑的问题|
|20230724 | v0.1.0: | 允许调整进度条的显示参数, 进度条和日志同步（如果进度条平滑了则日志也平滑）, 自动把tensor转到model.device上, 允许打印第一个batch来检查样本 |
|20230716 | v0.0.9 | 增加auto_set_cuda_devices自动选择显卡，增加log_info，log_warn, log_error等小函数 |
|20230625 | v0.0.8 | 增加EmailCallback和WandbCallback, 增加AccelerateTrainer和DeepSpeedTrainer, grad_accumulation_steps内算一个batch，修改Trainer中部分成员函数|
|20230517 | v0.0.7.post3: | 修复保存scheduler|
|20230517 | v0.0.7.post2: | Checkpoint Calback增加保存scheduler, save_weights可自行创建目录，Logger, Tensorboard模块加入lr, 修改predict和add_trainer|
|20230505 | v0.0.7 | 独立出callbacks.py文件, fit允许输入形式为字典，load_weights支持list输入，save_weights支持仅保存可训练参数 |
|20230212 | v0.0.6 | 增加resume_from_checkpoint和save_to_checkpoint；增加add_trainer方法，重构了Trainer(BaseModel)的实现(增加几个成员变量、增加initilize()、删除对forward参数个数的判断、dp和ddp不解析module、修改use_amp参数为mixed_precision)，增加了AccelerateCallback|
|20221217 | v0.0.5 | 增加Summary的Callback, 增加Tqdm的进度条展示，保留原有BaseModel的同时，增加Trainer(不从nn.Module继承), 从bert4torch的snippets迁移部分通用函数|
|20221127 | v0.0.4 | 为callback增加on_train_step_end方法, 修复BaseModel(net)方式的bug |
|20221107 | v0.0.3.post2 | 修复DDP下打印的bug|
|20221106 | v0.0.3 | 参考Keras修改了callback的逻辑|
|20221023 | v0.0.2 | 增加Checkpoint, Evaluator等自带Callback, 修改BaseModel(net)方式，修复DP和DDP的__init__()|
|20221019 | v0.0.1 | 初始版本|
