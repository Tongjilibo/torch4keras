from torch4keras.snippets import timeit, TimeitContextManager, TimeitLogger
import time


# 装饰器
@timeit
def func(n=10):
    for _ in range(n):
        time.sleep(0.1)
func()

# 上下文管理器 - 统计累计耗时
with TimeitContextManager() as ti:
    for i in range(10):
        time.sleep(0.1)
        ti.lap(name=i, reset=False)  # 统计累计耗时

# 上下文管理器 - 统计每段速度
with TimeitContextManager() as ti:
    for i in range(10):
        time.sleep(0.1)
        ti.lap(count=10, name=i, reset=True)
    ti(10) # 统计速度


# 上下文管理器 - 统计速度
with TimeitContextManager() as ti:
    for i in range(10):
        time.sleep(0.1)
        ti.lap(name=i, reset=True)
    ti(10) # 统计速度

ti = TimeitLogger()
for i in range(10):
    time.sleep(0.1)
    ti.lap(name=i)

for i in range(10):
    time.sleep(0.1)
    ti.lap(name=i)
ti.end() # 打印时长

