from torch4keras.snippets import timeit, Timeit
import time


# 装饰器
@timeit
def func(n=10):
    for _ in range(n):
        time.sleep(0.1)
func()

# 上下文管理器 - 统计累计耗时
with Timeit() as ti:
    for i in range(10):
        time.sleep(0.1)
        ti.lap(prefix=i, restart=False)  # 统计累计耗时

# 上下文管理器 - 统计每段速度
with Timeit() as ti:
    for i in range(10):
        time.sleep(0.1)
        ti.lap(count=10, prefix=i, restart=True)
    ti(10) # 统计速度


# 上下文管理器 - 统计速度
with Timeit() as ti:
    for i in range(10):
        time.sleep(0.1)
        ti.lap(prefix=i, restart=True)
    ti(10) # 统计速度