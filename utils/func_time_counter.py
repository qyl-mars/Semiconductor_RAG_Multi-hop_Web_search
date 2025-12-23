from functools import wraps
import time

def timer(func):
    """
    一个用来计算函数运行时间的装饰器
    """

    @wraps(func)  # 【关键点 1】保留原函数的元数据（如函数名、文档注释）
    def wrapper(*args, **kwargs):  # 【关键点 2】接收任意参数，不管原函数有几个参数都能用

        # A. 记录开始时间
        start_time = time.perf_counter()

        # B. 执行原函数 (并接住它的返回值)
        result = func(*args, **kwargs)

        # C. 记录结束时间
        end_time = time.perf_counter()

        # D. 计算并打印耗时
        elapsed_time = end_time - start_time
        print(f"⏱️  函数 [{func.__name__}] 运行耗时: {elapsed_time:.6f} 秒")

        # E. 返回原结果
        return result  # 【关键点 3】必须把原函数的执行结果返回去，否则外面调用会得到 None

    return wrapper