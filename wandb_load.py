import wandb

# 加载 .wandb 文件
run = wandb.restore(r"E:\image_inpainting\HINT-main\wandb\run-20240509_203349-irn20cds", run_path='.')

# 获取日志数据
history = run.history()
config = run.config

# 在这里可以使用日志数据或配置数据进行分析或可视化
print("配置信息:", config)
print("日志历史:", history)
