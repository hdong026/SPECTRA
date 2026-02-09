import os
import itertools
import subprocess
import re
import time
import statistics  # 用于计算平均值

# ==============================================================================
# 1. 搜索空间配置
# ==============================================================================
SEARCH_SPACE = {
    # 结构参数
    "degree": [5],            # 既然是 Chebyshev，degree 不宜过大，4 左右通常最好
    "dropout": [0.0],
    
    # 优化参数
    "weight_decay": [1e-4],
    "lr": [0.002, 0.005, 0.01], 
}

# ==============================================================================
# 2. 固定设置
# ==============================================================================
CONFIG_PATH = "examples/HoloST/HoloST_PEMS04.py" 
LOG_DIR = "logs/tuning"                               
GPUS = "0"                                            
EPOCHS = 1   # 🔥 建议至少跑 20 轮，否则"平均值"没有意义（1轮的平均值=它自己）

os.makedirs(LOG_DIR, exist_ok=True)

def parse_avg_mae_from_log(log_path):
    """
    从日志文件中提取所有 Test MAE 并计算平均值
    """
    if not os.path.exists(log_path):
        return 999.0
    
    # 稍微等待文件写入
    time.sleep(1)
    
    mae_list = []
    
    with open(log_path, "r", encoding="utf-8", errors='ignore') as f:
        for line in f:
            # 匹配日志中的 Test MAE 行
            # 通常格式: ... Test MAE: 18.34 ... 或 Result <test>: ... test_MAE: 18.34
            if "MAE" in line and "test" in line.lower():
                try:
                    # 提取行内的所有浮点数
                    nums = re.findall(r"\d+\.\d+", line)
                    if nums:
                        # 假设第一个看起来像 loss 的数字是 MAE
                        # 过滤掉大于 100 的数 (防止把时间戳或者 epoch 数算进去)
                        val = float(nums[0]) 
                        if val < 100.0: 
                            mae_list.append(val)
                except:
                    pass
    
    if len(mae_list) == 0:
        return 999.0
    
    # 🔥 策略：计算平均值
    # 你也可以选择只计算最后 5 个 Epoch 的平均值，以代表收敛后的性能：
    # return statistics.mean(mae_list[-5:]) if len(mae_list) >= 5 else statistics.mean(mae_list)
    
    # 这里计算所有记录的平均值
    avg_mae = statistics.mean(mae_list)
    return avg_mae

def run_tuning():
    keys = SEARCH_SPACE.keys()
    values = SEARCH_SPACE.values()
    combinations = list(itertools.product(*values))
    
    print(f"🚀 开始超参搜索! 总组合数: {len(combinations)}")
    print(f"🎯 目标: 每个组合跑 {EPOCHS} Epochs, 计算【平均 Test MAE】")
    print("-" * 60)
    
    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        run_id = i + 1
        
        # 构造 Tag
        tag = f"D{params['degree']}_DP{params['dropout']}_WD{params['weight_decay']}_LR{params['lr']}"
        log_file = os.path.join(LOG_DIR, f"{tag}.log")
        
        print(f"[{run_id}/{len(combinations)}] 正在运行: {params}")
        
        # 准备环境变量
        env = os.environ.copy()
        env["HOLOST_GRID"] = str(params["degree"]) # 注意这里是 grid (对应 degree)
        env["HOLOST_DROP"] = str(params["dropout"])
        env["HOLOST_LR"] = str(params["lr"])
        env["HOLOST_WD"] = str(params["weight_decay"])
        env["HOLOST_EPOCHS"] = str(EPOCHS)
        env["HOLOST_TAG"] = tag
        env["HOLOST_TUNING"] = "1"
        
        cmd = [
            "python", "examples/run.py", 
            "--cfg", CONFIG_PATH, 
            "--gpus", GPUS
        ]
        
        start_time = time.time()
        
        with open(log_file, "w") as f_log:
            try:
                subprocess.run(cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError:
                print(f"❌ 运行失败! 请检查日志: {log_file}")
                results.append((params, 999.0))
                continue
            
        duration = (time.time() - start_time) / 60.0
        
        # 🔥 计算平均 MAE
        avg_mae = parse_avg_mae_from_log(log_file)
        
        print(f"   ✅ 完成! 耗时: {duration:.1f}m | 平均 Test MAE: {avg_mae:.4f}")
        results.append((params, avg_mae))

    print("\n" + "="*60)
    print("🏆 搜索结果汇总 (按 平均 MAE 从低到高排序)")
    print("="*60)
    
    results.sort(key=lambda x: x[1])
    for p, mae in results:
        print(f"Avg MAE: {mae:.4f} | Params: {p}")
        
    if results:
        print(f"\n✨ 最稳健组合: {results[0][0]}")

if __name__ == "__main__":
    run_tuning()