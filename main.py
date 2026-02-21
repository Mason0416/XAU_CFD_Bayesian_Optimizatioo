import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def fast_clean_and_resample(df):
    #清洗：去重、去異常
    df = df.drop_duplicates(subset=['time_msc'])
    df['mid'] = (df['bid'] + df['ask']) / 2
    
    #轉換時間並降維至 1 分鐘
    df.index = pd.to_datetime(df['time_msc'], unit='ms')
    ohlc = df['mid'].resample('1min').ohlc()
    
    #標記「交易日」與「時間」，方便日內策略判斷尾盤
    ohlc['time'] = ohlc.index.time
    ohlc['date'] = ohlc.index.date
    return ohlc.dropna()

def orb_strategy_with_stop_loss(df, k_upper, k_lower, sl_pct=0.005, tp_pct=0.01, obs_mins=30):
    #前置處理：標記分鐘與計算區間
    df['count'] = df.groupby('date').cumcount()
    observation_period = df[df['count'] < obs_mins]
    range_high = observation_period.groupby('date')['high'].max()
    range_low = observation_period.groupby('date')['low'].min()
    df['obs_high'] = df['date'].map(range_high)
    df['obs_low'] = df['date'].map(range_low)
    df['long_threshold'] = df['obs_high'] + k_upper * (df['obs_high'] - df['obs_low'])
    df['short_threshold'] = df['obs_low'] - k_lower * (df['obs_high'] - df['obs_low'])

    #核心邏輯：加入止損的逐行運算
    signals = np.zeros(len(df))
    entry_price = 0
    curr_pos = 0 # 0: 空手, 1: 多單, -1: 空單

    #為了速度，我們用 values 進行運算
    close_prices = df['close'].values
    long_thresh = df['long_threshold'].values
    short_thresh = df['short_threshold'].values
    times = df['time'].values
    minute_counts = df['count'].values

    for i in range(1, len(df)):
        #每日尾盤強平 (23:00)
        if times[i] >= pd.to_datetime('23:00:00').time():
            curr_pos = 0
            entry_price = 0
        
        #只有在觀察期後才進場
        elif minute_counts[i] >= obs_mins:
            # --- 多單邏輯 ---
            if curr_pos == 0 and close_prices[i] > long_thresh[i]:
                curr_pos = 1
                entry_price = close_prices[i]
            elif curr_pos == 1:
                # 觸及止損或止盈
                if close_prices[i] <= entry_price * (1 - sl_pct) or close_prices[i] >= entry_price * (1 + tp_pct):
                    curr_pos = 0
                    entry_price = 0

            # --- 空單邏輯 ---
            elif curr_pos == 0 and close_prices[i] < short_thresh[i]:
                curr_pos = -1
                entry_price = close_prices[i]
            elif curr_pos == -1:
                # 觸及止損或止盈
                if close_prices[i] >= entry_price * (1 + sl_pct) or close_prices[i] <= entry_price * (1 - tp_pct):
                    curr_pos = 0
                    entry_price = 0
        
        signals[i] = curr_pos

    df['signal'] = signals
    # 計算收益 (同前)
    df['strat_ret'] = df['signal'].shift(1) * df['close'].pct_change()
    
    # 計算 Sharpe & MDD
    ann_factor = np.sqrt(252 * 1440)
    sharpe = (df['strat_ret'].mean() / df['strat_ret'].std()) * ann_factor if df['strat_ret'].std() != 0 else 0
    cum_ret = (1 + df['strat_ret']).cumprod()
    mdd = ((cum_ret.cummax() - cum_ret) / cum_ret.cummax()).max()
    
    return sharpe, mdd, cum_ret

class GoldBayesianOptimizer:
    def __init__(self, df):
        self.df = df
        self.bounds = np.array([[0.1, 3.0], [0.1, 3.0]]) # K值的搜尋範圍
        
    def expected_improvement(self, X, X_sample, Y_sample, gpr, xi=0.01):
        mu, sigma = gpr.predict(X, return_std=True)
        mu_sample_opt = np.max(Y_sample)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def optimize(self, train_data, n_iters=25):
        # 初始點
        X_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(8, 2))
        Y_sample = np.array([self.get_score(p, train_data) for p in X_sample])
        
        # 高斯過程模型
        kernel = C(1.0) * RBF(length_scale=[1.0, 1.0])
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        
        for _ in range(n_iters):
            gpr.fit(X_sample, Y_sample)
            X_next_candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(100, 2))
            ei = self.expected_improvement(X_next_candidates, X_sample, Y_sample, gpr)
            x_next = X_next_candidates[np.argmax(ei)]
            
            y_next = self.get_score(x_next, train_data)
            X_sample = np.vstack([X_sample, x_next])
            Y_sample = np.append(Y_sample, y_next)
            
        best_idx = np.argmax(Y_sample)
        return X_sample[best_idx], Y_sample[best_idx]

    def get_score(self, params, data):
        #調用之前的 orb_strategy_
        sharpe, mdd, _ = orb_strategy_with_stop_loss(data, params[0], params[1])
        #目標：最大化 Sharpe，但如果 MDD > 10% 嚴重懲罰
        return sharpe - (max(0, mdd - 0.10) * 20)
    

def run_full_walk_forward(df, train_size_months=6, test_size_months=2):
    df.index = pd.to_datetime(df.index)
    start_date = df.index.min()
    end_date = df.index.max()
    
    current_train_start = start_date
    oos_summary = []
    all_test_returns = [] # 用於拼接最終收益曲線

    optimizer = GoldBayesianOptimizer(df)

    print(f" 開始執行 Walk-Forward Validation")
    print(f" 資料範圍: {start_date.date()} 至 {end_date.date()}")
    print(f" 窗口配置: 訓練 {train_size_months}m / 測試 {test_size_months}m\n")

    window_idx = 1
    while True:
        train_end = current_train_start + pd.DateOffset(months=train_size_months)
        test_end = train_end + pd.DateOffset(months=test_size_months)
        
        if test_end > end_date:
            print(f" 剩餘資料不足一個完整測試窗口，驗證結束。")
            break
            
        # 切分資料
        train_data = df[(df.index >= current_train_start) & (df.index < train_end)]
        test_data = df[(df.index >= train_end) & (df.index < test_end)]
        
        print(f"--- [Window {window_idx}] ---")
        print(f" 訓練中: {current_train_start.date()} ~ {train_end.date()}")
        
        # 1. 訓練：貝式優化尋找最佳 K
        best_k, train_score = optimizer.optimize(train_data, n_iters=20)
        
        # 2. 測試：使用最佳參數跑 Out-of-Sample
        test_sharpe, test_mdd, test_history = orb_strategy_with_stop_loss(
            test_data, best_k[0], best_k[1]
        )
        
        # 儲存結果
        oos_summary.append({
            'window': window_idx,
            'test_period': f"{train_end.date()}~{test_end.date()}",
            'k_up': round(best_k[0], 2),
            'k_low': round(best_k[1], 2),
            'train_score': round(train_score, 2),
            'test_sharpe': round(test_sharpe, 2),
            'test_mdd': f"{test_mdd:.2%}"
        })
        
        print(f"最佳參數: K_up={best_k[0]:.2f}, K_low={best_k[1]:.2f}")
        print(f"測試表現: Sharpe={test_sharpe:.2f}, MDD={test_mdd:.2%}")
        
        #滾動
        current_train_start = current_train_start + pd.DateOffset(months=test_size_months)
        window_idx += 1

    # 轉成 DataFrame 方便查看
    results_df = pd.DataFrame(oos_summary)
    
    print("\n" + "="*50)
    print("WALK-FORWARD 總結報告")
    print("="*50)
    print(results_df.to_string(index=False))
    print("="*50)
    
    # 計算平均表現
    avg_sharpe = results_df['test_sharpe'].mean()
    print(f"平均測試 Sharpe Ratio: {avg_sharpe:.2f}")
    
    return results_df



# 使用範例:
df_clean = fast_clean_and_resample(pd.read_parquet('../2y_pq/XAUUSD_2y_pq.parquet',engine='fastparquet'))
print(df_clean.head())

walk_forward_results = run_full_walk_forward(df_clean)





#畫圖
# 1. 準備數據 (根據你的結果)
windows = walk_forward_results['window']
test_sharpe = walk_forward_results['test_sharpe']
# 將 MDD 字串轉回數字 (例如 "4.70%" -> 0.047)
test_mdd = walk_forward_results['test_mdd'].str.rstrip('%').astype(float) / 100
k_up = walk_forward_results['k_up']
k_low = walk_forward_results['k_low']

# 2. 開始繪圖
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- 上圖：Sharpe vs MDD ---
ax1.set_title('Gold Intraday ORB: Walk-Forward Test Performance', fontsize=14, fontweight='bold')
ax1.bar(windows, test_sharpe, color='skyblue', alpha=0.7, label='Test Sharpe Ratio')
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Benchmark (Sharpe=1.0)')
ax1.set_ylabel('Sharpe Ratio', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.legend(loc='upper left')

# 建立右軸畫 MDD
ax2 = ax1.twinx()
ax2.plot(windows, test_mdd, color='salmon', marker='o', linewidth=2, label='Max Drawdown (MDD)')
ax2.set_ylabel('Max Drawdown', color='indianred', fontsize=12)
ax2.tick_params(axis='y', labelcolor='indianred')
ax2.invert_yaxis() # 讓 MDD 往上走代表風險增加，或者你可以不反轉看深度
ax2.set_ylim(0.15, 0) # 鎖定 0~15% 範圍看回撤
ax2.legend(loc='upper right')

# --- 下圖：參數演變 ---
ax3.set_title('Optimal Parameters Evolution (Bayesian Opt)', fontsize=14, fontweight='bold')
ax3.plot(windows, k_up, marker='s', label='k_up (Long Trigger)', color='green', linewidth=2)
ax3.plot(windows, k_low, marker='d', label='k_low (Short Trigger)', color='orange', linewidth=2)
ax3.set_xlabel('Walk-Forward Window Index', fontsize=12)
ax3.set_ylabel('K Value', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()


