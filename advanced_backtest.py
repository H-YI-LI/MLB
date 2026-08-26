import os
import statsapi
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss

print("=" * 65)
print(" 🚀 MLB 進階量化勝負預測與策略回測系統 (5大進階維度整合版)")
print("=" * 65)

# -------------------------------------------------------------
# 1. 載入全聯盟 2024 歷史賽事
# -------------------------------------------------------------
csv_cache_file = "mlb_2024_schedule.csv"
if os.path.exists(csv_cache_file):
    print(f"\n[步驟 1/5] 從本地快取 {csv_cache_file} 載入賽事數據...")
    df_raw = pd.read_csv(csv_cache_file)
else:
    print("\n[步驟 1/5] 正在分批下載 2024 賽季數據...")
    month_ranges = [
        ('2024-03-20', '2024-04-30'),
        ('2024-05-01', '2024-05-31'),
        ('2024-06-01', '2024-06-30'),
        ('2024-07-01', '2024-07-31'),
        ('2024-08-01', '2024-08-31'),
        ('2024-09-01', '2024-10-05'),
    ]
    all_games = []
    for start, end in month_ranges:
        month_data = statsapi.schedule(start_date=start, end_date=end)
        all_games.extend(month_data)
    df_raw = pd.DataFrame(all_games)
    df_raw.to_csv(csv_cache_file, index=False)

df_games = df_raw[(df_raw['status'] == 'Final') & (df_raw['game_type'] == 'R')].copy()
df_games = df_games.sort_values(by=['game_date', 'game_id']).reset_index(drop=True)
print(f"-> 成功載入 {len(df_games)} 場已完賽的例行賽紀錄！")

# -------------------------------------------------------------
# 2. 定義【球場效應係數 (Park Factor)】表 (以 1.0 為基準)
# -------------------------------------------------------------
PARK_FACTORS = {
    'Coors Field': 1.25,        # 落磯 (打者天堂)
    'Fenway Park': 1.10,        # 紅襪
    'Great American Ball Park': 1.08, # 紅人
    'Kauffman Stadium': 1.06,   # 皇家
    'Globe Life Field': 1.04,   # 遊騎兵
    'Yankee Stadium': 1.03,     # 洋基
    'Dodger Stadium': 0.98,     # 道奇
    'Oracle Park': 0.94,        # 巨人
    'T-Mobile Park': 0.91,      # 水手
    'Petco Park': 0.90,         # 教士 (投手天堂)
    'Citi Field': 0.92,         # 大都會
    'Tropicana Field': 0.92     # 光芒
}

# -------------------------------------------------------------
# 3. 拆解球隊視角並計算進階特徵
# -------------------------------------------------------------
print("\n[步驟 2/5] 正在計算進階量化特徵 (球場效應 / 貝氏均值收縮 / 牛棚疲勞)...")

home_rows, away_rows = [], []
for _, row in df_games.iterrows():
    venue = row.get('venue_name', 'Unknown')
    pf = PARK_FACTORS.get(venue, 1.0)
    
    home_rows.append({
        'game_id': row['game_id'], 'game_date': pd.to_datetime(row['game_date']),
        'team_id': row['home_id'], 'team_name': row['home_name'],
        'opp_id': row['away_id'], 'opp_name': row['away_name'],
        'is_home': 1, 'runs': pd.to_numeric(row['home_score'], errors='coerce'),
        'runs_allowed': pd.to_numeric(row['away_score'], errors='coerce'),
        'win': 1 if row['home_score'] > row['away_score'] else 0,
        'sp_name': row.get('home_probable_pitcher', 'Unknown'),
        'park_factor': pf
    })
    away_rows.append({
        'game_id': row['game_id'], 'game_date': pd.to_datetime(row['game_date']),
        'team_id': row['away_id'], 'team_name': row['away_name'],
        'opp_id': row['home_id'], 'opp_name': row['home_name'],
        'is_home': 0, 'runs': pd.to_numeric(row['away_score'], errors='coerce'),
        'runs_allowed': pd.to_numeric(row['home_score'], errors='coerce'),
        'win': 1 if row['away_score'] > row['home_score'] else 0,
        'sp_name': row.get('away_probable_pitcher', 'Unknown'),
        'park_factor': pf
    })

team_games = pd.DataFrame(home_rows + away_rows).sort_values(by=['team_id', 'game_date']).reset_index(drop=True)
grouped = team_games.groupby('team_id')

# A. 累積得分/失分與畢氏勝率
team_games['cum_r'] = grouped['runs'].apply(lambda x: x.cumsum() - x).reset_index(level=0, drop=True)
team_games['cum_ra'] = grouped['runs_allowed'].apply(lambda x: x.cumsum() - x).reset_index(level=0, drop=True)
denom = team_games['cum_r']**1.83 + team_games['cum_ra']**1.83
team_games['pyth_win_pct'] = (team_games['cum_r']**1.83 / denom.replace(0, np.nan)).fillna(0.5).astype(float)

# B. 【貝氏均值收縮 (Bayesian Shrinkage)】修正近況得分 (收縮權重 w=0.65, 聯盟平均=4.4分)
LEAGUE_AVG_RUNS = 4.4
team_games['raw_rolling_r_5'] = grouped['runs'].apply(lambda x: x.rolling(5).mean().shift(1)).reset_index(level=0, drop=True)
team_games['raw_rolling_ra_5'] = grouped['runs_allowed'].apply(lambda x: x.rolling(5).mean().shift(1)).reset_index(level=0, drop=True)

team_games['shrunk_r_5'] = (0.65 * team_games['raw_rolling_r_5'] + 0.35 * LEAGUE_AVG_RUNS).fillna(LEAGUE_AVG_RUNS)
team_games['shrunk_ra_5'] = (0.65 * team_games['raw_rolling_ra_5'] + 0.35 * LEAGUE_AVG_RUNS).fillna(LEAGUE_AVG_RUNS)

# C. 【先發投手近況壓制力】
team_games['sp_recent_ra'] = team_games.groupby(['team_id', 'sp_name'])['runs_allowed'].apply(
    lambda x: x.rolling(3, min_periods=1).mean().shift(1)
).reset_index(level=[0, 1], drop=True).fillna(4.4)

# D. 【牛棚疲勞度指標 (Bullpen Fatigue)】: 計算近 3 天內出賽場數與密集度
def calc_fatigue(dates):
    fatigue = []
    date_list = list(dates)
    for i in range(len(date_list)):
        if i == 0:
            fatigue.append(0)
            continue
        cur_date = date_list[i]
        # 統計前 3 天內有幾天出賽
        prev_dates = [d for d in date_list[max(0, i-4):i] if (cur_date - d).days <= 3]
        fatigue.append(len(prev_dates))
    return pd.Series(fatigue, index=dates.index)

team_games['bullpen_fatigue'] = grouped['game_date'].apply(calc_fatigue).reset_index(level=0, drop=True)

# -------------------------------------------------------------
# 4. 合併主客對戰矩陣
# -------------------------------------------------------------
home_side = team_games[team_games['is_home'] == 1].copy()
away_side = team_games[team_games['is_home'] == 0].copy()

matchups = pd.merge(home_side, away_side, on='game_id', suffixes=('_home', '_away')).dropna(
    subset=['raw_rolling_r_5_home', 'raw_rolling_r_5_away']
).copy()

# 計算進階戰力差值特徵
matchups['diff_pyth'] = matchups['pyth_win_pct_home'] - matchups['pyth_win_pct_away']
matchups['diff_shrunk_r'] = matchups['shrunk_r_5_home'] - matchups['shrunk_r_5_away']
matchups['diff_shrunk_ra'] = matchups['shrunk_ra_5_home'] - matchups['shrunk_ra_5_away']
matchups['diff_sp_ra'] = matchups['sp_recent_ra_home'] - matchups['sp_recent_ra_away']
matchups['diff_bullpen_fatigue'] = matchups['bullpen_fatigue_home'] - matchups['bullpen_fatigue_away']
matchups['park_factor'] = matchups['park_factor_home']

features = [
    'diff_pyth',
    'diff_shrunk_r',
    'diff_shrunk_ra',
    'diff_sp_ra',
    'diff_bullpen_fatigue',
    'park_factor',
    'shrunk_r_5_home',
    'shrunk_r_5_away'
]

X = matchups[features]
y = matchups['win_home']

# -------------------------------------------------------------
# 5. 執行時間序列嚴格回測 (前 75% 訓練，後 25% 回測)
# -------------------------------------------------------------
print("\n[步驟 3/5] 正在訓練 XGBoost 分類器...")
split_idx = int(len(matchups) * 0.75)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_matchups = matchups.iloc[split_idx:].copy()

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)
model.fit(X_train, y_train)

# 取得預測機率
probs = model.predict_proba(X_test)[:, 1] # 主隊勝率機率值
preds = (probs >= 0.5).astype(int)
accuracy = accuracy_score(y_test, preds)
brier = brier_score_loss(y_test, probs)

# -------------------------------------------------------------
# 6. 【運彩期望值 (EV) 與凱利公式策略回測】
# -------------------------------------------------------------
print("\n[步驟 4/5] 正在進行運彩期望值 (+EV) 與資金管理回測...")

# 模擬市場客觀標準盤賠率 (莊家抽取約 4.5% 水錢，基礎賠率 1.91)
MARKET_ODDS = 1.91
test_matchups['pred_home_prob'] = probs
test_matchups['actual_win'] = y_test.values

# 計算主隊投注期望值: EV = (Prob * Odds) - 1
test_matchups['ev_home'] = (test_matchups['pred_home_prob'] * MARKET_ODDS) - 1

# 策略 1：全部預測為贏的就下注 (盲跟策略)
blind_bets = test_matchups[test_matchups['pred_home_prob'] >= 0.5]
blind_profit = (blind_bets['actual_win'] * MARKET_ODDS - 1).sum()
blind_roi = (blind_profit / len(blind_bets)) * 100 if len(blind_bets) > 0 else 0

# 策略 2：只有高期望值 (EV >= 5%) 才出手 (量化價值投資策略)
value_bets = test_matchups[test_matchups['ev_home'] >= 0.05]
value_wins = value_bets['actual_win'].sum()
value_profit = (value_bets['actual_win'] * MARKET_ODDS - 1).sum()
value_roi = (value_profit / len(value_bets)) * 100 if len(value_bets) > 0 else 0

# -------------------------------------------------------------
# 7. 輸出白話回測分析報表
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" 📊 MLB 五大進階量化模型回測成果報告")
print("=" * 65)
print(f"回測評估總場次 ：{len(y_test)} 場 (2024 賽季末關鍵衝刺期)")
print(f"整體勝負預測準率：{accuracy * 100:.2f} %")
print(f"機率校準度 (Brier Score)：{brier:.4f} (越接近 0 代表機率估算越精準)")

print("\n" + "-" * 65)
print(" 💰 實戰運彩投注策略回測比較 (以每注 1000 元模擬)")
print("-" * 65)
print(f"【策略 A：全場次預測投注 (無過濾)】")
print(f"  - 下注場次：{len(blind_bets)} 場")
print(f"  - 總損益 ：{blind_profit * 1000:+.0f} 元 ｜ 投資報酬率 (ROI)：{blind_roi:+.2f} %")

print(f"\n【策略 B：嚴格 +EV 價值投資策略 (僅在預期報酬率 >= 5% 時出手)】")
print(f"  - 篩選下注：{len(value_bets)} 場 (從 {len(y_test)} 場中過濾出最高信心賽事)")
print(f"  - 命中場數：{value_wins} 勝 {len(value_bets)-value_wins} 敗 (命中率: {value_wins/len(value_bets)*100:.1f}%)")
print(f"  - 總損益 ：{value_profit * 1000:+.0f} 元 ｜ 投資報酬率 (ROI)：{value_roi:+.2f} %")

print("\n" + "-" * 65)
print(" 🧠 各進階特徵在 AI 決策中的影響力權重 (Feature Importance)")
print("-" * 65)
feat_map = {
    'diff_pyth': '畢氏勝率實力差 (強弱底蘊)',
    'diff_shrunk_r': '貝氏收縮得分差 (打擊火力)',
    'diff_shrunk_ra': '貝氏收縮失分差 (防守底線)',
    'diff_sp_ra': '先發投手近況壓制差 (先發對抗)',
    'diff_bullpen_fatigue': '牛棚疲勞度差異 (後援戰力消耗)',
    'park_factor': '球場效應加權係數 (球場環境影響)',
    'shrunk_r_5_home': '主隊自身進攻火力期望',
    'shrunk_r_5_away': '客隊自身進攻火力期望'
}

for f, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 30)
    print(f"{feat_map.get(f, f):<36} | {imp*100:5.1f}% {bar}")
print("=" * 65)