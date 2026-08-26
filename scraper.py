import os
import statsapi
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

print("=" * 55)
print(" 🚀 MLB 機器學習勝負預測系統 (全聯盟多維度進階版)")
print("=" * 55)

# -------------------------------------------------------------
# 【步驟 1/4】：按月分批抓取全聯盟賽程，並加入本地 CSV 快取（避免 503 逾時）
# -------------------------------------------------------------
csv_cache_file = "mlb_2024_schedule.csv"

if os.path.exists(csv_cache_file):
    print(f"\n[步驟 1/4] 發現本地快取檔案 {csv_cache_file}，直接載入資料...")
    df_raw = pd.read_csv(csv_cache_file)
else:
    print("\n[步驟 1/4] 正在按月分批從 MLB API 下載賽程資料...")
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
        print(f" -> 正在下載 {start} 至 {end} 的比賽紀錄...")
        month_data = statsapi.schedule(start_date=start, end_date=end)
        all_games.extend(month_data)
        
    df_raw = pd.DataFrame(all_games)
    df_raw.to_csv(csv_cache_file, index=False)
    print(f" -> 下載完成並已儲存至 {csv_cache_file}！")

# 只保留已完賽的例行賽
df_games = df_raw[(df_raw['status'] == 'Final') & (df_raw['game_type'] == 'R')].copy()
df_games = df_games.sort_values(by=['game_date', 'game_id']).reset_index(drop=True)
print(f"-> 成功載入 {len(df_games)} 場已完賽的例行賽紀錄！")

# -------------------------------------------------------------
# 【步驟 2/4】：資料重構：拆解比賽維度與球隊視角
# -------------------------------------------------------------
print("\n[步驟 2/4] 正在拆解比賽維度與球隊視角...")

home_rows = []
away_rows = []

for _, row in df_games.iterrows():
    home_rows.append({
        'game_id': row['game_id'],
        'game_date': row['game_date'],
        'team_id': row['home_id'],
        'team_name': row['home_name'],
        'opp_id': row['away_id'],
        'opp_name': row['away_name'],
        'is_home': 1,
        'runs': pd.to_numeric(row['home_score'], errors='coerce'),
        'runs_allowed': pd.to_numeric(row['away_score'], errors='coerce'),
        'win': 1 if row['home_score'] > row['away_score'] else 0,
        'sp_name': row.get('home_probable_pitcher', 'Unknown')
    })
    away_rows.append({
        'game_id': row['game_id'],
        'game_date': row['game_date'],
        'team_id': row['away_id'],
        'team_name': row['away_name'],
        'opp_id': row['home_id'],
        'opp_name': row['home_name'],
        'is_home': 0,
        'runs': pd.to_numeric(row['away_score'], errors='coerce'),
        'runs_allowed': pd.to_numeric(row['home_score'], errors='coerce'),
        'win': 1 if row['away_score'] > row['home_score'] else 0,
        'sp_name': row.get('away_probable_pitcher', 'Unknown')
    })

team_games = pd.DataFrame(home_rows + away_rows)
team_games = team_games.sort_values(by=['team_id', 'game_date']).reset_index(drop=True)

# -------------------------------------------------------------
# 【步驟 3/4】：特徵工程（球隊滾動狀態、畢氏勝率、先發投手狀態）
# -------------------------------------------------------------
print("\n[步驟 3/4] 正在進行進階特徵工程 (防範未來數據污染)...")

grouped = team_games.groupby('team_id')

team_games['cum_r'] = grouped['runs'].apply(lambda x: x.cumsum() - x).reset_index(level=0, drop=True)
team_games['cum_ra'] = grouped['runs_allowed'].apply(lambda x: x.cumsum() - x).reset_index(level=0, drop=True)

denom = team_games['cum_r']**1.83 + team_games['cum_ra']**1.83
safe_denom = denom.replace(0, np.nan)
team_games['pyth_win_pct'] = (team_games['cum_r']**1.83 / safe_denom).fillna(0.5).astype(float)

team_games['rolling_r_5'] = grouped['runs'].apply(lambda x: x.rolling(5).mean().shift(1)).reset_index(level=0, drop=True)
team_games['rolling_ra_5'] = grouped['runs_allowed'].apply(lambda x: x.rolling(5).mean().shift(1)).reset_index(level=0, drop=True)

team_games['sp_rolling_ra_3'] = team_games.groupby(['team_id', 'sp_name'])['runs_allowed'].apply(
    lambda x: x.rolling(3, min_periods=1).mean().shift(1)
).reset_index(level=[0, 1], drop=True)

team_games['sp_rolling_ra_3'] = team_games['sp_rolling_ra_3'].fillna(4.5)

home_side = team_games[team_games['is_home'] == 1].copy()
away_side = team_games[team_games['is_home'] == 0].copy()

matchups = pd.merge(
    home_side,
    away_side,
    on='game_id',
    suffixes=('_home', '_away')
)

matchups = matchups.dropna(subset=['rolling_r_5_home', 'rolling_ra_5_home', 'rolling_r_5_away', 'rolling_ra_5_away']).copy()

matchups['diff_pyth_win_pct'] = matchups['pyth_win_pct_home'] - matchups['pyth_win_pct_away']
matchups['diff_rolling_r'] = matchups['rolling_r_5_home'] - matchups['rolling_r_5_away']
matchups['diff_rolling_ra'] = matchups['rolling_ra_5_home'] - matchups['rolling_ra_5_away']
matchups['diff_sp_ra'] = matchups['sp_rolling_ra_3_home'] - matchups['sp_rolling_ra_3_away']

features = [
    'diff_pyth_win_pct',
    'diff_rolling_r',
    'diff_rolling_ra',
    'diff_sp_ra',
    'rolling_r_5_home',
    'rolling_r_5_away'
]

X = matchups[features]
y = matchups['win_home']

# -------------------------------------------------------------
# 【步驟 4/4】：訓練與評估 (繁體中文白話輸出)
# -------------------------------------------------------------
print("\n[步驟 4/4] 正在訓練 XGBoost 分類器並進行勝負回測...")

split_idx = int(len(matchups) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

total_games = len(y_test)
correct_games = int(accuracy * total_games)
actual_home_wins = int(sum(y_test))
actual_away_wins = total_games - actual_home_wins
pred_home_wins = int(sum(preds))
pred_away_wins = total_games - pred_home_wins

print("\n" + "="*55)
print(" 📊 MLB 全聯盟比賽勝負預測評估報告 (XGBoost)")
print("="*55)
print(f"測試集評估場次 ：{total_games} 場 (涵蓋季末全大聯盟關鍵戰役)")
print(f"實際主客勝負統計：主場勝 {actual_home_wins} 場 ｜ 客場勝 {actual_away_wins} 場")
print(f"模型預測主客勝負：預測主勝 {pred_home_wins} 場 ｜ 預測客勝 {pred_away_wins} 場")
print(f"模型精準命中場次：{correct_games} 場")
print(f"🌟 最終測試集準確率：{accuracy * 100:.2f} %")

print("\n" + "-"*55)
print(" 🧠 各項數據對 AI 預測勝負的「影響力權重排行榜」")
print("-"*55)

feature_cn_names = {
    'diff_pyth_win_pct': '雙方累積畢氏勝率差距 (強弱底蘊差距)',
    'diff_rolling_r':    '雙方打線近況得分差 (打擊手感對比)',
    'diff_rolling_ra':   '雙方守備近況失分差 (防守狀態對比)',
    'diff_sp_ra':        '雙方先發投手近況差 (先發壓制力對比)',
    'rolling_r_5_home':  '主隊自身近 5 場得分力道',
    'rolling_r_5_away':  '客隊自身近 5 場得分力道'
}

for feature, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 30)
    name = feature_cn_names.get(feature, feature)
    print(f"{name:<34} | {imp*100:5.1f}% {bar}")

print("="*55)