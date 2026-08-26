import os
import statsapi
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="MLB 智慧量化預測系統", layout="wide")

# -------------------------------------------------------------
# 1. 訓練並載入五大進階維度模型
# -------------------------------------------------------------
@st.cache_resource
def load_and_train_advanced_model():
    csv_cache_file = "mlb_2024_schedule.csv"
    if os.path.exists(csv_cache_file):
        df_raw = pd.read_csv(csv_cache_file)
    else:
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

    PARK_FACTORS = {
        'Coors Field': 1.25, 'Fenway Park': 1.10, 'Great American Ball Park': 1.08,
        'Kauffman Stadium': 1.06, 'Globe Life Field': 1.04, 'Yankee Stadium': 1.03,
        'Dodger Stadium': 0.98, 'Oracle Park': 0.94, 'T-Mobile Park': 0.91,
        'Petco Park': 0.90, 'Citi Field': 0.92, 'Tropicana Field': 0.92
    }

    home_rows, away_rows = [], []
    for _, row in df_games.iterrows():
        pf = PARK_FACTORS.get(row.get('venue_name', 'Unknown'), 1.0)
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

    team_games['cum_r'] = grouped['runs'].apply(lambda x: x.cumsum() - x).reset_index(level=0, drop=True)
    team_games['cum_ra'] = grouped['runs_allowed'].apply(lambda x: x.cumsum() - x).reset_index(level=0, drop=True)
    denom = team_games['cum_r']**1.83 + team_games['cum_ra']**1.83
    team_games['pyth_win_pct'] = (team_games['cum_r']**1.83 / denom.replace(0, np.nan)).fillna(0.5).astype(float)

    LEAGUE_AVG = 4.4
    raw_r = grouped['runs'].apply(lambda x: x.rolling(5).mean().shift(1)).reset_index(level=0, drop=True)
    raw_ra = grouped['runs_allowed'].apply(lambda x: x.rolling(5).mean().shift(1)).reset_index(level=0, drop=True)
    team_games['shrunk_r_5'] = (0.65 * raw_r + 0.35 * LEAGUE_AVG).fillna(LEAGUE_AVG)
    team_games['shrunk_ra_5'] = (0.65 * raw_ra + 0.35 * LEAGUE_AVG).fillna(LEAGUE_AVG)

    team_games['sp_recent_ra'] = team_games.groupby(['team_id', 'sp_name'])['runs_allowed'].apply(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).reset_index(level=[0, 1], drop=True).fillna(LEAGUE_AVG)

    def calc_fatigue(dates):
        fatigue = []
        d_list = list(dates)
        for i in range(len(d_list)):
            if i == 0:
                fatigue.append(0)
                continue
            cur = d_list[i]
            prev = [d for d in d_list[max(0, i-4):i] if (cur - d).days <= 3]
            fatigue.append(len(prev))
        return pd.Series(fatigue, index=dates.index)

    team_games['bullpen_fatigue'] = grouped['game_date'].apply(calc_fatigue).reset_index(level=0, drop=True)

    home_side = team_games[team_games['is_home'] == 1]
    away_side = team_games[team_games['is_home'] == 0]

    matchups = pd.merge(home_side, away_side, on='game_id', suffixes=('_home', '_away')).dropna().copy()
    matchups['diff_pyth'] = matchups['pyth_win_pct_home'] - matchups['pyth_win_pct_away']
    matchups['diff_shrunk_r'] = matchups['shrunk_r_5_home'] - matchups['shrunk_r_5_away']
    matchups['diff_shrunk_ra'] = matchups['shrunk_ra_5_home'] - matchups['shrunk_ra_5_away']
    matchups['diff_sp_ra'] = matchups['sp_recent_ra_home'] - matchups['sp_recent_ra_away']
    matchups['diff_bullpen_fatigue'] = matchups['bullpen_fatigue_home'] - matchups['bullpen_fatigue_away']
    matchups['park_factor'] = matchups['park_factor_home']

    features = [
        'diff_pyth', 'diff_shrunk_r', 'diff_shrunk_ra', 'diff_sp_ra',
        'diff_bullpen_fatigue', 'park_factor', 'shrunk_r_5_home', 'shrunk_r_5_away'
    ]
    X = matchups[features]
    y = matchups['win_home']

    model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.02, max_depth=3, subsample=0.85, colsample_bytree=0.85, random_state=42)
    model.fit(X, y)

    latest_stats = {}
    for t_id, group in team_games.groupby('team_id'):
        last_row = group.iloc[-1]
        latest_stats[t_id] = {
            'team_name': last_row['team_name'],
            'pyth_win_pct': last_row['pyth_win_pct'],
            'shrunk_r_5': last_row['shrunk_r_5'],
            'shrunk_ra_5': last_row['shrunk_ra_5'],
            'bullpen_fatigue': last_row['bullpen_fatigue']
        }

    return model, latest_stats, PARK_FACTORS

# -------------------------------------------------------------
# 2. 蒙地卡羅模擬器
# -------------------------------------------------------------
def run_monte_carlo(home_exp_runs, away_exp_runs, n_simulations=10000):
    home_scores = np.random.poisson(lam=home_exp_runs, size=n_simulations)
    away_scores = np.random.poisson(lam=away_exp_runs, size=n_simulations)

    ties = home_scores == away_scores
    tie_breakers = np.random.choice([1, 0], size=np.sum(ties))
    home_scores[ties] += tie_breakers
    away_scores[ties] += (1 - tie_breakers)

    home_wins = np.sum(home_scores > away_scores)
    home_cover_minus_1_5 = np.sum((home_scores - away_scores) >= 2)
    away_cover_plus_1_5 = np.sum((away_scores - home_scores) >= -1)
    
    total_scores = home_scores + away_scores
    over_8_5 = np.sum(total_scores > 8.5)

    return {
        'home_win_rate': home_wins / n_simulations,
        'away_win_rate': (n_simulations - home_wins) / n_simulations,
        'home_cover_minus_1_5': home_cover_minus_1_5 / n_simulations,
        'away_cover_plus_1_5': away_cover_plus_1_5 / n_simulations,
        'over_8_5_rate': over_8_5 / n_simulations,
        'under_8_5_rate': (n_simulations - over_8_5) / n_simulations,
        'home_scores': home_scores,
        'away_scores': away_scores,
        'total_scores': total_scores
    }

# -------------------------------------------------------------
# 3. 期望值與凱利公式計算函數
# -------------------------------------------------------------
def calculate_ev_and_signal(prob, odds=1.91):
    ev = (prob * odds) - 1
    # 凱利公式計算資金下注比例: (bp - q) / b, 其中 b = odds - 1, q = 1 - prob
    b = odds - 1
    q = 1 - prob
    kelly_pct = max(0.0, (b * prob - q) / b)
    
    if ev >= 0.05:
        signal = "🟢 強烈建議進場 (+EV)"
        color = "green"
    elif ev >= 0.0:
        signal = "🟡 中性觀望 (微幅優勢)"
        color = "orange"
    else:
        signal = "🔴 不建議出手 (-EV)"
        color = "red"
        
    return ev, kelly_pct, signal, color

# -------------------------------------------------------------
# 4. 前端介面呈現
# -------------------------------------------------------------
st.title("⚾ MLB 智慧量化預測與模擬系統")
st.caption("整合五大進階特徵 ＋ 蒙地卡羅模擬 ＋ 運彩 +EV 價值下注訊號燈")

with st.spinner("正在初始化 AI 模型與數據庫..."):
    model, latest_stats, PARK_FACTORS = load_and_train_advanced_model()

# 【功能 1：預設為今日日期】
st.sidebar.header("🔍 選擇預測賽程")
selected_date = st.sidebar.date_input("選擇比賽日期", datetime.date.today())
date_str = selected_date.strftime("%Y-%m-%d")

day_schedule = statsapi.schedule(date=date_str)

if not day_schedule:
    st.warning(f"📅 日期 {date_str} 沒有安排賽事或尚未有賽程資料。你可以切換日期（如 2024-09-25）進行歷史回測！")
else:
    game_options = [f"{g['away_name']} @ {g['home_name']}" for g in day_schedule]
    selected_matchup = st.sidebar.selectbox("選擇對戰組合", game_options)
    
    game_idx = game_options.index(selected_matchup)
    game = day_schedule[game_idx]

    h_id, a_id = game['home_id'], game['away_id']
    h_name, a_name = game['home_name'], game['away_name']
    venue = game.get('venue_name', 'Unknown')
    pf = PARK_FACTORS.get(venue, 1.0)

    h_stat = latest_stats.get(h_id, {'pyth_win_pct': 0.5, 'shrunk_r_5': 4.4, 'shrunk_ra_5': 4.4, 'bullpen_fatigue': 1})
    a_stat = latest_stats.get(a_id, {'pyth_win_pct': 0.5, 'shrunk_r_5': 4.4, 'shrunk_ra_5': 4.4, 'bullpen_fatigue': 1})

    diff_pyth = h_stat['pyth_win_pct'] - a_stat['pyth_win_pct']
    diff_r = h_stat['shrunk_r_5'] - a_stat['shrunk_r_5']
    diff_ra = h_stat['shrunk_ra_5'] - a_stat['shrunk_ra_5']
    diff_sp = 0.0
    diff_fatigue = h_stat['bullpen_fatigue'] - a_stat['bullpen_fatigue']

    feature_vector = np.array([[diff_pyth, diff_r, diff_ra, diff_sp, diff_fatigue, pf, h_stat['shrunk_r_5'], a_stat['shrunk_r_5']]])

    xgb_probs = model.predict_proba(feature_vector)[0]
    xgb_away_win_prob, xgb_home_win_prob = xgb_probs[0], xgb_probs[1]

    home_expected_runs = ((h_stat['shrunk_r_5'] + a_stat['shrunk_ra_5']) / 2) * pf
    away_expected_runs = ((a_stat['shrunk_r_5'] + h_stat['shrunk_ra_5']) / 2) * pf
    mc_results = run_monte_carlo(home_expected_runs, away_expected_runs)

    st.subheader(f"🏟️ 對戰組合：{a_name} (客) vs {h_name} (主)")

    # 完賽比分檢驗面板
    if game['status'] == 'Final':
        actual_away_score = int(game['away_score'])
        actual_home_score = int(game['home_score'])
        actual_winner = h_name if actual_home_score > actual_away_score else a_name
        pred_winner = h_name if xgb_home_win_prob > 0.5 else a_name
        is_correct = (actual_winner == pred_winner)
        result_badge = "✅ AI 預測命中！" if is_correct else "❌ AI 預測失誤"
        st.success(f"🏁 **比賽已結束** ｜ 最終比分：**{a_name} {actual_away_score} : {actual_home_score} {h_name}** （{result_badge}）")
    else:
        st.info(f"⏳ **賽事狀態**：{game['status']}（比賽尚未結束）")

    # 關鍵指標卡片
    col1, col2, col3 = st.columns(3)
    col1.metric(f"🏠 主隊 ({h_name}) 勝率", f"{xgb_home_win_prob * 100:.1f} %")
    col2.metric(f"✈️ 客隊 ({a_name}) 勝率", f"{xgb_away_win_prob * 100:.1f} %")
    col3.metric("🎯 球場修正總期望得分", f"{home_expected_runs + away_expected_runs:.2f} 分", help=f"球場: {venue} (係數: {pf})")

    st.markdown("---")

    # 【功能 2：運彩 +EV 價值下注訊號燈】
    st.markdown("### 💡 運彩量化價值下注訊號 (+EV Signal)")
    ev_col1, ev_col2, ev_col3 = st.columns(3)

    # 1. 主隊獨贏
    h_ev, h_kelly, h_sig, _ = calculate_ev_and_signal(xgb_home_win_prob)
    with ev_col1:
        st.markdown(f"**🏠 主勝盤口 ({h_name})**")
        st.write(f"訊號狀態：{h_sig}")
        st.write(f"預期報酬 (EV)：`{h_ev*100:+.2f}%`")
        if h_kelly > 0:
            st.write(f"凱利建議部位：`{h_kelly*100:.1f}%` 本金")

    # 2. 客隊獨贏
    a_ev, a_kelly, a_sig, _ = calculate_ev_and_signal(xgb_away_win_prob)
    with ev_col2:
        st.markdown(f"**✈️ 客勝盤口 ({a_name})**")
        st.write(f"訊號狀態：{a_sig}")
        st.write(f"預期報酬 (EV)：`{a_ev*100:+.2f}%`")
        if a_kelly > 0:
            st.write(f"凱利建議部位：`{a_kelly*100:.1f}%` 本金")

    # 3. 大小分訊號 (以 8.5 分盤為例)
    over_prob = mc_results['over_8_5_rate']
    o_ev, o_kelly, o_sig, _ = calculate_ev_and_signal(over_prob)
    with ev_col3:
        st.markdown("**🔥 8.5 大分盤口**")
        st.write(f"訊號狀態：{o_sig}")
        st.write(f"預期報酬 (EV)：`{o_ev*100:+.2f}%`")
        if o_kelly > 0:
            st.write(f"凱利建議部位：`{o_kelly*100:.1f}%` 本金")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🎲 蒙地卡羅 10,000 次模擬結果")
        st.write(f"- **主隊讓分盤 ({h_name} -1.5)** 過盤率：`{mc_results['home_cover_minus_1_5']*100:.1f}%`")
        st.write(f"- **客隊受讓盤 ({a_name} +1.5)** 過盤率：`{mc_results['away_cover_plus_1_5']*100:.1f}%`")
        st.write(f"- **大小分 (8.5 分盤口)**：大分 `{mc_results['over_8_5_rate']*100:.1f}%` ｜ 小分 `{mc_results['under_8_5_rate']*100:.1f}%`")

        df_sim = pd.DataFrame({
            '得分': np.concatenate([mc_results['home_scores'], mc_results['away_scores']]),
            '球隊': [h_name] * 10000 + [a_name] * 10000
        })
        fig_score = px.histogram(df_sim, x='得分', color='球隊', barmode='overlay', title="10,000 場虛擬模擬得分分佈")
        st.plotly_chart(fig_score, use_container_width=True)

    with col_right:
        st.markdown("### 📊 雙方進階量化指標對比")
        comparison_df = pd.DataFrame({
            '指標項目': ['累積畢氏勝率', '貝氏修正預期得分', '貝氏修正預期失分', '近 3 天出賽場次 (牛棚疲勞)'],
            f'主隊 ({h_name})': [f"{h_stat['pyth_win_pct']*100:.1f}%", f"{h_stat['shrunk_r_5']:.2f}", f"{h_stat['shrunk_ra_5']:.2f}", f"{h_stat['bullpen_fatigue']} 場"],
            f'客隊 ({a_name})': [f"{a_stat['pyth_win_pct']*100:.1f}%", f"{a_stat['shrunk_r_5']:.2f}", f"{a_stat['shrunk_ra_5']:.2f}", f"{a_stat['bullpen_fatigue']} 場"]
        })
        st.table(comparison_df)

        fig_tot = px.histogram(x=mc_results['total_scores'], nbins=20, labels={'x': '兩隊合計總分'}, title="兩隊合計總得分機率分佈 (大小分參考)")
        st.plotly_chart(fig_tot, use_container_width=True)