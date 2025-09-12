import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import itertools
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import pingouin as pg

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="CRC 6‑Rater Agreement Analyzer", layout="wide")
st.title("CRC 6‑Rater Agreement Analyzer")
st.caption("6명의 평가자가 5점 척도로 평가한 ① AI vs MDT, ② AI vs Guideline, ③ MDT vs Guideline 의 일치도·신뢰도와 불일치 위험요인(다변량 로지스틱)을 자동 계산합니다.")

# =========================
# Utilities & Statistical Functions
# =========================

@st.cache_data
def read_excel(file):
    """Uploaded Excel 파일을 Pandas DataFrame으로 읽습니다."""
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"엑셀 파일 읽기 오류: {e}")
        return None

def safe_num(s):
    """문자열을 숫자형으로 안전하게 변환합니다. 변환 실패 시 NaT/NaN을 반환합니다."""
    return pd.to_numeric(s, errors="coerce")

def make_distance_matrix(k: int, scheme: str = "quadratic") -> np.ndarray:
    """가중 카파(Weighted Kappa) 계산을 위한 거리 행렬을 생성합니다."""
    idx = np.arange(k)
    D = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            d = abs(i - j) / (k - 1) if k > 1 else 0.0
            if scheme == "linear":
                D[i, j] = d
            elif scheme == "quadratic":
                D[i, j] = d ** 2
            else: # nominal / unweighted
                D[i, j] = 0.0 if i == j else 1.0
    return D

def weighted_kappa_custom(a: pd.Series, b: pd.Series, labels, D: np.ndarray) -> float:
    """두 평가자 간의 가중 카파(Weighted Kappa)를 계산합니다."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.empty:
        return np.nan
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    ai = df["a"].map(lab_to_i)
    bi = df["b"].map(lab_to_i)
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for i_, j_ in zip(ai, bi):
        if pd.isna(i_) or pd.isna(j_):
            continue
        cm[int(i_), int(j_)] += 1
    n = cm.sum()
    if n == 0: return np.nan
    r = cm.sum(axis=1)
    c = cm.sum(axis=0)
    Do = (D * cm).sum()
    E = np.outer(r, c) / n
    De = (D * E).sum()
    if De == 0: return np.nan
    return float(1.0 - Do / De)

def bootstrap_ci_stat(stat_fun, n_items: int, B: int = 2000, seed: int = 42):
    """통계량의 부트스트랩 신뢰구간을 계산합니다."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n_items, n_items)
        vals.append(stat_fun(idx))
    vals = np.array(vals, dtype=float)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    p_boot = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(lo), float(hi), float(p_boot), vals

def fleiss_kappa(ratings_df: pd.DataFrame, B: int = 2000, seed: int = 42):
    """다수 평가자 간의 Fleiss' Kappa를 계산합니다 (비가중치, 결측치 허용)."""
    X = ratings_df.copy()
    if X.empty:
        return np.nan, np.nan, np.nan, np.array([])
    
    cats = sorted([c for c in pd.unique(X.values.ravel()) if pd.notna(c)])
    if not cats: return np.nan, np.nan, np.nan, np.array([])
    
    m = len(cats)
    cat_to_i = {c: i for i, c in enumerate(cats)}
    n_items = X.shape[0]

    def calculate_kappa(df_sub):
        n_items_sub = df_sub.shape[0]
        if n_items_sub == 0: return np.nan

        N = np.zeros((n_items_sub, m), dtype=float)
        for u in range(n_items_sub):
            vc = pd.Series(df_sub.iloc[u, :]).dropna().value_counts()
            for c, cnt in vc.items():
                if c in cat_to_i:
                    N[u, cat_to_i[c]] = cnt
        
        n_i = N.sum(axis=1)
        valid_items_mask = n_i >= 2
        if not np.any(valid_items_mask): return np.nan
        
        N = N[valid_items_mask, :]
        n_i = n_i[valid_items_mask]
        
        Pu = (np.sum(N * (N - 1), axis=1)) / (n_i * (n_i - 1))
        P_bar = np.nanmean(Pu)
        
        total_ratings = np.sum(n_i)
        if total_ratings == 0: return np.nan
        
        pj = np.sum(N, axis=0) / total_ratings
        Pe = np.sum(pj ** 2)
        
        if (1 - Pe) == 0: return np.nan
        kappa = (P_bar - Pe) / (1 - Pe)
        return kappa

    kappa_0 = calculate_kappa(X)
    rng = np.random.default_rng(seed)
    boots = [calculate_kappa(X.iloc[rng.integers(0, n_items, n_items)]) for _ in range(B)]
    boots = np.array(boots, dtype=float)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(kappa_0), float(lo), float(hi), boots

def krippendorff_alpha(ratings_df: pd.DataFrame, B: int = 2000, seed: int = 42):
    """다수 평가자 간의 Krippendorff's Alpha를 계산합니다 (서열척도)."""
    X = ratings_df.copy()
    cats = sorted([c for c in pd.unique(X.values.ravel()) if pd.notna(c)])
    if not cats: return np.nan, np.nan, np.nan, np.array([])
    
    cat_to_i = {c: i for i, c in enumerate(cats)}
    m = len(cats)

    def calculate_alpha(df):
        O = np.zeros((m, m), dtype=float)
        for _, row in df.iterrows():
            vals = [v for v in row.values if pd.notna(v)]
            if len(vals) < 2: continue
            vc = pd.Series(vals).value_counts()
            for i_c, ci in vc.items():
                ii = cat_to_i[i_c]
                O[ii, ii] += ci * (ci - 1)
                for j_c, cj in vc.items():
                    if j_c != i_c:
                        jj = cat_to_i[j_c]
                        O[ii, jj] += ci * cj
        
        if O.sum() == 0: return np.nan
        D = np.zeros((m, m), dtype=float)
        for i in range(m):
            for j in range(m):
                d = abs(i - j) / (m - 1) if m > 1 else 0.0
                D[i, j] = d ** 2
        
        Do = (O * D).sum() / O.sum()
        n_i = O.sum(axis=1)
        E = np.outer(n_i, n_i)
        for i in range(m): E[i, i] = n_i[i] * (n_i[i] - 1)
        if E.sum() == 0: return np.nan
        De = (E * D).sum() / E.sum()
        if De == 0: return np.nan
        return 1.0 - Do / De

    alpha = calculate_alpha(X)
    rng = np.random.default_rng(seed)
    n_items = len(X)
    boots = [calculate_alpha(X.iloc[rng.integers(0, n_items, n_items)]) for _ in range(B)]
    boots = np.array(boots, dtype=float)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(alpha), float(lo), float(hi), boots

def calculate_icc(ratings_df: pd.DataFrame):
    """Intraclass Correlation Coefficient (ICC)를 계산합니다."""
    df_long = ratings_df.reset_index().melt(
        id_vars='index', var_name='rater', value_name='rating'
    ).rename(columns={'index': 'case'}).dropna()
    
    if df_long.empty or df_long['case'].nunique() < 2 or df_long['rater'].nunique() < 2:
        return np.nan, np.nan, np.nan
    try:
        icc = pg.intraclass_corr(data=df_long, targets='case', raters='rater', ratings='rating', nan_policy='omit')
        icc.set_index('Type', inplace=True)
        icc_res = icc.loc['ICC2k'] # Two-way random, mean rating, absolute agreement
        val, ci = icc_res['ICC'], icc_res['CI95%']
        return val, ci[0], ci[1]
    except Exception:
        return np.nan, np.nan, np.nan

def pairwise_kappa_table(raters_df: pd.DataFrame, labels, D: np.ndarray, B: int = 2000, seed: int = 42):
    """모든 평가자 쌍에 대한 가중 카파 테이블을 생성합니다."""
    rows = []
    n = len(raters_df)
    if n == 0: return pd.DataFrame()
    
    for a, b in itertools.combinations(raters_df.columns, 2):
        def stat_fun(idx):
            return weighted_kappa_custom(raters_df[a].iloc[idx], raters_df[b].iloc[idx], labels, D)
        k0 = stat_fun(np.arange(n))
        lo, hi, p_boot, _ = bootstrap_ci_stat(stat_fun, n, B=B, seed=seed)
        pea = float((raters_df[a] == raters_df[b]).mean()) * 100.0
        rows.append({
            "Rater A": a, "Rater B": b,
            "Exact Agreement (%)": f"{pea:.2f}",
            "Weighted Kappa": f"{k0:.4f}",
            "95% CI Lower": f"{lo:.4f}",
            "95% CI Upper": f"{hi:.4f}",
            "P-value (Boot)": f"{p_boot:.4f}"
        })
    return pd.DataFrame(rows)

def lights_kappa(raters_df: pd.DataFrame, labels, D: np.ndarray, B: int = 2000, seed: int = 42):
    """Light's Kappa (모든 쌍별 카파의 평균)를 계산합니다."""
    pairs = list(itertools.combinations(raters_df.columns, 2))
    n = len(raters_df)
    if n == 0: return np.nan, np.nan, np.nan, np.array([])
    def lk_from_idx(idx):
        vals = [weighted_kappa_custom(raters_df[a].iloc[idx], raters_df[b].iloc[idx], labels, D) for a, b in pairs]
        return float(np.nanmean(vals))
    k0 = lk_from_idx(np.arange(n))
    lo, hi, _, boots = bootstrap_ci_stat(lk_from_idx, n, B=B, seed=seed)
    return float(k0), float(lo), float(hi), boots

def fit_logit(df: pd.DataFrame, y_col: str, x_cols, robust=True):
    """로지스틱 회귀 모델을 적합하고 결과를 반환합니다."""
    X = df[x_cols].copy()
    cat_cols = [c for c in x_cols if (not pd.api.types.is_numeric_dtype(X[c])) or X[c].nunique() <= 6]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[X.index, y_col].astype(int)
    
    if len(X) < len(x_cols) + 1:
        raise ValueError("샘플 수가 변수 수보다 적어 모델을 실행할 수 없습니다.")
    if y.nunique() < 2:
        raise ValueError("종속변수(불일치 여부)에 두 가지 이상의 범주가 존재해야 합니다.")

    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X)
    res = model.fit(disp=False, maxiter=200)
    if robust: res = res.get_robustcov_results(cov_type="HC3")
    
    params, conf = res.params, res.conf_int()
    out = pd.DataFrame({
        "Variable": params.index, "OR": np.exp(params),
        "CI Lower": np.exp(conf.iloc[:, 0]), "CI Upper": np.exp(conf.iloc[:, 1]),
        "P-value": res.pvalues
    }).round(4)
    out = out[out["Variable"] != "const"].reset_index(drop=True)
    
    vif_rows, Xv = [], X.drop(columns=["const"], errors="ignore")
    if Xv.shape[1] >= 2:
        for i in range(Xv.shape[1]):
            vif_rows.append({"Variable": Xv.columns[i], "VIF": variance_inflation_factor(Xv.values, i)})
    return res, out, pd.DataFrame(vif_rows)

def infer_labels(raters_df: pd.DataFrame):
    """평가 데이터에서 고유한 라벨(척도)을 추론합니다."""
    vals = pd.unique(raters_df.values.ravel())
    vals = sorted([v for v in vals if pd.notna(v)])
    return vals if vals else [1, 2, 3, 4, 5]

# =========================
# Sidebar – Upload & Settings
# =========================
with st.sidebar:
    st.header("1) 파일 업로드 & 설정")
    up = st.file_uploader("Excel (.xlsx)", type=["xlsx"], help="Wide 형식의 엑셀 파일을 업로드하세요.")
    
    if st.button("샘플 템플릿(.xlsx) 다운로드"):
        n=300; np.random.seed(0)
        def r5(n): return np.random.choice([1,2,3,4,5], n, p=[0.1,0.2,0.4,0.2,0.1])
        df_tmp = pd.DataFrame({ "case_id": [f"C{str(i+1).zfill(3)}" for i in range(n)], "A_M_S1": r5(n), "A_M_S2": r5(n), "A_M_S3": r5(n), "A_M_S4": r5(n), "A_M_S5": r5(n), "A_M_S6": r5(n), "A_G_S1": r5(n), "A_G_S2": r5(n), "A_G_S3": r5(n), "A_G_S4": r5(n), "A_G_S5": r5(n), "A_G_S6": r5(n), "M_G_S1": r5(n), "M_G_S2": r5(n), "M_G_S3": r5(n), "M_G_S4": r5(n), "M_G_S5": r5(n), "M_G_S6": r5(n), "age": np.random.normal(65, 10, n).round().astype(int), "sex": np.random.choice(["M","F"], n), "ASA": np.random.choice([1,2,3,4],n,p=[0.1,0.4,0.4,0.1]),"ECOG":np.random.choice([0,1,2,3],n,p=[0.3,0.4,0.2,0.1]), "T": np.random.choice(["T1","T2","T3","T4"],n,p=[0.1,0.2,0.5,0.2]),"N":np.random.choice(["N0","N1","N2"],n,p=[0.5,0.3,0.2]), "stage":np.random.choice(["I","II","III","IV"],n,p=[0.1,0.4,0.4,0.1]),"PNI":np.random.choice([0,1],n,p=[0.7,0.3]), "EMVI":np.random.choice([0,1],n,p=[0.7,0.3]),"obstruction":np.random.choice([0,1],n,p=[0.7,0.3])})
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w: df_tmp.to_excel(w, index=False, sheet_name="data")
        bio.seek(0)
        st.download_button("sample_template.xlsx 다운로드", data=bio, file_name="sample_template.xlsx", key="download_template")

    st.divider(); st.header("2) 분석 설정")
    B = st.number_input(
        "Bootstrap 반복 횟수", 
        min_value=100, 
        value=2000, 
        step=100, 
        help="신뢰구간 추정 시뮬레이션 횟수. 높을수록 안정적이지만 계산 시간이 길어집니다."
    )
    seed = st.number_input("Random Seed", value=42, min_value=0, step=1, help="분석 재현성을 위한 난수 시드. 같은 시드를 사용하면 항상 동일한 결과가 나옵니다.")
    scheme = st.selectbox("가중 방식 (Kappa)", ["quadratic","linear","unweighted"], 0, help="가중 카파 계산 시 불일치에 대한 페널티 방식입니다. Quadratic은 차이가 클수록 페널티를 더 크게 부여합니다.")
    st.divider()
    if st.button("모든 설정 및 데이터 초기화", type="secondary"): st.session_state.clear(); st.rerun()

if up is None:
    st.info("👈 사이드바에서 엑셀 파일을 업로드하거나 샘플 템플릿을 다운로드하여 시작하세요.")
    st.stop()

# =========================
# Main Panel Logic
# =========================
df = read_excel(up)
if df is None: st.stop()

if 'df_shape' not in st.session_state or st.session_state.df_shape != df.shape:
    st.session_state.clear()
    st.session_state.df_shape = df.shape

st.success(f"파일 로드 완료: {df.shape[0]} 행 × {df.shape[1]} 열")
df.reset_index(inplace=True) # Use index for unique case identification

with st.expander("STEP 1: 분석할 열 선택 (각 비교쌍별 평가자 6명)", expanded=True):
    cols = df.columns.tolist()
    st.session_state.case_id_col = st.selectbox("환자 ID 열", cols, index=cols.index("case_id") if "case_id" in cols else 0, key="case_id_select")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.session_state.am_cols = st.multiselect("**AI vs MDT**", cols, default=[c for c in cols if c.startswith("A_M_S")][:6], key="am_cols_select")
    with c2: st.session_state.ag_cols = st.multiselect("**AI vs Guideline**", cols, default=[c for c in cols if c.startswith("A_G_S")][:6], key="ag_cols_select")
    with c3: st.session_state.mg_cols = st.multiselect("**MDT vs Guideline**", cols, default=[c for c in cols if c.startswith("M_G_S")][:6], key="mg_cols_select")

for c in st.session_state.am_cols + st.session_state.ag_cols + st.session_state.mg_cols:
    if c in df.columns: df[c] = safe_num(df[c])

pair_tabs = st.tabs(["AI vs MDT", "AI vs Guideline", "MDT vs Guideline", "불일치 위험요인 분석"])

def render_pair_block(name: str, cols: list[str]):
    st.subheader(f"📊 {name} — 6인 평가 일치도/신뢰도 분석")
    if len(cols) != 6: st.warning("👈 위에서 평가자 6명의 열을 선택해주세요."); return {}, {}
    
    R = df[cols].copy()
    if R.dropna(how='all').empty: st.error("선택한 열에 유효한 데이터가 없습니다."); return {}, {}
    
    labels, D = infer_labels(R), make_distance_matrix(len(infer_labels(R)), scheme)
    try:
        pair_tab = pairwise_kappa_table(R, labels, D, B=B, seed=seed)
        lk, llo, lhi, _ = lights_kappa(R, labels, D, B=B, seed=seed)
        fk, fk_lo, fk_hi, _ = fleiss_kappa(R, B=B, seed=seed)
        ka, ka_lo, ka_hi, _ = krippendorff_alpha(R, B=B, seed=seed)
        icc, icc_lo, icc_hi = calculate_icc(R)
    except Exception as e: st.error(f"통계 계산 중 오류 발생: {e}"); return {}, {}

    c1, c2 = st.columns(2)
    with c1:
        st.info("쌍별 가중 Kappa (Weighted Kappa)", icon="ℹ️")
        st.caption("모든 평가자 쌍(15개)에 대해 가중 카파를 계산합니다.")
        st.dataframe(pair_tab, use_container_width=True)
        st.info("종합 신뢰도 지표", icon="ℹ️")
        st.caption("ICC: 평가자 간 신뢰도, 평균점수의 신뢰성을 나타냅니다.\nLight's Kappa: 모든 쌍별 가중 카파의 평균입니다.\nFleiss' Kappa: 다수 평가자 간 일치도 (비가중치).\nKrippendorff's Alpha: 유연성이 높은 신뢰도 지표 (서열척도).")
        summary_df = pd.DataFrame([
            {"Metric": "ICC (2,k) - Absolute Agreement", "Value": f"{icc:.4f}", "95% CI": f"({icc_lo:.4f} - {icc_hi:.4f})"},
            {"Metric": "Light's Kappa (weighted)", "Value": f"{lk:.4f}", "95% CI": f"({llo:.4f} - {lhi:.4f})"},
            {"Metric": "Fleiss' Kappa (unweighted)", "Value": f"{fk:.4f}", "95% CI": f"({fk_lo:.4f} - {fk_hi:.4f})"},
            {"Metric": "Krippendorff's Alpha (ordinal)", "Value": f"{ka:.4f}", "95% CI": f"({ka_lo:.4f} - {ka_hi:.4f})"},
        ]).set_index("Metric")
        st.dataframe(summary_df, use_container_width=True)

    with c2:
        st.info("합의 점수(Consensus Score) 분포", icon="ℹ️")
        st.caption("각 케이스별 6인 평가의 중앙값(median)을 합의 점수로 정의하고, 그 분포를 보여줍니다.")
        consensus = R.median(axis=1, skipna=True)
        fig = px.histogram(consensus.dropna(), nbins=len(labels)*2, title=f"<b>{name}: 합의 점수 분포</b>")
        fig.update_layout(showlegend=False, yaxis_title="케이스 수", xaxis_title="합의 점수 (중앙값)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("전체 평점 분포", icon="ℹ️")
        st.caption("모든 평가자와 케이스에 걸쳐 각 점수가 몇 번씩 부여되었는지 보여줍니다.")
        ratings_long = R.melt(var_name='rater', value_name='rating').dropna()
        fig2 = px.histogram(ratings_long, x='rating', title=f"<b>{name}: 전체 평점 분포</b>", category_orders={"rating": labels})
        fig2.update_layout(yaxis_title="빈도 수", xaxis_title="평점")
        st.plotly_chart(fig2, use_container_width=True)


    with st.expander("케이스별 불일치 분석 보기 (Top 10)"):
        st.caption("각 케이스에 대한 6명 평가의 표준편차(StDev)를 계산하여 불일치가 가장 큰 상위 10개 케이스를 보여줍니다.")
        disagreement = R.std(axis=1, skipna=True).sort_values(ascending=False)
        top_10_df = df.loc[disagreement.head(10).index, [st.session_state.case_id_col] + cols]
        top_10_df['Disagreement (StDev)'] = disagreement.head(10)
        st.dataframe(top_10_df.style.format({'Disagreement (StDev)': "{:.4f}"}), use_container_width=True)

    out_tables = { f"{name}_pairwise_kappa": pair_tab, f"{name}_summary_reliability": summary_df.reset_index(), f"{name}_top_disagreement": top_10_df }
    return out_tables, {"consensus": consensus}

with pair_tabs[0]: st.session_state.tables_am, st.session_state.meta_am = render_pair_block("AI_vs_MDT", st.session_state.am_cols)
with pair_tabs[1]: st.session_state.tables_ag, st.session_state.meta_ag = render_pair_block("AI_vs_Guideline", st.session_state.ag_cols)
with pair_tabs[2]: st.session_state.tables_mg, st.session_state.meta_mg = render_pair_block("MDT_vs_Guideline", st.session_state.mg_cols)

with pair_tabs[3]:
    st.subheader("⚡️ 불일치 위험요인 분석 (다변량 로지스틱 회귀분석)")
    meta_map = {"AI_vs_MDT": st.session_state.get('meta_am',{}), "AI_vs_Guideline": st.session_state.get('meta_ag',{}), "MDT_vs_Guideline": st.session_state.get('meta_mg',{})}
    c1,c2 = st.columns(2)
    with c1: pair = st.selectbox("분석할 비교쌍", list(meta_map.keys()), key="logit_pair", help="어떤 비교쌍의 불일치 위험요인을 분석할지 선택합니다.")
    with c2: thr = st.number_input("불일치(Discordance) 정의 임계값", 1, 5, 2, 1, help="합의 점수(중앙값)가 이 값 '이하'일 경우를 불일치(1)로 정의합니다.")

    if 'consensus' not in meta_map[pair]: st.warning(f"먼저 '{pair}' 탭에서 일치도 분석을 실행하여 합의 점수를 계산해야 합니다."); st.stop()
    
    df["_discordant"] = (meta_map[pair]['consensus'] <= thr).astype(int)
    st.info(f"'{pair}' 비교쌍에서 합의 점수가 **{thr} 이하**인 경우를 '불일치'로 정의했습니다. (총 {df['_discordant'].sum()} / {len(df['_discordant'].dropna())} 케이스)")

    excluded = {st.session_state.case_id_col, "_discordant", "index"} | set(st.session_state.am_cols) | set(st.session_state.ag_cols) | set(st.session_state.mg_cols)
    candidates = [c for c in df.columns if c not in excluded]
    default_covars = [c for c in candidates if c in ["age", "sex", "ASA", "ECOG", "stage", "T", "N", "PNI", "EMVI", "obstruction"]]
    covars = st.multiselect("분석에 포함할 공변량(독립변수)", candidates, default=default_covars, key="logit_covars", help="불일치에 영향을 줄 것으로 예상되는 변수들을 선택합니다.")
    robust = st.checkbox("Robust Standard Errors (HC3) 사용", True, help="이상치(outlier) 등에 덜 민감한 강건한 표준오차를 사용하여 p-value를 계산합니다.")

    if st.button("위험요인 분석 실행", type="primary", use_container_width=True) and covars:
        try:
            res, or_tab, vif_df = fit_logit(df.dropna(subset=covars+['_discordant']), "_discordant", covars, robust=robust)
            st.session_state.or_tab, st.session_state.vif_df, st.session_state.logit_summary = or_tab, vif_df, res.summary2().as_text()
        except Exception as e: st.error(f"로지스틱 회귀분석 오류: {e}"); st.session_state.or_tab = None

    if 'or_tab' in st.session_state and st.session_state.or_tab is not None:
        st.markdown("---"); st.subheader("분석 결과")
        c1, c2 = st.columns([0.6, 0.4])
        with c1:
            st.info("Odds Ratio (OR) Plot", icon="📈")
            st.caption("OR이 1보다 크면 불일치 위험을 높이는 요인, 1보다 작으면 낮추는 요인입니다.")
            fig = px.scatter(st.session_state.or_tab, x="OR", y="Variable", error_x="CI Upper", error_x_minus="CI Lower", log_x=True, labels={"Variable":""}, title="<b>불일치 위험요인 Odds Ratios</b>")
            fig.add_vline(x=1, line_dash="dash", line_color="grey"); fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.info("Odds Ratio (OR) Table", icon="📋"); st.dataframe(st.session_state.or_tab.style.format(precision=4), use_container_width=True)
            st.info("다중공선성 (VIF)", icon="⚠️"); st.caption("VIF가 보통 10을 초과하면 다중공선성을 의심할 수 있습니다.")
            st.dataframe(st.session_state.vif_df.sort_values("VIF", ascending=False).style.format({"VIF": "{:.2f}"}), use_container_width=True)
        with st.expander("전체 모델 요약 보기 (Statsmodels)"): st.text(st.session_state.logit_summary)

# =========================
# Export Results
# =========================
st.divider()
st.subheader("📥 결과 다운로드 (Excel)")
def build_results_workbook():
    tables = {}
    for key in ['tables_am', 'tables_ag', 'tables_mg']:
        if key in st.session_state: tables.update(st.session_state[key] or {})
    if 'or_tab' in st.session_state and st.session_state.or_tab is not None:
        tables[f"RiskFactors_{st.session_state.logit_pair}"] = st.session_state.or_tab
        tables[f"VIF_{st.session_state.logit_pair}"] = st.session_state.vif_df
    if not tables: st.warning("다운로드할 분석 결과가 없습니다."); return None

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, table_df in tables.items():
            if isinstance(table_df, pd.DataFrame) and not table_df.empty:
                table_df.to_excel(writer, sheet_name=name[:31], index=False)
    bio.seek(0)
    return bio

try:
    xls_data = build_results_workbook()
    if xls_data:
        st.download_button("통합 분석 결과 (results.xlsx) 다운로드", xls_data, "CRC_Agreement_Analysis_Results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
except Exception as e: st.error(f"결과 파일 생성 중 오류 발생: {e}")

st.caption("© 2025 — CRC 6‑Rater Agreement Analyzer (Enhanced by Gemini)")

