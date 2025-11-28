import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import core modules
import core_concordance
import core_plots

st.set_page_config(page_title="MDT/AI Concordance App", layout="wide")


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------
def _init_session_state():
    """Initialize Streamlit session variables for idempotent reruns."""
    if "users" not in st.session_state:
        # Preload a demo account
        st.session_state.users = {"demo@example.com": "demo123"}
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
    if "df_rec" not in st.session_state:
        st.session_state.df_rec = None
    if "df_conc" not in st.session_state:
        st.session_state.df_conc = None


def _create_sample_data():
    """Generate a small demo dataset for immediate testing."""
    df_rec = pd.DataFrame(
        [
            {
                "case_id": 1,
                "patient_group": "A",
                "guideline_rec": "Surgery",
                "mdt_rec": "Surgery",
                "gpt_rec": "Chemotherapy",
                "discordance_reason_category": "MDT/clinician factor",
                "discordance_reason_detail": "MDT preferred early operation",
            },
            {
                "case_id": 2,
                "patient_group": "A",
                "guideline_rec": "Chemotherapy",
                "mdt_rec": "Chemotherapy",
                "gpt_rec": "Chemotherapy",
                "discordance_reason_category": "",
                "discordance_reason_detail": "",
            },
            {
                "case_id": 3,
                "patient_group": "B",
                "guideline_rec": "Radiation",
                "mdt_rec": "Radiation",
                "gpt_rec": "Radiation",
                "discordance_reason_category": "",
                "discordance_reason_detail": "",
            },
            {
                "case_id": 4,
                "patient_group": "B",
                "guideline_rec": "Observation",
                "mdt_rec": "Observation",
                "gpt_rec": "Surgery",
                "discordance_reason_category": "Guideline-related factor",
                "discordance_reason_detail": "AI suggested earlier intervention",
            },
        ]
    )

    # All raters R1-R6 rate each pair for each case
    rows = []
    pairs = ["MDT vs Guideline", "GPT vs Guideline", "MDT vs GPT"]
    scores_lookup = {
        1: {"MDT vs Guideline": 5, "GPT vs Guideline": 2, "MDT vs GPT": 2},
        2: {"MDT vs Guideline": 5, "GPT vs Guideline": 5, "MDT vs GPT": 5},
        3: {"MDT vs Guideline": 5, "GPT vs Guideline": 5, "MDT vs GPT": 5},
        4: {"MDT vs Guideline": 4, "GPT vs Guideline": 2, "MDT vs GPT": 2},
    }
    for case_id, scores in scores_lookup.items():
        for pair in pairs:
            for rater in ["R1", "R2", "R3", "R4", "R5", "R6"]:
                rows.append({
                    "case_id": case_id,
                    "pair": pair,
                    "rater_id": rater,
                    "score": scores[pair] + (0 if rater in {"R1", "R2"} else -1),
                })
    df_conc = pd.DataFrame(rows)
    return df_rec, df_conc


def _authenticate(email: str, password: str) -> bool:
    return st.session_state.users.get(email) == password


def _signup(email: str, password: str):
    if email in st.session_state.users:
        st.warning("이미 가입된 이메일입니다.")
    else:
        st.session_state.users[email] = password
        st.success("회원가입 완료! 이제 로그인하세요.")


def _render_auth_sidebar():
    st.sidebar.header("계정")
    if st.sidebar.button("데모 계정으로 바로 체험", help="로그인 없이 샘플 데이터까지 자동 불러오기"):
        st.session_state.logged_in = True
        st.session_state.username = "demo@example.com"
        st.session_state.df_rec, st.session_state.df_conc = _create_sample_data()
        st.session_state.success_message = "데모 계정으로 로그인하고 샘플 데이터를 불러왔습니다."

    tab_login, tab_signup = st.sidebar.tabs(["로그인", "회원가입"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("이메일", key="login_email")
            password = st.text_input("비밀번호", type="password", key="login_password")
            submitted = st.form_submit_button("로그인")
        if submitted:
            if _authenticate(email, password):
                st.session_state.logged_in = True
                st.session_state.username = email
                st.session_state.success_message = "로그인 성공!"
            else:
                st.session_state.error_message = "이메일 또는 비밀번호가 올바르지 않습니다."

    with tab_signup:
        with st.form("signup_form"):
            new_email = st.text_input("이메일", key="signup_email")
            new_password = st.text_input("비밀번호", type="password", key="signup_password")
            confirm_password = st.text_input("비밀번호 확인", type="password", key="signup_password_confirm")
            if st.form_submit_button("회원가입"):
                if not new_email or not new_password:
                    st.warning("이메일과 비밀번호를 모두 입력하세요.")
                elif new_password != confirm_password:
                    st.warning("비밀번호가 일치하지 않습니다.")
                else:
                    _signup(new_email, new_password)

    if st.session_state.logged_in:
        st.sidebar.success(f"로그인: {st.session_state.username}")
        if st.sidebar.button("로그아웃"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.df_rec = None
            st.session_state.df_conc = None
            st.experimental_rerun()
    else:
        st.sidebar.info("데모 계정: demo@example.com / demo123")


def _render_template_section():
    st.sidebar.header("1. 템플릿 다운로드")
    try:
        with open("templates/mdt_ai_concordance_template.xlsx", "rb") as f:
            st.sidebar.download_button(
                label="엑셀 템플릿 다운로드",
                data=f,
                file_name="mdt_ai_concordance_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except FileNotFoundError:
        st.sidebar.warning("templates/mdt_ai_concordance_template.xlsx 파일을 찾을 수 없습니다.")


def _load_uploaded_data(uploaded_file):
    try:
        df_rec, df_conc = core_concordance.load_data(uploaded_file)
        st.session_state.df_rec = df_rec
        st.session_state.df_conc = df_conc
        st.success("데이터를 성공적으로 불러왔습니다!")
    except Exception as e:
        st.session_state.df_rec = None
        st.session_state.df_conc = None
        st.error(f"업로드 데이터 처리 중 오류가 발생했습니다: {e}")


def _render_data_loader():
    st.sidebar.header("2. 데이터 업로드 또는 데모 실행")
    uploaded_file = st.sidebar.file_uploader("완성된 엑셀 파일 업로드", type=["xlsx"])
    if uploaded_file is not None:
        _load_uploaded_data(uploaded_file)

    if st.sidebar.button("데모 데이터 불러오기"):
        st.session_state.df_rec, st.session_state.df_conc = _create_sample_data()
        st.success("샘플 데이터셋으로 분석을 실행할 준비가 되었습니다.")


def _render_dataset_preview():
    st.subheader("데이터 미리보기")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("recommendations (상위 5행)")
        st.dataframe(st.session_state.df_rec.head())
    with col2:
        st.caption("concordance_ratings (상위 5행)")
        st.dataframe(st.session_state.df_conc.head())


def _render_exact_concordance(df_rec):
    st.header("A. Exact Concordance (문자열 일치)")
    exact_summary = core_concordance.build_global_exact_concordance_summary(df_rec)
    st.dataframe(exact_summary.style.format({"agreement_rate": "{:.2%}", "kappa": "{:.3f}"}))

    st.download_button(
        "Exact Concordance Summary 다운로드 (CSV)",
        exact_summary.to_csv(index=False).encode("utf-8"),
        "exact_concordance_summary.csv",
        "text/csv",
    )

    st.subheader("혼동 행렬")
    pair_choice = st.selectbox("비교 쌍 선택", exact_summary["pair_name"].unique())

    if pair_choice == "MDT vs Guideline":
        res = core_concordance.compute_pairwise_exact_concordance(df_rec, "mdt_rec", "guideline_rec")
    elif pair_choice == "GPT vs Guideline":
        res = core_concordance.compute_pairwise_exact_concordance(df_rec, "gpt_rec", "guideline_rec")
    else:  # MDT vs GPT
        res = core_concordance.compute_pairwise_exact_concordance(df_rec, "mdt_rec", "gpt_rec")

    st.dataframe(res["confusion_matrix"])

    return exact_summary


def _render_ratings(df_conc):
    st.header("B. Concordance Ratings (0-5 척도)")
    ratings_results = core_concordance.summarize_concordance_scores(df_conc)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("기술 통계")
        st.dataframe(
            ratings_results["descriptive"].style.format({"mean_score": "{:.2f}", "sd_score": "{:.2f}"})
        )
        st.download_button(
            "Ratings Summary 다운로드 (CSV)",
            ratings_results["descriptive"].to_csv(index=False).encode("utf-8"),
            "ratings_summary.csv",
            "text/csv",
        )
    with col2:
        st.subheader("신뢰도 (ICC)")
        st.dataframe(ratings_results["icc"].style.format({"ICC_single": "{:.3f}", "ICC_average": "{:.3f}"}))
    return ratings_results


def _render_discordance(df_rec):
    st.header("C. 불일치 사유")
    reasons_df = core_concordance.analyze_discordance_reasons(df_rec)
    if not reasons_df.empty:
        st.dataframe(reasons_df.style.format({"percent": "{:.1f}%"}))
        st.download_button(
            "불일치 사유 다운로드 (CSV)",
            reasons_df.to_csv(index=False).encode("utf-8"),
            "discordance_reasons.csv",
            "text/csv",
        )
    else:
        st.info("불일치 사유가 없습니다.")
    return reasons_df


def _render_figures(exact_summary, ratings_results, reasons_df):
    st.header("D. 시각화")
    tab1, tab2, tab3 = st.tabs(["Exact Concordance", "Ratings (0-5)", "Discordance Reasons"])

    with tab1:
        fig1 = core_plots.plot_exact_concordance_bar(exact_summary)
        st.pyplot(fig1)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png")
        st.download_button("그림 다운로드 (PNG)", buf1.getvalue(), "exact_concordance.png", "image/png")

    with tab2:
        fig2 = core_plots.plot_concordance_score_bar(ratings_results["descriptive"])
        st.pyplot(fig2)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png")
        st.download_button("그림 다운로드 (PNG)", buf2.getvalue(), "ratings_plot.png", "image/png")

    with tab3:
        if not reasons_df.empty:
            fig3 = core_plots.plot_discordance_reasons(reasons_df)
            st.pyplot(fig3)
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format="png")
            st.download_button("그림 다운로드 (PNG)", buf3.getvalue(), "reasons_plot.png", "image/png")
        else:
            st.info("그릴 불일치 사유가 없습니다.")


def main():
    _init_session_state()

    st.title("MDT vs Guideline vs GPT Concordance Analysis")
    st.markdown(
        """
    MDT, Guideline, GPT 권고안을 비교하여 일치율, 신뢰도, 불일치 사유를 한 번에 확인하는 웹 애플리케이션입니다.
    반복 실행해도 상태가 안전하도록 세션 상태를 초기화하고, 로그인/회원가입 및 데모 데이터를 제공합니다.
    """
    )

    _render_auth_sidebar()
    _render_template_section()
    _render_data_loader()

    if "success_message" in st.session_state:
        st.success(st.session_state.pop("success_message"))
    if "error_message" in st.session_state:
        st.error(st.session_state.pop("error_message"))

    if not st.session_state.logged_in:
        st.info("분석을 시작하려면 로그인하세요. 데모 계정: demo@example.com / demo123")
        return

    if st.session_state.df_rec is None or st.session_state.df_conc is None:
        st.warning("엑셀을 업로드하거나 데모 데이터를 불러오면 분석 결과가 표시됩니다.")
        return

    # -----------------------------------------
    # 분석 섹션
    # -----------------------------------------
    _render_dataset_preview()
    exact_summary = _render_exact_concordance(st.session_state.df_rec)
    ratings_results = _render_ratings(st.session_state.df_conc)
    reasons_df = _render_discordance(st.session_state.df_rec)
    _render_figures(exact_summary, ratings_results, reasons_df)


if __name__ == "__main__":
    main()
