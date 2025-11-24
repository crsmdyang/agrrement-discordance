import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

# Import core modules
import core_concordance
import core_plots

st.set_page_config(page_title="MDT/AI Concordance App", layout="wide")

def main():
    st.title("MDT vs Guideline vs GPT Concordance Analysis")
    st.markdown("""
    This app analyzes concordance between MDT, Guideline, and GPT recommendations.
    It supports exact string matching and expert concordance ratings (0-5 scale).
    """)

    # ---------------------------------------------------------
    # 1. Template Download
    # ---------------------------------------------------------
    st.sidebar.header("1. Download Template")
    try:
        with open("templates/mdt_ai_concordance_template.xlsx", "rb") as f:
            st.sidebar.download_button(
                label="Download Excel Template",
                data=f,
                file_name="mdt_ai_concordance_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except FileNotFoundError:
        st.sidebar.warning("Template file not found. Please generate it first.")

    # ---------------------------------------------------------
    # 2. Upload Data
    # ---------------------------------------------------------
    st.sidebar.header("2. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload filled Excel file", type=["xlsx"])

    if uploaded_file:
        try:
            df_rec, df_conc = core_concordance.load_data(uploaded_file)
            
            st.success("Data loaded successfully!")
            
            with st.expander("Preview 'recommendations'"):
                st.dataframe(df_rec.head())
                
            with st.expander("Preview 'concordance_ratings'"):
                st.dataframe(df_conc.head())

            # ---------------------------------------------------------
            # A. Exact Concordance
            # ---------------------------------------------------------
            st.header("A. Exact Concordance (String Match)")
            
            exact_summary = core_concordance.build_global_exact_concordance_summary(df_rec)
            st.dataframe(exact_summary.style.format({"agreement_rate": "{:.2%}", "kappa": "{:.3f}"}))
            
            st.download_button(
                "Download Exact Concordance Summary (CSV)",
                exact_summary.to_csv(index=False).encode('utf-8'),
                "exact_concordance_summary.csv",
                "text/csv"
            )
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            pair_choice = st.selectbox("Select Pair", exact_summary["pair_name"].unique())
            
            if pair_choice == "MDT vs Guideline":
                res = core_concordance.compute_pairwise_exact_concordance(df_rec, "mdt_rec", "guideline_rec")
            elif pair_choice == "GPT vs Guideline":
                res = core_concordance.compute_pairwise_exact_concordance(df_rec, "gpt_rec", "guideline_rec")
            else: # MDT vs GPT
                res = core_concordance.compute_pairwise_exact_concordance(df_rec, "mdt_rec", "gpt_rec")
                
            st.dataframe(res["confusion_matrix"])

            # ---------------------------------------------------------
            # B. Concordance Ratings (0-5)
            # ---------------------------------------------------------
            st.header("B. Concordance Ratings (0-5 Scale)")
            
            ratings_results = core_concordance.summarize_concordance_scores(df_conc)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Descriptive Statistics")
                st.dataframe(ratings_results["descriptive"].style.format({"mean_score": "{:.2f}", "sd_score": "{:.2f}"}))
                
                st.download_button(
                    "Download Ratings Summary (CSV)",
                    ratings_results["descriptive"].to_csv(index=False).encode('utf-8'),
                    "ratings_summary.csv",
                    "text/csv"
                )
                
            with col2:
                st.subheader("Reliability (ICC)")
                st.dataframe(ratings_results["icc"].style.format({"ICC_single": "{:.3f}", "ICC_average": "{:.3f}"}))

            # ---------------------------------------------------------
            # C. Discordance Reasons
            # ---------------------------------------------------------
            st.header("C. Discordance Reasons")
            
            reasons_df = core_concordance.analyze_discordance_reasons(df_rec)
            
            if not reasons_df.empty:
                st.dataframe(reasons_df.style.format({"percent": "{:.1f}%"}))
                
                st.download_button(
                    "Download Reasons (CSV)",
                    reasons_df.to_csv(index=False).encode('utf-8'),
                    "discordance_reasons.csv",
                    "text/csv"
                )
            else:
                st.info("No discordance reasons found.")

            # ---------------------------------------------------------
            # D. Figures
            # ---------------------------------------------------------
            st.header("D. Figures")
            
            tab1, tab2, tab3 = st.tabs(["Exact Concordance", "Ratings (0-5)", "Discordance Reasons"])
            
            with tab1:
                fig1 = core_plots.plot_exact_concordance_bar(exact_summary)
                st.pyplot(fig1)
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format="png")
                st.download_button("Download Figure (PNG)", buf1.getvalue(), "exact_concordance.png", "image/png")
                
            with tab2:
                fig2 = core_plots.plot_concordance_score_bar(ratings_results["descriptive"])
                st.pyplot(fig2)
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="png")
                st.download_button("Download Figure (PNG)", buf2.getvalue(), "ratings_plot.png", "image/png")
                
            with tab3:
                if not reasons_df.empty:
                    fig3 = core_plots.plot_discordance_reasons(reasons_df)
                    st.pyplot(fig3)
                    buf3 = io.BytesIO()
                    fig3.savefig(buf3, format="png")
                    st.download_button("Download Figure (PNG)", buf3.getvalue(), "reasons_plot.png", "image/png")
                else:
                    st.info("No reasons to plot.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            # st.exception(e) # Uncomment for debugging

if __name__ == "__main__":
    main()
