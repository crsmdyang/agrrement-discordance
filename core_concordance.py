import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy import stats

def load_data(file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the Excel file and returns two DataFrames: recommendations and concordance_ratings.
    Validates required columns and data types.
    """
    try:
        xls = pd.ExcelFile(file)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    # Check sheets
    required_sheets = {"recommendations", "concordance_ratings"}
    if not required_sheets.issubset(xls.sheet_names):
        raise ValueError(f"Excel file must contain sheets: {required_sheets}")

    # ---------------------------------------------------------
    # 1. Load recommendations
    # ---------------------------------------------------------
    df_rec = pd.read_excel(xls, "recommendations")
    req_cols_rec = {
        "case_id", "guideline_rec", "mdt_rec", "gpt_rec", 
        "discordance_reason_category"
    }
    # patient_group and discordance_reason_detail are optional or have defaults
    
    if not req_cols_rec.issubset(df_rec.columns):
        raise ValueError(f"Sheet 'recommendations' missing columns: {req_cols_rec - set(df_rec.columns)}")
    
    # Handle optional columns
    if "patient_group" not in df_rec.columns:
        df_rec["patient_group"] = "All"
    else:
        df_rec["patient_group"] = df_rec["patient_group"].fillna("All")
        
    if "discordance_reason_detail" not in df_rec.columns:
        df_rec["discordance_reason_detail"] = ""

    # Validate discordance categories
    valid_categories = {
        "Patient factor", "Disease/tumor factor", "MDT/clinician factor",
        "System/institution factor", "Guideline-related factor"
    }
    # Allow blank/nan if fully concordant, but if present must be in list
    # Actually, user said "MUST be one of the 5 categories below". 
    # Usually empty is allowed if concordant. Let's check if any non-empty invalid exists.
    
    # Normalize to string, strip
    df_rec["discordance_reason_category"] = df_rec["discordance_reason_category"].astype(str).str.strip()
    
    # Filter out 'nan' or empty strings
    present_cats = df_rec[df_rec["discordance_reason_category"].replace({'nan': '', 'None': ''}) != ""]["discordance_reason_category"]
    invalid_cats = set(present_cats) - valid_categories
    if invalid_cats:
        raise ValueError(f"Invalid discordance categories found: {invalid_cats}. Allowed: {valid_categories}")

    # ---------------------------------------------------------
    # 2. Load concordance_ratings
    # ---------------------------------------------------------
    df_conc = pd.read_excel(xls, "concordance_ratings")
    req_cols_conc = {"case_id", "pair", "rater_id", "score"}
    if not req_cols_conc.issubset(df_conc.columns):
        raise ValueError(f"Sheet 'concordance_ratings' missing columns: {req_cols_conc - set(df_conc.columns)}")

    # Validate rater_id
    valid_raters = {"R1", "R2", "R3", "R4", "R5", "R6"}
    if not set(df_conc["rater_id"].unique()).issubset(valid_raters):
        raise ValueError(f"Invalid rater_ids found. Must be R1-R6.")

    # Validate score (0-5)
    if not pd.api.types.is_numeric_dtype(df_conc["score"]):
        try:
            df_conc["score"] = pd.to_numeric(df_conc["score"])
        except ValueError:
            raise ValueError("Column 'score' in 'concordance_ratings' must be numeric.")
            
    if not df_conc["score"].between(0, 5).all():
        raise ValueError("Scores in 'concordance_ratings' must be between 0 and 5.")

    # Validate case_ids match
    rec_case_ids = set(df_rec["case_id"].astype(str))
    conc_case_ids = set(df_conc["case_id"].astype(str))
    
    if not conc_case_ids.issubset(rec_case_ids):
        missing = conc_case_ids - rec_case_ids
        raise ValueError(f"case_ids in 'concordance_ratings' not found in 'recommendations': {list(missing)[:5]}...")

    return df_rec, df_conc

def compute_pairwise_exact_concordance(df_rec: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Computes pairwise exact concordance (string match).
    """
    valid_df = df_rec[[col1, col2]].dropna()
    n_cases = len(valid_df)
    
    if n_cases == 0:
        return {
            "n_cases": 0, "n_agree": 0, "agreement_rate": 0.0,
            "kappa": np.nan, "confusion_matrix": pd.DataFrame()
        }

    y1 = valid_df[col1].astype(str).str.strip()
    y2 = valid_df[col2].astype(str).str.strip()

    n_agree = (y1 == y2).sum()
    agreement_rate = n_agree / n_cases
    
    labels = sorted(list(set(y1.unique()) | set(y2.unique())))
    if len(labels) > 1:
        kappa = cohen_kappa_score(y1, y2, labels=labels)
    else:
        kappa = np.nan # Kappa not defined for 1 class

    cm = pd.crosstab(y1, y2, rownames=[col1], colnames=[col2])

    return {
        "n_cases": n_cases,
        "n_agree": n_agree,
        "agreement_rate": agreement_rate,
        "kappa": kappa,
        "confusion_matrix": cm
    }

def build_global_exact_concordance_summary(df_rec: pd.DataFrame) -> pd.DataFrame:
    """
    Builds summary table for MDT vs Guideline, GPT vs Guideline, MDT vs GPT.
    """
    pairs = [
        ("MDT vs Guideline", "mdt_rec", "guideline_rec"),
        ("GPT vs Guideline", "gpt_rec", "guideline_rec"),
        ("MDT vs GPT", "mdt_rec", "gpt_rec")
    ]
    
    results = []
    for name, c1, c2 in pairs:
        res = compute_pairwise_exact_concordance(df_rec, c1, c2)
        results.append({
            "pair_name": name,
            "n_cases": res["n_cases"],
            "n_agree": res["n_agree"],
            "agreement_rate": res["agreement_rate"],
            "kappa": res["kappa"]
        })
        
    return pd.DataFrame(results)

def summarize_concordance_scores(df_conc: pd.DataFrame) -> dict:
    """
    Analyzes 0-5 concordance scores by 6 raters.
    Computes Mean, SD, and ICC (Intraclass Correlation Coefficient).
    """
    # Descriptive stats
    summary = df_conc.groupby("pair")["score"].agg(["count", "mean", "std"]).reset_index()
    summary.columns = ["pair", "n_ratings", "mean_score", "sd_score"]
    
    # Calculate ICC for each pair
    # ICC(3,k) or ICC(2,k)? Usually for fixed raters we might use ICC(3,k) or (2,k).
    # Here we have 6 specific raters (R1-R6).
    # We will use a simplified approach or a library if available. 
    # Since 'pingouin' is not in requirements, we can implement a basic ANOVA-based ICC or just skip complex ICC if not strictly required.
    # However, user asked for "ICC / reliability results".
    # We can try to use `statsmodels` or just calculate Cronbach's alpha or simple ICC formula.
    # Let's use a simple ICC(3,1) or ICC(3,k) formula based on Mean Squares from ANOVA.
    
    icc_results = []
    
    for pair in df_conc["pair"].unique():
        sub_df = df_conc[df_conc["pair"] == pair]
        
        # Pivot to (cases x raters)
        # Check for duplicates
        if sub_df.duplicated(subset=["case_id", "rater_id"]).any():
            # Handle duplicates if any (shouldn't be per spec)
            sub_df = sub_df.groupby(["case_id", "rater_id"])["score"].mean().reset_index()
            
        pivot_df = sub_df.pivot(index="case_id", columns="rater_id", values="score")
        
        # Filter to cases with all raters present for ICC calculation
        pivot_df = pivot_df.dropna()
        
        if pivot_df.shape[0] > 1 and pivot_df.shape[1] > 1:
            # Two-way mixed effects model (consistency) - ICC(3,k) often used for fixed raters
            # Using pingouin logic simplified:
            # MS_R = mean square for rows (cases)
            # MS_E = mean square for error
            # MS_C = mean square for columns (raters)
            
            k = pivot_df.shape[1] # number of raters
            n = pivot_df.shape[0] # number of cases
            
            grand_mean = pivot_df.values.mean()
            
            # Sum of Squares
            SST = ((pivot_df.values - grand_mean) ** 2).sum()
            SSR = k * ((pivot_df.mean(axis=1) - grand_mean) ** 2).sum() # Rows (subjects)
            SSC = n * ((pivot_df.mean(axis=0) - grand_mean) ** 2).sum() # Columns (raters)
            SSE = SST - SSR - SSC
            
            MSR = SSR / (n - 1)
            MSC = SSC / (k - 1)
            MSE = SSE / ((n - 1) * (k - 1))
            
            # ICC(3,1) = (MSR - MSE) / (MSR + (k-1)*MSE)
            # ICC(3,k) = (MSR - MSE) / MSR
            
            icc_val = (MSR - MSE) / (MSR + (k - 1) * MSE) # Single rater reliability
            icc_avg = (MSR - MSE) / MSR # Average k raters reliability
            
            icc_results.append({
                "pair": pair,
                "n_cases_icc": n,
                "ICC_single": icc_val,
                "ICC_average": icc_avg
            })
        else:
            icc_results.append({
                "pair": pair,
                "n_cases_icc": 0,
                "ICC_single": np.nan,
                "ICC_average": np.nan
            })
            
    return {
        "descriptive": summary,
        "icc": pd.DataFrame(icc_results)
    }

def analyze_discordance_reasons(df_rec: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes discordance reasons.
    Discordant if NOT (guideline == mdt == gpt).
    """
    # Fill NA to ensure comparison works
    temp = df_rec.fillna("")
    
    # Check exact match across all 3
    is_discordant = ~((temp["guideline_rec"] == temp["mdt_rec"]) & 
                      (temp["mdt_rec"] == temp["gpt_rec"]))
    
    discordant_df = df_rec[is_discordant]
    
    if len(discordant_df) == 0:
        return pd.DataFrame(columns=["discordance_reason_category", "n", "percent"])
        
    counts = discordant_df["discordance_reason_category"].value_counts().reset_index()
    counts.columns = ["discordance_reason_category", "n"]
    
    # Filter out empty reasons if they exist (should ideally be filled for discordant cases)
    counts = counts[counts["discordance_reason_category"] != ""]
    
    total_discordant = counts["n"].sum()
    if total_discordant > 0:
        counts["percent"] = (counts["n"] / total_discordant) * 100
    else:
        counts["percent"] = 0.0
        
    return counts
