import pandas as pd
import numpy as np

def create_template():
    # 1. Recommendations Sheet
    # 5-10 dummy cases
    rec_data = {
        "case_id": [f"Case{i:02d}" for i in range(1, 11)],
        "patient_group": ["Stage IV", "Stage III", "Stage IV", "Stage II", "Stage III"] * 2,
        "guideline_rec": ["FOLFOX", "Surgery", "FOLFOX+Bev", "Observation", "Surgery"] * 2,
        "mdt_rec": ["FOLFOX", "Surgery", "FOLFOX", "Observation", "Surgery", 
                    "FOLFOX", "Surgery", "FOLFOX+Bev", "Surgery", "Surgery"], 
        "gpt_rec": ["FOLFOX", "Surgery", "FOLFOX+Bev", "Observation", "Chemo", 
                    "FOLFOX", "Surgery", "FOLFOX", "Observation", "Surgery"],
        "discordance_reason_category": [
            "", "", "Patient factor", "", "Patient factor", 
            "", "", "System/institution factor", "Disease/tumor factor", ""
        ],
        "discordance_reason_detail": [
            "", "", "Comorbidity", "", "Refused surgery", 
            "", "", "Cost issue", "High risk", ""
        ]
    }
    df_rec = pd.DataFrame(rec_data)

    # 2. Concordance Ratings Sheet
    # 6 raters (R1-R6) for each case and each pair
    pairs = ["MDT_vs_Guideline", "GPT_vs_Guideline", "MDT_vs_GPT"]
    raters = ["R1", "R2", "R3", "R4", "R5", "R6"]
    
    conc_rows = []
    for case in rec_data["case_id"]:
        for pair in pairs:
            for rater in raters:
                # Random score 0-5
                score = np.random.randint(0, 6)
                conc_rows.append({
                    "case_id": case,
                    "pair": pair,
                    "rater_id": rater,
                    "score": score,
                    "scale_type": "ordinal"
                })
                
    df_conc = pd.DataFrame(conc_rows)

    # Write to Excel
    output_path = "templates/mdt_ai_concordance_template.xlsx"
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_rec.to_excel(writer, sheet_name="recommendations", index=False)
        df_conc.to_excel(writer, sheet_name="concordance_ratings", index=False)
        
    print(f"Template created at {output_path}")

if __name__ == "__main__":
    create_template()
