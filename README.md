# MDT/AI Concordance App

A Streamlit application for analyzing concordance between MDT, Guideline, and GPT treatment recommendations.

## Features

- **Exact Concordance**: Calculates agreement rates and Kappa based on exact string matching.
- **Concordance Ratings**: Analyzes 0-5 expert ratings from 6 raters (Mean, SD, ICC).
- **Discordance Analysis**: Summarizes discordance reasons using 5 predefined categories.
- **Visualization**: Generates bar charts for all metrics.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the app:
   ```bash
   streamlit run app.py
   ```
2. Download the **Excel Template**.
3. Fill in your data.
4. Upload and view results.

## Excel Structure

### 1. `recommendations`
- `case_id`: Case identifier.
- `patient_group`: Optional grouping.
- `guideline_rec`, `mdt_rec`, `gpt_rec`: Recommendation strings.
- `discordance_reason_category`: Must be one of:
  - Patient factor
  - Disease/tumor factor
  - MDT/clinician factor
  - System/institution factor
  - Guideline-related factor

### 2. `concordance_ratings`
- `case_id`: Matches recommendations.
- `pair`: "MDT_vs_Guideline", "GPT_vs_Guideline", "MDT_vs_GPT".
- `rater_id`: "R1" to "R6".
- `score`: Integer 0-5.
