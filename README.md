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

### Streamlit 웹앱

1. 앱 실행
   ```bash
   streamlit run app.py
   ```
2. 기본 제공 계정으로 로그인하거나 사이드바에서 회원가입 후 로그인합니다. (데모 계정: `demo@example.com / demo123`)
3. 사이드바에서 **엑셀 템플릿 다운로드** 후 데이터를 채워 업로드하거나, "데모 데이터 불러오기" 버튼으로 즉시 테스트합니다.
4. 업로드/데모 데이터가 준비되면 메인 화면에서 일치율, 신뢰도, 불일치 사유, 시각화를 바로 확인하고 CSV/PNG로 다운로드할 수 있습니다.

### CLI (automatic batch analysis)

If you have a completed Excel file that follows the template structure, you can run the
full statistical analysis without Streamlit:

```bash
python cli_analyze.py path/to/filled_dataset.xlsx --output-dir results/
```

The script will print the summaries to the console and save CSV outputs (exact
concordance, confusion matrices, ratings summaries, ICC values, and discordance
reasons) under the chosen output directory.

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
