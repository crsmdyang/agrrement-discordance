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
2. 왼쪽 사이드바 상단의 **"데모 계정으로 바로 체험"** 버튼을 누르면 자동으로 로그인+샘플 데이터 로딩까지 완료됩니다. (직접 로그인하려면 데모 계정: `demo@example.com / demo123`)
3. 직접 데이터를 쓰려면 사이드바에서 **엑셀 템플릿 다운로드** 후 데이터를 채워 업로드하거나, "데모 데이터 불러오기" 버튼으로 즉시 테스트합니다.
4. 업로드/데모 데이터가 준비되면 메인 화면에서 일치율, 신뢰도, 불일치 사유, 시각화를 바로 확인하고 CSV/PNG로 다운로드할 수 있습니다.

### GitHub에서 최신 버전으로 업데이트하기

#### 1) 자동 패치 스크립트 사용 (권장)

작업 디렉터리가 깨끗한 상태(커밋/스태시 완료)이고, `origin` 원격이 설정되어 있을 때 아래 한 줄이면 최신 커밋을 가져오고 의존성을 재설치합니다.

```bash
python auto_update.py
```

> `origin` 원격이 없으면 `git remote add origin <repo-url>`로 추가한 뒤 다시 실행하세요.

옵션: 의존성 재설치를 건너뛰려면 `--skip-install`을 덧붙입니다.

```bash
python auto_update.py --skip-install
```

#### 2) 수동 업데이트

자동 스크립트 대신 직접 명령을 실행하려면 아래 순서를 따르세요.

1. 저장소 루트로 이동합니다.
   ```bash
   cd /path/to/agrrement-discordance
   ```
2. 원격 저장소에서 최신 커밋을 가져옵니다.
   ```bash
   git pull origin main
   ```
3. 의존성이 바뀌었을 수 있으니 필요하면 다시 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
4. 새 버전으로 Streamlit 앱을 실행합니다.
   ```bash
   streamlit run app.py
   ```

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
