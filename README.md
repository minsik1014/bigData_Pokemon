# Pokédex Insights – 데이터 처리 및 시각화 개요

이 문서는 `pokemom.py` Streamlit 앱을 구성한 핵심 로직을 간단한 README 형식으로 정리한 것입니다.

## 배포 링크

- Streamlit Cloud: https://bigdatapokemon.streamlit.app/

## 데이터 출처

- 포켓몬 CSV: https://zenodo.org/records/4661775  
- PokéAPI: https://pokeapi.co/

## 1. Pandas를 이용한 데이터 준비

- **CSV 로드 (`pd.read_csv`)**  
  `pokemon.csv` 파일을 `load_data()` 함수에서 읽어 들이고, `rename(columns=lambda c: c.strip())`, `drop(columns=["Unnamed: 0"])`로 불필요한 열을 정리했습니다.

- **세대 분류 (`assign_generation`)**  
  도감 번호(`DexID`)를 기준으로 1~9세대 경계를 직접 정의한 뒤, `apply(assign_generation)`으로 각 포켓몬의 세대 정보를 생성했습니다.

- **타입 분리 (`infer_types`)**  
  원본 CSV의 `Type` 열에는 “Grass Poison”처럼 공백으로 두 타입이 기록되어 있어, 문자열을 나눈 뒤 `(Type 1, Type 2)` 형태로 분리 저장했습니다.

- **숫자 정보 추출 (`parse_numeric_front`)**  
  `catch_rate`, `base_friendship`, `Height`, `Weight` 등 문자열 안에 있는 숫자 값을 정규식으로 파싱해 각각 `CatchRate`, `BaseFriendship`, `Height_m`, `Weight_kg` 숫자 컬럼으로 변환했습니다.

- **필터링/그룹핑**  
  - 포켓몬 탐색기: `multiselect`/`slider` 값으로 `Generation`, `Type 1/2`, 스탯 범위 등을 필터링 후 `df[...]` 조건으로 subset을 생성했습니다.  
  - 타입 분석: `groupby("Type 1")[STAT_COLS].mean()` 으로 타입별 평균 스탯 테이블을 만들었습니다.  
  - 스탯 분포: 단일 컬럼을 선택해 `df[stat]` 형태로 분포를 분석했습니다.  
  - 팀 빌더: 선택된 도감번호 리스트로 `df.loc[team_choices]`를 뽑아 합계/평균(`agg(["sum","mean"])`)을 계산했습니다.

## 2. Seaborn/Matplotlib을 이용한 시각화

- **선/막대 그래프 (`sns.lineplot`, `sns.barplot`)**  
  대시보드에서는 `sns.lineplot`으로 세대별 평균 Total 추이를, `sns.barplot`으로 타입별 포켓몬 수를 시각화했습니다.

- **타입 분석**  
  선택한 타입만 추려 `melt` 후 `sns.barplot`으로 스탯 비교를 보여주고, 상위 타입 TOP3 표를 함께 제공합니다.

- **스탯 분포**  
  `sns.histplot(..., kde=True)`로 선택한 스탯의 분포를 표현하고, Matplotlib 축 타이틀을 최소화해 스탯 이름만 강조했습니다.

- **키·몸무게 산점도**  
  선택된 타입만 필터링한 뒤, `sns.scatterplot` + `hue="DisplayType"`으로 타입별 높이/무게 분포를 비교합니다.

- **레이더 차트 (`matplotlib` polar plot)**  
  선택 포켓몬 또는 팀 평균 스탯을 `radar_chart()` 함수에서 폴라 좌표로 그리고 `st.pyplot`으로 표시합니다.

- **팀 빌더 바 차트 (`st.bar_chart`)**  
  선택 팀의 타입 분포를 `value_counts()` 후 정수형으로 변환하여 Streamlit의 기본 바 차트로 표현했습니다.

## 3. Streamlit UI 구성

- **페이지 구조**  
  `st.sidebar.radio`로 “홈 / 포켓몬 탐색기 / 타입 분석 / 스탯 분포 / 키·몸무게 / 팀 빌더” 페이지를 선택하고, 각 함수(`render_*`)가 독립적으로 UI를 렌더링합니다.

- **이미지 로딩 (`st.image`)**  
  - 오박사/로고 등은 로컬 파일(`123.png`, `pokemon.jpg`)을 사용.  
  - 포켓몬 스프라이트는 `fetch_sprite()`에서 PokéAPI를 호출해 가져오고 실패 시 백업 URL(`get_sprite_url`)을 사용합니다.

- **인터랙티브 필터**  
  `st.multiselect`, `st.slider`, `st.text_input`, `st.selectbox`를 조합해 포켓몬 목록을 즉시 필터링하고, 선택한 포켓몬의 상세 표/이미지/레이더 차트가 업데이트되도록 구성했습니다.

이와 같이 Pandas의 전처리(GroupBy, Apply, 필터링)와 Seaborn·Matplotlib의 다양한 그래프, Streamlit의 인터랙티브 위젯을 조합해 포켓몬 데이터를 분석/시각화하는 “Pokédex Insights” 앱을 작성했습니다.

## 4. 표·그래프에 사용된 세부 명령어

- **데이터프레임/표 렌더링**
  - `st.dataframe(filtered[display_cols], use_container_width=True, height=320)`  
    탐색기에서 필터링된 결과표를 가로폭에 맞춰 표시합니다.
  - `st.table(summary)` / `st.table(summary_df)`  
    팀 요약 스탯, 스탯 분포 요약 등 작은 표를 고정 폭으로 그립니다.
  - `st.table(top.rename(...))`, `st.table(total_top.rename(...))`  
    타입별 TOP3 랭킹을 간단한 표로 보여줍니다.

- **요약 통계 계산**
  - `df.groupby("Generation")["Total"].mean()`  
    세대별 평균 Total KPI/라인 차트용 집계를 만듭니다.
  - `df.groupby("Type 1")[STAT_COLS].mean()`  
    타입별 평균 스탯 테이블과 바 차트에 사용하는 집계입니다.
  - `team_df[["Total"] + STAT_COLS].agg(["sum", "mean"]).T`  
    선택한 팀의 합계/평균 스탯을 전치해 표 형태로 변환합니다.
  - `pd.concat([team_df["Type 1"], team_df["Type 2"]]).value_counts()`  
    팀 빌더 바 차트용 타입 분포를 계산합니다.

- **그래프 관련 명령**
  - `sns.lineplot(data=gen_avg, x="Generation", y="Total", marker="o")`  
    세대별 평균 Total 추이를 선 그래프로 표현합니다.
  - `sns.barplot(data=type_counts, y="Type 1", x="Count")`  
    타입별 포켓몬 수를 가로 막대 그래프로 시각화합니다.
  - `sns.barplot(data=melted, x="Stat", y="Value", hue="Type 1")`  
    타입 비교 바 차트에서 melt한 데이터셋을 사용합니다.
  - `sns.histplot(df[stat_choice], kde=True)`  
    선택 스탯 분포를 KDE가 포함된 히스토그램으로 그립니다.
  - `sns.scatterplot(..., hue="DisplayType")`  
    키·몸무게 페이지에서 선택 타입별 색상을 지정합니다.
  - `st.bar_chart(bar_data.astype(int))`  
    Streamlit 기본 바 차트로 팀 타입 분포를 직관적으로 표시합니다.

- **기타 표시 요소**
  - `st.metric("포켓몬 수", f"{total_pkm:,}")` 등 KPI 카드 생성
  - `st.sidebar.expander`, `st.columns`, `st.image(fetch_sprite(...))`  
    필터 패널, 컬럼 레이아웃, 포켓몬 스프라이트 이미지를 구성하는 핵심 명령입니다.
  - `avg_value = df[stat_choice].mean()` / `max_row = df.loc[df[stat_choice].idxmax()]`  
    스탯 분포 요약 표에서 평균값과 최고 스탯 포켓몬을 구하는 Pandas Series 메서드 조합입니다.

## 5. 코드/문법 상세 설명

- **모듈 임포트와 타입 힌트**
  - `from __future__ import annotations` : Python 3.11 이하에서도 PEP 563 방식의 지연 평가 타입 힌트를 사용하기 위한 선언입니다.
  - `from functools import lru_cache`, `from pathlib import Path` 등 표준 라이브러리 임포트는 상단에 정리되어 있어 각 기능(파일 경로 처리, 캐싱)이 어떤 모듈에서 오는지 명시합니다.
  - 함수 정의 시 `def radar_chart(stats: dict[str, float], title: str = "Base Stat Radar")`처럼 타입 힌트와 기본값을 함께 제공해 파라미터 의미를 분명히 합니다.

- **데이터 로딩/정제 문법**
  - `pd.read_csv(csv_path, encoding="utf-8-sig")` : CSV 파일을 UTF-8-SIG로 읽어 들여 한글/윈도우 BOM 이슈를 방지합니다.
  - `df.rename(columns=lambda c: c.strip())` : 람다를 활용해 컬럼 이름 전체에 공백 제거 함수를 일괄 적용합니다.
  - `df.drop(columns=["Unnamed: 0"])` : 불필요한 열을 제거할 때 `columns` 파라미터로 리스트를 넘겨 가독성을 높입니다.
  - `df.insert(0, "DexID", np.arange(1, len(df) + 1))` : 첫 번째 위치에 연속 ID 컬럼을 삽입하는 Pandas Insert 문법입니다.
  - `df["Type 1"], df["Type 2"] = zip(*df["Type"].apply(infer_types))` : `apply` 결과(튜플)를 `zip(*)`으로 언패킹해 여러 새 컬럼을 동시에 생성합니다.
  - `df["Generation"] = df["DexID"].apply(assign_generation)` : 사용자 정의 함수와 `apply`를 결합해 세대 정보를 계산합니다.
  - `df["CatchRate"] = df["catch_rate"].apply(parse_numeric_front)` 등 문자열 컬럼을 숫자로 변환할 때 재사용 가능한 파서를 적용합니다.

- **데코레이터 활용**
  - `@lru_cache(maxsize=None)` : 동일 파라미터로 호출되는 `get_sprite_url` 결과를 메모리에 저장해 네트워크 호출을 줄입니다.
  - `@lru_cache(maxsize=512)` : `fetch_sprite`는 512개의 다른 이름까지 캐시하여 API 응답을 재활용합니다.
  - `@st.cache_data(show_spinner=False)` : Streamlit에서 데이터프레임을 캐시해 앱이 재실행되더라도 CSV 로드를 반복하지 않도록 합니다.

- **Pandas 문법**
  - `df.rename(columns=lambda c: c.strip())` : 람다 함수를 사용해 모든 컬럼명 좌우 공백을 제거합니다.
  - `df.insert(0, "DexID", np.arange(1, len(df) + 1))` : 0번째 위치에 새로운 열을 삽입해 연속된 도감 번호를 생성합니다.
  - `df["Type 1"], df["Type 2"] = zip(*df["Type"].apply(infer_types))` : `apply`로 반환된 튜플을 언패킹하여 두 개의 컬럼에 동시에 할당합니다.
  - `filtered = filtered[(filtered[col] >= min_v) & (filtered[col] <= max_v)]` : 불리언 마스크를 체인으로 적용할 때 괄호로 우선순위를 명시하고, `&` 연산자를 사용해 조건을 결합합니다.
  - `type_avg[type_avg["Type 1"].isin(selected_types)]` : `isin`을 통해 멀티셀렉트 결과 리스트에 포함된 행만 추출합니다.
  - `subset.melt(id_vars="Type 1", var_name="Stat", value_name="Value")` : wide 포맷을 long 포맷으로 변환해 여러 스탯을 동시에 시각화할 수 있게 합니다.

- **NumPy/정규식 처리**
  - `np.arange(1, len(df) + 1)` : 1부터 `len(df)`까지의 정수 배열을 생성합니다.
  - `np.linspace(0, 2 * np.pi, len(labels), endpoint=False)` : 레이더 차트 각도를 균등 분포로 생성합니다.
  - `re.search(r"(\d+(\.\d+)?)", str(value))` : 정규식을 사용해 문자열 선두의 숫자(정수 또는 소수)를 추출합니다.

- **Streamlit 인터랙션**
  - `st.sidebar.radio("페이지 선택", pages)` : 사이드바 라디오 버튼에서 문자열 목록 중 하나를 선택하면 해당 페이지 렌더 함수를 호출합니다.
  - `st.multiselect("세대", generations, default=generations)` : 기본적으로 모든 세대가 선택된 상태에서 사용자가 특정 세대만 필터링할 수 있습니다.
  - `st.slider(f"{col} 범위", min_val, max_val, (min_val, max_val))` : 튜플 기본값으로 최소/최대 범위를 지정해 슬라이더가 범위 선택 모드로 작동합니다.
  - `st.selectbox(..., format_func=lambda dex: option_label(candidate_df.loc[dex]))` : 선택 옵션에 사용자 친화적 라벨을 표시하기 위해 `format_func`를 사용합니다.

- **데이터 필터링/선택 문법**
  - `df[df["Generation"].isin(gen_filter) & df["Type 1"].isin(type1_filter) & df["Type 2"].isin(type2_filter)]` : `isin`과 논리 연산자를 조합해 다중 컬럼 필터를 작성합니다.
  - `df["Name"].str.contains(name_query, case=False, na=False)` : 문자열 열에서 대소문자 무시 부분 일치를 수행하며 결측치를 False로 처리합니다.
  - `candidate_df = candidate_df[candidate_df["Name"].str.contains(detail_query, case=False, na=False) | candidate_df["DexID"].astype(str).str.contains(detail_query)]` : 문자열/숫자 조건을 `|`로 묶어 검색 UI를 구현합니다.
  - `candidate_df["DexID"].astype(str).str.contains(detail_query)` : 숫자형 도감번호를 문자열로 변환해 텍스트 검색을 가능하게 합니다.
  - `candidate_df = candidate_df.set_index("DexID")` 후 `candidate_df.loc[selected_dex]` : 인덱스를 도감번호로 바꾸고 `.loc`로 특정 행을 즉시 조회합니다.
  - `options_df.loc[team_choices].reset_index().rename(columns={"index": "DexID"})` : 리스트 선택 결과를 다시 데이터프레임으로 모아 인덱스를 컬럼으로 복원합니다.
  - `type_filtered = df[df["Type 1"].isin(selected_types) | df["Type 2"].isin(selected_types)]` : `|` 연산자로 두 컬럼 중 하나라도 조건을 만족하면 통과시키는 필터를 만듭니다.
  - `type_counts = df["Type 1"].value_counts().reset_index(); type_counts.columns = ["Type 1", "Count"]` : `value_counts` 결과를 데이터프레임 형태로 재정렬하고 컬럼명을 명시적으로 지정합니다.
  - `df["Type 1"].mode().iat[0]` : 가장 빈도가 높은 타입을 찾을 때 `mode()`로 복수 결과가 나올 수 있으므로 `.iat[0]`으로 첫 값을 선택합니다.
  - `type_counts.drop(labels=["None"], errors="ignore")` : 특정 레이블만 제거하고 없으면 에러 없이 넘어가도록 `errors="ignore"`를 사용합니다.

- **집계/요약 문법**
  - `df.groupby("Generation")["Total"].mean()` / `df.groupby("Type 1")[STAT_COLS].mean()` : 그룹 기준 컬럼과 집계 타깃 컬럼을 명시해 세대·타입별 평균을 계산합니다.
  - `df.groupby("Generation")["Total"].mean().reset_index()` : 그룹 결과를 다시 일반 데이터프레임으로 펼쳐 차트에 사용합니다.
  - `team_df[["Total"] + STAT_COLS].agg(["sum", "mean"]).T` : 여러 통계 함수를 동시에 적용한 뒤 `.T`로 전치해 표 형태를 바꿉니다.
  - `pd.concat([team_df["Type 1"], team_df["Type 2"]]).value_counts()` : 두 시리즈를 위아래로 붙여 타입 빈도를 계산합니다.
  - `type_avg.sort_values(stat, ascending=False).head(3)` : 특정 스탯 기준 내림차순 정렬 후 상위 3개만 추출합니다.
  - `total_top = df.groupby("Type 1")["Total"].mean().sort_values(...).head(3).reset_index()` : 그룹 집계 → 정렬 → 상위 행 선택 → 인덱스 리셋까지 한 번에 체인 방식으로 작성합니다.
  - `df["Generation"].nunique()` / `df["Total"].mean()` : KPI 카드에 사용되는 고유 세대 수, 전체 평균 Total을 간단한 Series 메서드로 구합니다.
  - `bar_data = type_counts.astype(int)` : `value_counts` 결과를 Streamlit 차트에 맞게 정수형으로 변환합니다.
  - `round(avg_value, 2)` / `int(max_row[stat_choice])` : 출력 표에서 소수 둘째 자리까지 반올림하거나 정수형으로 변환해 값 표현을 통일합니다.

- **Seaborn/Matplotlib 설정**
  - `sns.set_theme(style="whitegrid")` : 그래프 전역 스타일을 설정해 모든 차트가 일관된 배경/그리드를 갖습니다.
  - `fig, ax = plt.subplots(figsize=(5, 3))` : 명시적 Figure/Axes 생성 후 Seaborn 플롯을 배치하면 Streamlit에서 레이아웃 제어가 용이합니다.
  - `ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)` : 레이더 차트에서 각도 라벨을 한글 폰트로 표시하기 위해 축 설정을 별도로 수행합니다.
  - `plt.close(fig)` : Streamlit에 그래프를 전달한 후 리소스를 해제하여 메모리 누수를 방지합니다.

- **네트워크/예외 처리**
  - `requests.get(url, timeout=10)` : PokéAPI 호출 시 타임아웃을 지정해 Streamlit 앱이 장시간 멈추는 것을 방지합니다.
  - `resp.raise_for_status()` : HTTP 에러를 즉시 감지해 `except` 블록으로 넘어가 대체 이미지를 로딩합니다.
  - `except Exception: return get_sprite_url(fallback_dex)` : 모든 예외를 포괄해도 앱이 죽지 않고 백업 URL을 사용하는 방어 코드를 포함합니다.

- **컬럼 생성과 사용자 정의 함수**
  - `df["Generation"] = df["DexID"].apply(assign_generation)` : 사용자 정의 경계 로직을 적용해 세대 정보를 결정합니다.
  - `type_filtered["DisplayType"] = type_filtered.apply(pick_display_type, axis=1)` : 행 단위로 함수를 적용하여 산점도의 색상 기준을 결정합니다.
  - `option_label(row)` : 문자열 포맷팅으로 `#001 Bulbasaur • Gen 1 • Total 318`처럼 직관적인 라벨을 만들어 선택 UI 전반에 재사용합니다.
