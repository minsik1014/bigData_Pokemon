# Pokédex Insights – 데이터 처리 및 시각화 개요

이 문서는 `pokemom.py` Streamlit 앱을 구성한 핵심 로직을 간단한 README 형식으로 정리한 것입니다.

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
