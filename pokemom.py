from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Pokédex Insights",
    page_icon=":sparkles:",
    layout="wide",
)

def configure_font():
    preferred_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Arial Unicode MS"]
    available = {font.name for font in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


configure_font()
sns.set_theme(style="whitegrid")

STAT_COLS = ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"]
STAT_DISPLAY = ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"]
SPRITE_TEMPLATE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{dex}.png"


@lru_cache(maxsize=None)
def get_sprite_url(dex_id: int) -> str:
    return SPRITE_TEMPLATE.format(dex=dex_id)


@lru_cache(maxsize=512)
def fetch_sprite(name: str, fallback_dex: int) -> str:
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{name.lower().strip()}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        sprite = (
            data["sprites"]["other"]["official-artwork"]["front_default"]
            or data["sprites"]["front_default"]
        )
        if sprite:
            return sprite
        return get_sprite_url(int(data["id"]))
    except Exception:
        return get_sprite_url(fallback_dex)


def option_label(row: pd.Series) -> str:
    form = str(row.get("Evolution", "")).strip()
    form_suffix = f" ({form})" if form else ""
    fallback = row.name if row.name is not None else row.get("EntryID", 0)
    dex_id = int(row.get("DexID", fallback))
    return f"#{dex_id:03d} {row['Name']}{form_suffix} • Gen {row['Generation']} • Total {row['Total']}"


def assign_generation(dex_id: int) -> int:
    boundaries = {
        1: 151,
        2: 251,
        3: 386,
        4: 493,
        5: 649,
        6: 721,
        7: 809,
        8: 905,
        9: 1010,
    }
    for gen, upper in boundaries.items():
        if dex_id <= upper:
            return gen
    return 9


def parse_numeric_front(value: Any, default: float = np.nan) -> float:
    match = re.search(r"(\d+(\.\d+)?)", str(value))
    return float(match.group(1)) if match else default


def infer_types(cell: str | float | int) -> tuple[str, str]:
    tokens = str(cell).replace("/", " ").replace(",", " ").split()
    type1 = tokens[0].title() if tokens else "Unknown"
    type2 = tokens[1].title() if len(tokens) > 1 else "None"
    return type1, type2


def build_natdex_ids(names: pd.Series) -> pd.Series:
    """Assign the same Pokédex number to alternate forms that share a base name."""
    seen: dict[str, int] = {}
    next_id = 0
    natdex: list[int] = []
    for name in names:
        if name not in seen:
            next_id += 1
            seen[name] = next_id
        natdex.append(seen[name])
    return pd.Series(natdex, index=names.index, dtype=int)


@st.cache_data(show_spinner=False)
def load_data(path: str = "pokemon.csv") -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        st.error("pokemon.csv 파일을 찾을 수 없습니다.")
        st.stop()

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns=lambda c: c.strip())
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df.insert(0, "EntryID", np.arange(1, len(df) + 1))
    df.insert(1, "DexID", build_natdex_ids(df["Name"]))
    df["Type 1"], df["Type 2"] = zip(*df["Type"].apply(infer_types))
    df["Generation"] = df["DexID"].apply(assign_generation)
    df["CatchRate"] = df["catch_rate"].apply(parse_numeric_front)
    df["BaseFriendship"] = df["base_friendship"].apply(parse_numeric_front)
    df["Height_m"] = df["Height"].apply(parse_numeric_front)
    df["Weight_kg"] = df["Weight"].apply(parse_numeric_front)
    return df


def radar_chart(stats: dict[str, float], title: str = "Base Stat Radar", figsize: float = 2.4):
    labels = STAT_DISPLAY
    ordered = [stats.get(label, 0) for label in labels]
    values = ordered + ordered[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    max_stat = max(values)
    upper = min(200, max(80, max_stat + 20))

    fig, ax = plt.subplots(figsize=(figsize, figsize), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#ff5c8a", linewidth=2)
    ax.fill(angles, values, color="#ff5c8a", alpha=0.3)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(0, upper)
    ax.set_rgrids(np.linspace(20, upper, 4), angle=90, color="#94a3b8", fontsize=7)
    ax.set_title(title, pad=12, fontsize=10)
    ax.grid(color="#94a3b8", linewidth=0.4, alpha=0.6)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def render_dashboard(df: pd.DataFrame):
    st.header("홈")
    img_col, text_col = st.columns([1, 3])
    with img_col:
        st.image(
            "123.png",
            width=220,
            caption="오박사의 도감 브리핑",
        )
    with text_col:
        st.markdown(
            """
            <div style="margin-top:160px;">
            포켓몬은 각각 체력(HP), 공격, 방어, 특수공격, 특수방어, 스피드의 6가지 스탯을 가지며<br/>
            이 스탯을 모두 합한 값을 <strong>종족값(Total)</strong> 이라고 부릅니다.<br/>
            종족값이 높을수록 전반적인 성능이 좋은 포켓몬일 확률이 높아요.<br/>
            아래 카드와 그래프는 이런 종족값 관점에서 도감 데이터를 빠르게 훑어볼 수 있게 정리해 둔 것입니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
    total_pkm = len(df)
    generation_count = df["Generation"].nunique()
    avg_total = round(df["Total"].mean(), 1)
    common_type = df["Type 1"].mode().iat[0]

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("포켓몬 수", f"{total_pkm:,}")
    kpi_cols[1].metric("세대 수", generation_count)
    kpi_cols[2].metric("평균 Total", avg_total)
    kpi_cols[3].metric("가장 많은 타입", common_type)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("세대별 평균 Total")
        gen_avg = df.groupby("Generation")["Total"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.lineplot(data=gen_avg, x="Generation", y="Total", marker="o", ax=ax, color="#2563eb")
        ax.set_ylabel("Average Total")
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.subheader("타입별 포켓몬 수 (Type 1)")
        type_counts = df["Type 1"].value_counts().reset_index()
        type_counts.columns = ["Type 1", "Count"]
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=type_counts, y="Type 1", x="Count", ax=ax, palette="viridis")
        ax.set_xlabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        """
        <div style="font-size:0.9rem; color:#475569;">
        데이터 출처<br/>
        • 포켓몬 종합 정보: <a href="https://zenodo.org/records/4661775" target="_blank">Zenodo Pokémon CSV</a><br/>
        • 스프라이트 및 상세 정보: <a href="https://pokeapi.co/" target="_blank">PokéAPI</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explorer(df: pd.DataFrame):
    st.header("포켓몬 탐색기")
    with st.sidebar.expander("탐색기 필터", expanded=True):
        generations = sorted(df["Generation"].unique())
        gen_filter = st.multiselect("세대", generations, default=generations)
        types = sorted(set(df["Type 1"]).union(df["Type 2"]))
        type1_filter = st.multiselect("타입 1", types, default=types)
        type2_filter = st.multiselect("타입 2", types + ["None"], default=types + ["None"])
        stat_filters = {}
        for col in ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"]:
            min_val, max_val = int(df[col].min()), int(df[col].max())
            stat_filters[col] = st.slider(f"{col} 범위", min_val, max_val, (min_val, max_val))
        name_query = st.text_input("이름 검색 (부분 문자열)", "")

    filtered = df[
        df["Generation"].isin(gen_filter)
        & df["Type 1"].isin(type1_filter)
        & df["Type 2"].isin(type2_filter)
    ]

    for col, (min_v, max_v) in stat_filters.items():
        filtered = filtered[(filtered[col] >= min_v) & (filtered[col] <= max_v)]

    if name_query:
        filtered = filtered[filtered["Name"].str.contains(name_query, case=False, na=False)]

    st.write(f"🔍 조건에 맞는 포켓몬: **{len(filtered)}마리**")
    display_cols = ["DexID", "Name", "Type 1", "Type 2", "Generation", "Total"] + STAT_COLS
    st.dataframe(filtered[display_cols], use_container_width=True, height=320)

    if not filtered.empty:
        detail_query = st.text_input("상세 검색 (포켓몬 이름 또는 도감 번호)", "")
        candidate_df = filtered.copy()
        if detail_query:
            candidate_df = candidate_df[
                candidate_df["Name"].str.contains(detail_query, case=False, na=False)
                | candidate_df["DexID"].astype(str).str.contains(detail_query)
            ]
        if candidate_df.empty:
            st.info("검색과 필터 조건에 맞는 포켓몬이 없습니다.")
            return
        candidate_df = candidate_df.set_index("EntryID")
        selected_entry = st.selectbox(
            "상세 확인할 포켓몬",
            candidate_df.index.tolist(),
            format_func=lambda entry: option_label(candidate_df.loc[entry]),
        )
        detail = candidate_df.loc[selected_entry]
        st.subheader(f"{detail['Name']} 상세")
        cols = st.columns([1, 1])
        with cols[0]:
            sprite_url = fetch_sprite(detail["Name"], int(detail["DexID"]))
            st.image(sprite_url, width=180, caption="Official Artwork")
            st.write(
                f"""
                - 도감 번호: {int(detail['DexID'])}
                - 세대: {detail['Generation']}
                - 타입: {detail['Type 1']} / {detail['Type 2']}
                - Total: {detail['Total']}
                """
            )
        with cols[1]:
            radar_chart(detail[STAT_COLS].to_dict(), f"{detail['Name']} Base Stats")
    else:
        st.info("필터를 완화하면 결과를 볼 수 있습니다.")


def render_type_analysis(df: pd.DataFrame):
    st.header("타입 분석")
    type_avg = df.groupby("Type 1")[STAT_COLS].mean().reset_index()
    st.subheader("타입별 평균 스탯")
    type_list = sorted(type_avg["Type 1"].unique())
    default_selection = type_list[:3]
    selected_types = st.multiselect(
        "비교할 타입을 선택하세요 (최대 5개 권장)",
        type_list,
        default=default_selection,
    )
    if not selected_types:
        st.info("타입을 최소 1개 선택하면 평균 스탯을 볼 수 있습니다.")
    else:
        subset = type_avg[type_avg["Type 1"].isin(selected_types)]
        melted = subset.melt(id_vars="Type 1", var_name="Stat", value_name="Value")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=melted, x="Stat", y="Value", hue="Type 1", ax=ax)
        ax.set_ylabel("Average Stat")
        ax.legend(title="Type")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(subset.set_index("Type 1"), use_container_width=True)

    st.subheader("능력치별 상위 타입 TOP3")
    rank_cols = st.columns(3)
    stat_groups = [
        ("🔥 공격형", ["Attack", "Sp.Atk"]),
        ("🛡️ 방어형", ["Defense", "Sp.Def"]),
        ("⚡ 속도형", ["Speed"]),
    ]
    for idx, (title, stats) in enumerate(stat_groups):
        with rank_cols[idx]:
            st.markdown(f"**{title}**")
            for stat in stats:
                top = type_avg.sort_values(stat, ascending=False).head(3)[["Type 1", stat]]
                st.table(top.rename(columns={"Type 1": "Type", stat: stat}))
    st.markdown("**📊 종합 TOP3 (Total 기준)**")
    total_top = (
        df.groupby("Type 1")["Total"].mean().sort_values(ascending=False).head(3).reset_index()
    )
    st.table(total_top.rename(columns={"Type 1": "Type", "Total": "Avg Total"}))


def render_stat_distribution(df: pd.DataFrame):
    st.header("스탯 분포")
    stat_choice = st.selectbox("분포 확인할 스탯", ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"])
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df[stat_choice], kde=True, ax=ax, color="#0ea5e9")
    ax.set_title(stat_choice)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("스탯 요약")
    avg_value = df[stat_choice].mean()
    max_row = df.loc[df[stat_choice].idxmax()]
    summary_df = pd.DataFrame(
        {
            "통계": ["평균", "최고치"],
            "값": [round(avg_value, 2), int(max_row[stat_choice])],
            "포켓몬": ["-", max_row["Name"]],
        }
    )
    st.table(summary_df)


def render_size_page(df: pd.DataFrame):
    st.header("키·몸무게 분석")
    search_col, select_col = st.columns([2, 2])
    with search_col:
        name_query = st.text_input("포켓몬 이름 검색", "")
        candidate_df = (
            df[df["Name"].str.contains(name_query, case=False, na=False)]
            if name_query
            else df
        )
    with select_col:
        candidate_df = candidate_df.set_index("EntryID")
        selected_entry = st.selectbox(
            "포켓몬 선택",
            candidate_df.index.tolist(),
            format_func=lambda entry: option_label(candidate_df.loc[entry]),
        )

    selected_row = candidate_df.loc[selected_entry]
    st.subheader(f"{selected_row['Name']} 키·몸무게")
    st.write(f"- 키: {selected_row['Height_m']} m")
    st.write(f"- 몸무게: {selected_row['Weight_kg']} kg")
    st.write(f"- 타입: {selected_row['Type 1']} / {selected_row['Type 2']}")
    st.image(
        fetch_sprite(selected_row["Name"], int(selected_row["DexID"])),
        width=200,
        caption="Official Artwork",
    )

    st.subheader("타입별 키·몸무게 분포")
    type_options = sorted(set(df["Type 1"]).union(df["Type 2"]))
    selected_types = st.multiselect("타입 선택", type_options, default=type_options[:3])
    if selected_types:
        type_filtered = df[
            df["Type 1"].isin(selected_types) | df["Type 2"].isin(selected_types)
        ]
        def pick_display_type(row):
            for t in selected_types:
                if row["Type 1"] == t or row["Type 2"] == t:
                    return t
            return row["Type 1"]

        type_filtered = type_filtered.copy()
        type_filtered["DisplayType"] = type_filtered.apply(pick_display_type, axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = sns.scatterplot(
            data=type_filtered,
            x="Weight_kg",
            y="Height_m",
            hue="DisplayType",
            ax=ax,
            s=50,
        )
        ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_xlabel("Weight (kg)")
        ax.set_ylabel("Height (m)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("타입을 최소 1개 선택하면 그래프가 표시됩니다.")


def render_team_builder(df: pd.DataFrame):
    st.header("팀 빌더 & 밸런스 체크")
    options_df = df.set_index("EntryID")
    team_choices = st.multiselect(
        "최대 6마리 선택",
        options_df.index.tolist(),
        max_selections=6,
        format_func=lambda entry: option_label(options_df.loc[entry]),
    )
    if not team_choices:
        st.info("팀을 선택하면 스탯 요약을 볼 수 있습니다.")
        return

    team_df = options_df.loc[team_choices].reset_index().rename(columns={"index": "EntryID"})
    summary = team_df[["Total"] + STAT_COLS].agg(["sum", "mean"]).T
    summary.columns = ["합계", "평균"]
    st.table(summary)

    radar_chart(team_df[STAT_COLS].mean().to_dict(), "Mean Total", figsize=2.8)

    type_counts = (
        pd.concat([team_df["Type 1"], team_df["Type 2"]])
        .value_counts()
        .drop(labels=["None"], errors="ignore")
    )
    st.write("### 타입 분포")
    bar_data = type_counts.astype(int)
    st.bar_chart(bar_data)

    st.write("### 팀 구성 이미지")
    image_cols = st.columns(min(6, len(team_choices)))
    for idx, entry_id in enumerate(team_choices):
        col = image_cols[idx % len(image_cols)]
        row = options_df.loc[entry_id]
        with col:
            st.image(fetch_sprite(row["Name"], int(row["DexID"])), width=120, caption=row["Name"])


def render_playground(df: pd.DataFrame):
    st.header("EDA 플레이그라운드")
    x_axis = st.selectbox("X 축 컬럼", ["Total"] + STAT_COLS, index=0)
    y_axis = st.selectbox("Y 축 컬럼", STAT_COLS, index=1)
    hue = st.selectbox("Hue", ["None", "Generation", "Type 1"], index=0)
    plot_type = st.selectbox("그래프 유형", ["scatter", "box", "violin"])

    hue_arg = hue if hue != "None" else None
    fig, ax = plt.subplots(figsize=(6, 4))
    if plot_type == "scatter":
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue_arg, ax=ax)
    elif plot_type == "box":
        sns.boxplot(data=df, x=x_axis, y=y_axis, hue=hue_arg, ax=ax)
    else:
        if hue_arg:
            sns.violinplot(data=df, x=x_axis, y=y_axis, hue=hue_arg, ax=ax, split=True)
        else:
            sns.violinplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)
    plt.close(fig)


def main():
    df = load_data()
    with st.sidebar:
        logo_col, title_col = st.columns([1, 3])
        with logo_col:
            st.image("pokemon.jpg", width=70)
        with title_col:
            st.markdown("### Pokédex 네비게이션")
    pages = [
        "홈",
        "포켓몬 탐색기",
        "타입 분석",
        "스탯 분포",
        "키·몸무게",
        "팀 빌더",
    ]
    page = st.sidebar.radio("페이지 선택", pages)

    if page == "홈":
        render_dashboard(df)
    elif page == "포켓몬 탐색기":
        render_explorer(df)
    elif page == "타입 분석":
        render_type_analysis(df)
    elif page == "스탯 분포":
        render_stat_distribution(df)
    elif page == "키·몸무게":
        render_size_page(df)
    elif page == "팀 빌더":
        render_team_builder(df)


if __name__ == "__main__":
    main()
