
import time
import io
import math
import random
import pandas as pd
import numpy as np
import streamlit as st
from ortools.sat.python import cp_model

# ------------------------------
# Page setup + CSS
# ------------------------------
st.set_page_config(page_title="교회 매칭 프로그램 (팀 번호 + 이름만)", layout="wide")
st.markdown(f"""
<style>
:root {{
  --brand:#2e5a88; --brand-2:#3a7ca5; --bg1:#f5f8fc; --bg2:#eaf2fb; --text:#1b365d; --muted:#7a8ca3;
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, var(--bg1), var(--bg2));
  color: var(--text);
}}
/* Container width & spacing */
.block-container {{
  padding-top: 1rem;
  padding-bottom: 2rem;
  max-width: 1100px;
}}
/* Header */
.app-header {{
  text-align:center; margin: 6px 0 10px 0;
}}
.app-header h1 {{
  font-weight: 800; letter-spacing: 0.3px; margin: 0; color: var(--brand);
}}
.app-sub {{
  text-align:center; color: var(--muted); margin-bottom: 18px;
}}
/* Cards */
.card, .big-card, .team-card {{
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(46,90,136,0.08);
  border-radius: 18px;
  box-shadow: 0 10px 24px rgba(46,90,136,0.10);
  padding: 18px 22px;
  backdrop-filter: blur(6px);
}}
.big-card {{ padding: 26px 28px; }}
.team-card-title {{
  font-weight: 700; color: var(--brand); margin-bottom: 6px; text-align:center;
}}
.team-card-names {{ text-align:center; color: var(--text); }}
/* Titles & Names (sizes overridden by sliders below) */
.team-title {{ text-align:center; font-size: {title_px if 'title_px' in globals() else 64}px; font-weight: 800; margin: 10px 0 8px 0; color: var(--brand); }}
.names-line {{ text-align:center; font-size: {names_px if 'names_px' in globals() else 36}px; line-height: 1.9; color: var(--text); }}
/* Toolbar & badges */
.navbar {{ display:flex; gap:12px; justify-content:center; align-items:center; margin: 12px 0 16px 0; }}
.badge {{ font-weight:600; padding:6px 12px; border-radius:999px; border:1px solid rgba(46,90,136,0.25); background:#fff; }}
/* Streamlit buttons */
.stButton>button, .stDownloadButton>button {{
  border-radius: 12px; padding: 0.45rem 0.9rem; border: 1px solid rgba(46,90,136,0.22);
  background: linear-gradient(180deg, #ffffff, #f2f6fb);
  color: var(--text); font-weight: 600;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
  border-color: var(--brand-2); box-shadow: 0 6px 16px rgba(58,124,165,0.18);
}}
/* Sidebar tweaks */
[data-testid="stSidebar"] .block-container {{ padding-top: 0.6rem; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 가나다순 정렬 키
# ------------------------------
BASE, CHOS, JUNG = 0xAC00, 588, 28
def hangul_key(s: str):
    ks = []
    for ch in str(s):
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            sidx = o - BASE
            cho = sidx // CHOS
            jung = (sidx % CHOS) // JUNG
            jong = sidx % JUNG
            ks.append((0, cho, jung, jong))
        else:
            ks.append((1, o))
    return tuple(ks)

# ------------------------------
# 유틸: 데이터 전처리
# ------------------------------
AGE_BANDS = ["10대","20대","30대","40대","50대","60대","70대"]

def age_to_band(age: int) -> str:
    try:
        a = int(age)
    except Exception:
        return None
    if a < 20:
        return "10대"
    if a < 30:
        return "20대"
    if a < 40:
        return "30대"
    if a < 50:
        return "40대"
    if a < 60:
        return "50대"
    if a < 70:
        return "60대"
    return "70대"

def normalize_gender(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in ["남","남자","M","m","male","Male"]:
        return "남"
    if s in ["여","여자","F","f","female","Female"]:
        return "여"
    return None

# ------------------------------
# 그룹 크기 결정: 7명 기본, 6~8명 허용, 6/8인 조는 최대 4개
# ------------------------------
def choose_group_sizes(N: int, max_offsize: int = 4):
    best = None
    target_T = int(round(N/7))
    for x6 in range(0, max_offsize+1):
        for x8 in range(0, max_offsize - x6 + 1):
            rem = N - (6*x6 + 8*x8)
            if rem < 0:
                continue
            if rem % 7 != 0:
                continue
            x7 = rem // 7
            T = x6 + x7 + x8
            off = x6 + x8
            score = (abs(T - target_T), off, abs(x8 - x6))
            cand = (score, x6, x7, x8)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None, f"해결 실패: 6/7/8인 조의 조합으로 총원 {N}명을 구성할 수 없습니다."
    else:
        (_, x6, x7, x8) = best
        sizes = [6]*x6 + [7]*x7 + [8]*x8
        return sizes, None

def allowed_male_bounds(size):
    if size == 7: return 3,4
    if size == 6: return 2,4
    if size == 8: return 3,5
    lo = int(math.floor(0.4*size))
    hi = int(math.ceil(0.6*size))
    return lo, hi

# ------------------------------
# OR-Tools CP-SAT 모델
# ------------------------------
def solve_assignment(df, seed=0, time_limit=10, max_per_church=4):
    people = df.to_dict('records')
    N = len(people)
    sizes, warn = choose_group_sizes(N, max_offsize=4)
    if sizes is None:
        return None, None, "조 크기 계산 실패", None
    G = len(sizes)

    males = [i for i,p in enumerate(people) if p['성별'] == '남']

    churches = sorted(df['교회 이름'].fillna("미상").astype(str).unique().tolist())
    church_members = {c: [i for i,p in enumerate(people) if str(p['교회 이름']) == c] for c in churches}

    church_counts = {c: len(members) for c, members in church_members.items()}
    # 각 교회는 2명/팀을 기본 목표로 하고, 초과 인원은 반드시 배치해야 하는 'extra'로 계산
    extra_needed = {c: max(0, cnt - 2*G) for c, cnt in church_counts.items()}

    bands = AGE_BANDS
    band_members = {b: [i for i,p in enumerate(people) if p['나이대'] == b] for b in bands}
    # 나이대 초과: 기본 2/팀, 불가 시 3/팀 허용 - 필요한 3인팀 수 계산
    age_counts = {b: len(members) for b, members in band_members.items()}
    age_extra_needed = {b: max(0, cnt - 2*G) for b, cnt in age_counts.items()}

    # 사전 타당성: 교회/나이대 인원수가 max_per_church*G 초과면 불가능
    overload = []
    for c, members in church_members.items():
        if len(members) > max_per_church*G:
            overload.append((c, len(members), max_per_church*G))
    if overload:
        msg = "불가능: 일부 교회 인원이 너무 많아(최대 {max_per_church}명/팀) 배치가 불가합니다.\n" + \
              "\n".join([f" - {c}: {cnt}명 > 허용 {cap}명" for c,cnt,cap in overload])
        return None, None, msg, None
    for b, members in band_members.items():
        if len(members) > 3*G:  # 나이대는 기본 2명, 불가 시 3명 허용
            msg = "불가능: 일부 나이대 인원이 너무 많아(최대 3명/팀(기본 2명, 불가 시 3명)) 배치가 불가합니다.\n" + \
                  "\n".join([f" - {b}: {len(band_members[b])}명 > 허용 {3*G}명"])
            return None, None, msg, None

    model = cp_model.CpModel()

    x = {}
    for i in range(N):
        for g in range(G):
            x[(i,g)] = model.NewBoolVar(f"x_{i}_{g}")

    # 각 사람은 정확히 1개 팀
    for i in range(N):
        model.Add(sum(x[(i,g)] for g in range(G)) == 1)

    # 팀 크기 고정
    for g in range(G):
        model.Add(sum(x[(i,g)] for i in range(N)) == sizes[g])

    # 성비 제약(유연 슬랙 허용)
    sL = []
    sU = []
    for g in range(G):
        mc = model.NewIntVar(0, sizes[g], f"male_{g}")
        model.Add(mc == sum(x[(i,g)] for i in males))
        lo, hi = allowed_male_bounds(sizes[g])
        sl = model.NewIntVar(0, sizes[g], f"sL_{g}")
        su = model.NewIntVar(0, sizes[g], f"sU_{g}")
        model.Add(mc >= lo - sl)
        model.Add(mc <= hi + su)
        sL.append(sl)
        sU.append(su)


    # 교회: 팀당 최대 max_per_church(하드)
    # 기본 목표는 팀당 <=2, 불가피한 경우에만 3·4 허용(정확히 필요한 만큼만)
    church_is3_flags = []  # cnt==3
    church_is4_flags = []  # cnt==4
    church_extras_sum = [] # z = is3 + 2*is4 (팀별 초과합)
    for g in range(G):
        pass  # placeholder to keep loop variable available

    # Per-church per-team variables
    church_cnt = {}  # (c,g) -> IntVar
    church_z = {}    # (c,g) -> IntVar in [0,2]
    for c in churches:
        z_vars = []
        for g in range(G):
            members = church_members[c]
            cnt = model.NewIntVar(0, min(max_per_church, len(members)), f"church_{c}_{g}")
            model.Add(cnt == sum(x[(i,g)] for i in members))
            model.Add(cnt <= max_per_church)
            church_cnt[(c,g)] = cnt

            # is3 / is4 booleans
            is3 = model.NewBoolVar(f"is3_{c}_{g}")
            is4 = model.NewBoolVar(f"is4_{c}_{g}")
            model.Add(cnt == 3).OnlyEnforceIf(is3)
            model.Add(cnt != 3).OnlyEnforceIf(is3.Not())
            model.Add(cnt == 4).OnlyEnforceIf(is4)
            model.Add(cnt != 4).OnlyEnforceIf(is4.Not())
            church_is3_flags.append(is3)
            church_is4_flags.append(is4)

            # z extras: 0 if cnt<=2, 1 if cnt==3, 2 if cnt==4
            z = model.NewIntVar(0, 2, f"z_extra_{c}_{g}")
            model.Add(z == is3 + 2*is4)
            church_z[(c,g)] = z
            z_vars.append(z)

        # 필요한 초과 인원 합을 정확히 맞춤(= 불가피한 경우에만 3/4 허용)
        need = extra_needed[c]
        model.Add(sum(z_vars) == need)

    age_pair_flags = []

    age_is3_flags = []
    for b in bands:
        y_vars = []
        members = band_members[b]
        for g in range(G):
            cnt = model.NewIntVar(0, min(3, len(members)), f"band_{b}_{g}")
            model.Add(cnt == sum(x[(i,g)] for i in members))
            model.Add(cnt <= 3)
            is3 = model.NewBoolVar(f"is3_band_{b}_{g}")
            model.Add(cnt == 3).OnlyEnforceIf(is3)
            model.Add(cnt != 3).OnlyEnforceIf(is3.Not())
            age_is3_flags.append(is3)
            y_vars.append(is3)
        # 필요한 3인 팀 개수만큼 정확히 배치
        model.Add(sum(y_vars) == int(age_extra_needed[b]))

    # 목적함수
    rand = random.Random(int(time.time()) % (10**6))
    noise_terms = []
    for i in range(N):
        for g in range(G):
            w = rand.randint(0, 3)
            if w > 0:
                noise_terms.append(w * x[(i,g)])

    model.Minimize(
        1000 * sum(sL) + 1000 * sum(sU) +
        5 * sum(church_is4_flags) + 2 * sum(church_is3_flags) +
        1 * sum(age_is3_flags) +
        1 * sum(noise_terms)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, "해결 실패: 제약을 만족하는 해를 찾지 못했습니다. (시간 제한/분포 문제 가능)", None

    groups = []
    for g in range(G):
        members = [i for i in range(N) if solver.Value(x[(i,g)]) == 1]
        groups.append(members)

    total_slack = int(sum(solver.Value(v) for v in sL) + sum(solver.Value(v) for v in sU))
    warn_list = []
    if total_slack > 0:
        warn_list.append(f"주의: 성비 제약을 {total_slack}명만큼 완화하여 해를 구성했습니다.")

    return groups, warn_list, None, sizes

# ------------------------------
# UI
# ------------------------------
st.markdown('<div class="app-header"><h1>교회 매칭 프로그램</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">팀 번호 + 이름(가나다순, “/” 구분) • 균형 매칭</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("설정")
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"])
    time_limit = st.slider("해결 시간 제한(초)", min_value=5, max_value=30, value=10, step=1)
    MAX_PER_CHURCH = 4  # 분포 분석 결과: 팀당 동일 교회 최대 4명 필요
    run_btn = st.button("🎲 매칭 시작")

# 글자 크기 조절(조화롭게)
title_px = st.sidebar.slider("제목 글자 크기(px)", 48, 96, 64, 2)
names_px = st.sidebar.slider("이름 글자 크기(px)", 24, 64, 36, 2)
st.markdown(f"""
<style>
.team-title {{text-align:center; font-size: {title_px}px; font-weight: 800; margin: 24px 0 8px 0;}}
.names-line {{text-align:center; font-size: {names_px}px; line-height: 1.8;}}
</style>
""", unsafe_allow_html=True)


st.markdown("필수 컬럼: `이름`, `성별(남/여)`, `교회 이름`, `나이` · 결과는 **팀 번호 + 이름(가나다순, `/` 구분)** 만 표시됩니다.", unsafe_allow_html=True)

df = None
if uploaded is not None:
    try:
        df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"엑셀 읽기 오류: {e}")

if df is not None:
    required = ["이름","성별","교회 이름","나이"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼 누락: {missing}")
        st.stop()

    df = df.copy()
    df["성별"] = df["성별"].apply(normalize_gender)
    if df["성별"].isna().any():
        st.error("성별 값 표준화 실패 행이 있습니다. ('남'/'여'만 허용)")
        st.dataframe(df[df["성별"].isna()])
        st.stop()
    df["나이대"] = df["나이"].apply(age_to_band)
    if df["나이대"].isna().any():
        st.error("나이 → 나이대 변환 실패 행이 있습니다. (정수 나이 필요)")
        st.dataframe(df[df["나이대"].isna()])
        st.stop()

    N = len(df)
    sizes, warn = choose_group_sizes(N, max_offsize=4)
    if sizes is None:
        st.error(warn)
        st.stop()
    st.info(f"총 {N}명 → 후보 그룹 크기: " + ", ".join(map(str, sorted(sizes))))
    if warn:
        st.warning(warn)

    # ### 진단 패널 (요약)
    with st.expander("진단 요약 보기", expanded=False):
        G = len(sizes)
        st.write(f"팀 수: {G}, 팀 크기: {sorted(sizes)}")
        # 교회별/나이대별 초과 요약
        church_counts = df["교회 이름"].fillna("미상").astype(str).str.strip().value_counts().rename_axis("교회").reset_index(name="인원")
        church_counts["초과필요(z합)"] = (church_counts["인원"] - 2*G).clip(lower=0)
        st.dataframe(church_counts)
        age_counts = df["나이대"].value_counts().rename_axis("나이대").reset_index(name="인원").sort_values("나이대")
        age_counts["초과필요(3인팀수)"] = (age_counts["인원"] - 2*G).clip(lower=0)
        st.dataframe(age_counts)

    if run_btn:
        ph = st.empty()
        for pct in range(0, 101, 7):
            ph.progress(pct, text="배치 탐색 중...")
            time.sleep(0.03)

        groups, warn_list, err, sizes = solve_assignment(df, time_limit=time_limit, max_per_church=MAX_PER_CHURCH)

        if err:
            st.error(err)
            st.stop()
        if warn_list:
            for w in warn_list:
                st.warning(w)

        people = df.to_dict('records')

        # Prepare names per team (ga-na-da order, " / " join)
        names_per_team = []
        for g, members in enumerate(groups):
            team_names = [people[i]['이름'] for i in members]
            team_names_sorted = sorted(team_names, key=hangul_key)
            names_per_team.append(" / ".join(team_names_sorted))

        # Initialize session state
        st.session_state.assignment_ready = True
        st.session_state.names_per_team = names_per_team
        st.session_state.team_count = len(names_per_team)
        st.session_state.team_idx = 0
        st.session_state.final_view = False

# Viewer
if st.session_state.get("assignment_ready", False):
    st.markdown("<div class='navbar'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("◀ 이전 팀"):
            st.session_state.team_idx = (st.session_state.team_idx - 1) % st.session_state.team_count
            st.session_state.final_view = False
    with c2:
        if st.button("최종 결과 보기"):
            st.session_state.final_view = True
    with c3:
        st.markdown(f"<span class='badge'>{st.session_state.team_idx+1} / {st.session_state.team_count}팀</span>", unsafe_allow_html=True)
    with c4:
        if st.button("다음 팀 ▶"):
            if st.session_state.team_idx < st.session_state.team_count - 1:
                st.session_state.team_idx += 1
                st.session_state.final_view = False
            else:
                st.session_state.final_view = True
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.final_view:
        st.markdown("<div class='team-title'>최종 결과</div>", unsafe_allow_html=True)
        for g, names_line_tmp in enumerate(st.session_state.names_per_team, start=1):
            st.markdown(f"<div class='names-line'><b>팀 {g}</b> — {names_line_tmp}</div>", unsafe_allow_html=True)
    else:
        cur_idx = st.session_state.team_idx
        st.markdown(f"<div class='team-title'>팀 {cur_idx+1}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='names-line'>{st.session_state.names_per_team[cur_idx]}</div>", unsafe_allow_html=True)

    # Download
    rows = []
    for g, names_line_tmp in enumerate(st.session_state.names_per_team):
        for name in names_line_tmp.split(" / "):
            rows.append({"팀": g+1, "이름": name})
    out_df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="TeamsOnly")
    st.download_button("결과 엑셀 다운로드(팀+이름, 가나다순)", data=buf.getvalue(),
                       file_name="teams_names_only.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("엑셀 업로드 후 '🎲 매칭 시작'을 눌러주세요.")
