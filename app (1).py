
import io
import random
import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model

st.set_page_config(page_title="매칭 프로그램", layout="wide")

# ------------------------------
# 가나다 정렬 키
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
# 전처리
# ------------------------------
AGE_BANDS = ["10대","20대","30대","40대","50대","60대","70대"]

def age_to_band(age) -> str:
    try:
        a = int(age)
    except Exception:
        return None
    if a < 20:  return "10대"
    if a < 30:  return "20대"
    if a < 40:  return "30대"
    if a < 50:  return "40대"
    if a < 60:  return "50대"
    if a < 70:  return "60대"
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
# 팀 크기: 6명 고정
# ------------------------------
def choose_group_sizes(N: int):
    if N % 6 != 0:
        need = 6 - (N % 6)
        if need == 6: need = 0
        return None, f"해결 실패: 6인 고정 규칙상 총원 {N}명은 6의 배수여야 합니다. ±{need if need!=0 else 6}명 조정 후 다시 시도하세요."
    return [6] * (N // 6), None

def allowed_male_bounds(_: int):
    return 2, 4  # 6인 전용

# ------------------------------
# 핵심 솔버
# ------------------------------
def solve_assignment(df, seed=0, time_limit=20, max_per_church=4):
    people = df.to_dict('records')
    N = len(people)
    sizes, warn = choose_group_sizes(N)
    if sizes is None:
        return None, None, warn, None
    G = len(sizes)

    males = [i for i,p in enumerate(people) if p['성별'] == '남']

    churches = sorted(df['교회 이름'].fillna("미상").astype(str).unique().tolist())
    church_members = {c: [i for i,p in enumerate(people) if str(p['교회 이름']) == c] for c in churches}

    bands = AGE_BANDS
    band_members = {b: [i for i,p in enumerate(people) if p['나이대'] == b] for b in bands}

    # Precheck
    for c, members in church_members.items():
        if len(members) > max_per_church * G:
            return None, None, f"불가능: 교회 '{c}' 인원 {len(members)} > 허용 {max_per_church*G}", None
    for b, members in band_members.items():
        if len(members) > 3 * G:
            return None, None, f"불가능: 나이대 '{b}' 인원 {len(members)} > 허용 {3*G}", None

    # 초과 필요 계산
    church_counts = {c: len(members) for c, members in church_members.items()}
    extra_needed = {c: max(0, cnt - 2*G) for c, cnt in church_counts.items()}
    age_counts = {b: len(members) for b, members in band_members.items()}
    age_extra_needed = {b: max(0, cnt - 2*G) for b, cnt in age_counts.items()}

    model = cp_model.CpModel()

    # 배정 변수
    x = {}
    for i in range(N):
        for g in range(G):
            x[(i,g)] = model.NewBoolVar(f"x_{i}_{g}")

    # 각 사람 정확히 1팀
    for i in range(N):
        model.Add(sum(x[(i,g)] for g in range(G)) == 1)

    # 팀 크기=6 하드
    for g in range(G):
        model.Add(sum(x[(i,g)] for i in range(N)) == 6)

    # 성비 (2..4) with slack
    sL, sU = [], []
    for g in range(G):
        mc = model.NewIntVar(0, 6, f"male_{g}")
        model.Add(mc == sum(x[(i,g)] for i in males))
        lo, hi = allowed_male_bounds(6)
        sl = model.NewIntVar(0, 6, f"sL_{g}")
        su = model.NewIntVar(0, 6, f"sU_{g}")
        model.Add(mc >= lo - sl)
        model.Add(mc <= hi + su)
        sL.append(sl); sU.append(su)

    # 동일 교회: 기본 ≤2, 불가 시 3·4만 (≤4 하드), 정확히 필요한 만큼
    zero = model.NewIntVar(0, 0, "zero_const")
    is4_flags = []
    shortfall_church = {}
    for c in churches:
        z_vars = []
        members = church_members[c]
        for g in range(G):
            cnt = model.NewIntVar(0, min(max_per_church, len(members)), f"church_{c}_{g}")
            model.Add(cnt == sum(x[(i,g)] for i in members))
            model.Add(cnt <= max_per_church)  # ≤4 하드

            # z = max(0, cnt-2) ∈ {0,1,2}
            t = model.NewIntVar(-2, max_per_church-2, f"t_{c}_{g}")
            model.Add(t == cnt - 2)
            z = model.NewIntVar(0, 2, f"z_{c}_{g}")
            model.AddMaxEquality(z, [t, zero])
            z_vars.append(z)

            is4 = model.NewBoolVar(f"is4_{c}_{g}")
            model.Add(cnt == 4).OnlyEnforceIf(is4)
            model.Add(cnt != 4).OnlyEnforceIf(is4.Not())
            is4_flags.append(is4)

        s_c = model.NewIntVar(0, int(extra_needed[c]), f"short_c_{c}")
        shortfall_church[c] = s_c
        model.Add(sum(z_vars) + s_c == int(extra_needed[c]))

    # 동일 나이대: 기본 ≤2, 불가 시 3만 (≤3 하드), 정확히 필요한 만큼
    is3_age_flags = []
    shortfall_age = {}
    for b in bands:
        members = band_members[b]
        y_vars = []
        for g in range(G):
            cnt = model.NewIntVar(0, min(3, len(members)), f"band_{b}_{g}")
            model.Add(cnt == sum(x[(i,g)] for i in members))
            model.Add(cnt <= 3)  # ≤3 하드
            is3 = model.NewBoolVar(f"is3_{b}_{g}")
            model.Add(cnt == 3).OnlyEnforceIf(is3)
            model.Add(cnt != 3).OnlyEnforceIf(is3.Not())
            is3_age_flags.append(is3)
            y_vars.append(is3)
        s_b = model.NewIntVar(0, int(age_extra_needed[b]), f"short_b_{b}")
        shortfall_age[b] = s_b
        model.Add(sum(y_vars) + s_b == int(age_extra_needed[b]))

    # 목적함수
    rand = random.Random(12345)
    noise = sum(rand.randint(0,1) * x[(i,g)] for i in range(N) for g in range(G))
    model.Minimize(
        5000 * sum(shortfall_church.values()) +
        5000 * sum(shortfall_age.values()) +
        1000 * (sum(sL) + sum(sU)) +
        5 * sum(is4_flags) +
        1 * sum(is3_age_flags) +
        1 * noise
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    try:
        solver.parameters.random_seed = int(seed)
    except Exception:
        pass

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
# 사이드바 (고유 key로 충돌 방지)
# ------------------------------
with st.sidebar:
    st.header("설정", anchor=False)
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="upload_xlsx")
    time_limit = st.slider("해결 시간 제한(초)", min_value=5, max_value=60, value=20, step=1, key="limit_sec")
    # 글자 크기 조정
    title_px = st.slider("제목 글자 크기(px)", 48, 96, 64, 2, key="title_px_slider")
    names_px = st.slider("이름 글자 크기(px)", 24, 64, 36, 2, key="names_px_slider")
    run_btn = st.button("🎲 매칭 시작", key="run_btn_main")

# 글자 크기 적용 CSS
st.markdown(
    f"<style>.team-title{{font-size:{title_px}px;font-weight:800;text-align:center;margin:10px 0 8px;}} "
    f".names-line{{font-size:{names_px}px;line-height:1.9;text-align:center;}}</style>",
    unsafe_allow_html=True,
)

st.write("필수 컬럼: `이름`, `성별(남/여)`, `교회 이름`, `나이`")

# ------------------------------
# 데이터 로드/검증
# ------------------------------
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
    sizes, warn = choose_group_sizes(N)
    if sizes is None:
        st.error(warn); st.stop()

    # 진단 요약 보기
    with st.expander("진단 요약 보기", expanded=False):
        G = len(sizes)
        st.write(f"팀 수: {G}, 팀 크기: {sorted(sizes)}")
        church_counts = df['교회 이름'].fillna('미상').astype(str).str.strip().value_counts().rename_axis('교회').reset_index(name='인원')
        church_counts['초과필요(z합)'] = (church_counts['인원'] - 2*G).clip(lower=0)
        st.dataframe(church_counts, use_container_width=True)
        age_counts = df['나이대'].value_counts().rename_axis('나이대').reset_index(name='인원').sort_values('나이대')
        age_counts['초과필요(3인팀수)'] = (age_counts['인원'] - 2*G).clip(lower=0)
        st.dataframe(age_counts, use_container_width=True)

    if run_btn:
        groups, warn_list, err, sizes = solve_assignment(df, time_limit=time_limit, max_per_church=4)
        if err:
            st.error(err); st.stop()
        for w in (warn_list or []):
            st.warning(w)

        people = df.to_dict("records")
        names_per_team = []
        for g, members in enumerate(groups, start=1):
            team_names = sorted([people[i]["이름"] for i in members], key=hangul_key)
            names_per_team.append(" / ".join(team_names))

        # 상태 저장
        st.session_state.assignment_ready = True
        st.session_state.names_per_team = names_per_team
        st.session_state.team_count = len(names_per_team)
        st.session_state.team_idx = 0
        st.session_state.final_view = False

# ------------------------------
# Viewer
# ------------------------------
if st.session_state.get("assignment_ready", False):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("◀ 이전 팀", key="prev_team"):
            st.session_state.team_idx = (st.session_state.team_idx - 1) % st.session_state.team_count
            st.session_state.final_view = False
    with col2:
        st.markdown(f"<div class='team-title'>{st.session_state.team_idx+1}팀</div>", unsafe_allow_html=True)
    with col3:
        if st.button("다음 팀 ▶", key="next_team"):
            if st.session_state.team_idx < st.session_state.team_count - 1:
                st.session_state.team_idx += 1
                st.session_state.final_view = False
            else:
                st.session_state.final_view = True

    if st.session_state.final_view:
        st.markdown("<div class='team-title'>최종 결과</div>", unsafe_allow_html=True)
        for idx, line in enumerate(st.session_state.names_per_team, start=1):
            st.markdown(f"<div class='team-title'>{idx}팀</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='names-line'>{line}</div>", unsafe_allow_html=True)
    else:
        idx = st.session_state.team_idx
        st.markdown(f"<div class='team-title'>{idx+1}팀</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='names-line'>{st.session_state.names_per_team[idx]}</div>", unsafe_allow_html=True)

    # 다운로드 (팀, 이름)
    rows = []
    for g, line in enumerate(st.session_state.names_per_team, start=1):
        for name in line.split(" / "):
            rows.append({"팀": g, "이름": name})
    out_df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="TeamsOnly")
    st.download_button("결과 엑셀 다운로드(팀+이름, 가나다순)", data=buf.getvalue(),
                       file_name="teams_names_only.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
