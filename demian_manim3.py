"""
데미안 LLM — Manim 영상 (대본: demian_script.md 확정본, 목소리: Vrew, 10분 14초)

· 화면: 1920×1080, 30fps (코드에 고정)
· 여백: 위 5%, 아래 6%, 좌우 약 5% — 이 안에만 그림을 놓습니다
· 화면은 그림·글자·수식·그래프·행렬·3D 만. 하단 자막 없음 (목소리는 Vrew)
· 장면과 그림이 바뀌는 시각은 Vrew 자막(srt)의 문장 시각에 맞춰 둠 (CUES)

이 파일은 demian_llm.py 와 같은 폴더에 두세요.

[실행]
  manim demian_manim.py Full -o demian_v4          # 전체 10분 14초 (1080p 30fps)
  manim demian_manim.py S06Init -o s06              # 장면 하나만

[미리보기 (저화질, 빠름)]  PowerShell 에서 먼저:  $env:PREVIEW=1
  manim -p demian_manim.py S06Init
  미리보기 끄기:  Remove-Item Env:PREVIEW

장면           대본               시작      길이
  S00Opening    오프닝           0:00.00   27.17초
  S01Text       0. 원문          0:27.17   37.80초
  S02Tokenize   1. 토큰화        1:04.97   50.91초
  S03Vocab      2. 어휘사전      1:55.88   37.10초
  S04Encode     3. 인코딩        2:32.98   22.31초
  S05Table      4. 확률표        2:55.29   85.06초
  S06Init       5-1. 초기화      4:20.35   53.84초
  S07Forward    5-2. 순전파      5:14.19   57.31초
  S08Backward   5-3. 역전파      6:11.50   39.03초
  S09Update     5-4. 업데이트    6:50.53   71.47초
  S10Generate   6. 생성          8:02.00   51.39초
  S11Chat       7. 대화          8:53.39   38.47초
  S12Outro      마무리           9:31.86   42.24초
"""

import math
import os
import random
import re
import shutil
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")  # demian_llm 이 창을 띄우지 않게

import demian_llm as d
import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    PI,
    RESAMPLING_ALGORITHMS,
    RIGHT,
    UP,
    AddTextLetterByLetter,
    AnimationGroup,
    Arc,
    Arrow,
    Axes,
    Brace,
    Circumscribe,
    Create,
    CurvedArrow,
    DashedLine,
    Dot,
    Ellipse,
    FadeIn,
    FadeOut,
    FadeTransform,
    Flash,
    Group,
    GrowArrow,
    GrowFromEdge,
    ImageMobject,
    Indicate,
    LaggedStart,
    Line,
    MathTex,
    MoveAlongPath,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Square,
    Succession,
    SurroundingRectangle,
    Text,
    ThreeDScene,
    Transform,
    TransformFromCopy,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    always_redraw,
    config,
    linear,
)

# ── 화면 설정: 1920×1080, 30fps (미리보기는 PREVIEW=1) ─────────
PREVIEW = bool(os.environ.get("PREVIEW"))
config.pixel_width, config.pixel_height = (854, 480) if PREVIEW else (1920, 1080)
config.frame_rate = 15 if PREVIEW else 30
config.background_color = "#0F172A"  # 딥블루
FPS = config.frame_rate

# 안전 영역 (Manim 단위, 화면 = 가로 14.22 × 세로 8)
SAFE_X = 6.4  # 좌우 약 5%
SAFE_TOP = 3.6  # 위 5%   (화면 위 끝 = 4)
SAFE_BOTTOM = -3.52  # 아래 6% (화면 아래 끝 = -4)
TITLE_Y = 3.38  # 장면 제목 자리 (제목 윗변 ≈ 위 여백 5% 선)
TITLE_X = (
    -config.frame_width / 2 + 0.08 * config.frame_width
)  # 제목 왼쪽 끝 = 화면 왼쪽에서 8% (≈ -5.97)
NUM_SIZE = 36  # 숫자 크기 (Consolas, 화면 전체 통일)
TITLE_SIZE = 35  # 장면 제목 글자 크기 (40 → 12% 작게)
TOP_Y = 2.85  # 제목 아래 내용이 시작하는 높이
CONTENT_Y = (
    TITLE_Y - 0.22 + SAFE_BOTTOM
) / 2  # 제목 아래 ~ 아래 여백 사이의 한가운데 (≈ -0.14): 중심 그래픽을 여기에


# 글꼴: 한글·영어·숫자 모두 D2Coding 하나 (한 줄 안에서 높이가 어긋나지 않음)
#       D2Coding 이 설치돼 있지 않으면 맑은 고딕(한글) + Consolas(영어·숫자) 로 대신함
def _has_font(name):
    try:
        import manimpango

        return name in manimpango.list_fonts()
    except Exception:
        return False


if _has_font("D2Coding"):
    KO_FONT = MONO_FONT = "D2Coding"
else:
    print("[알림] D2Coding 글꼴을 찾지 못해 맑은 고딕 + Consolas 로 만듭니다")
    KO_FONT, MONO_FONT = "Malgun Gothic", "Consolas"

# ── 색 규칙 (3Blue1Brown 풍) ─────────────────────────────────
# · 배경: 아주 어두운 딥블루 (완전 검정 X)
# · 주연 색은 2개: 청록 BLUE_C(=Parameter 쪽), 노랑 YELLOW(=Probability·정답 egg 쪽)
#   + 꼭 필요할 때만 3번째 빨강(=Loss·Gradient). 한 장면에 눈에 띄는 색은 최대 3개
# · 나머지는 회색·흐린 톤
# · 수식 속 변수 색 = 그림 속 같은 대상의 색 (W 는 청록, P 는 노랑, Loss 는 빨강)
BLUE_C = "#58C4DD"  # Manim BLUE_C
YELLOW = "#FFFF00"  # Manim YELLOW
YELLOW_D = "#F4D345"  # 조금 차분한 노랑 (면 채우기용)
RED_C = "#FC6255"  # Manim RED_C

C_BG = "#0F172A"  # 딥블루 배경
C_CELL = "#1B2438"  # 행렬 빈 칸 (배경보다 살짝 밝게)
C_TEXT = "#E6E8EE"
C_DIM = "#8B93A3"  # 회색 (조연)
C_FAINT = "#4A5468"  # 더 흐린 회색
C_WHOLE = BLUE_C  # token (통째)
C_PIECE = YELLOW_D  # token 조각
C_PUNCT = C_FAINT  # 문장부호
C_ACCENT = YELLOW_D  # 정답 egg, 강조
C_GOOD = YELLOW_D  # Probability (노랑)
C_PARAM = BLUE_C  # Parameter (청록)
C_W = BLUE_C  # Weight = Parameter
C_B = BLUE_C  # Bias = Parameter
C_NEG = C_FAINT  # 행렬의 − 값 (흐린 회색)
C_POS = RED_C  # Loss (3번째 색)
C_GRAD = RED_C  # Gradient = Loss 의 기울기 → 같은 빨강

USE_LATEX = shutil.which("latex") is not None

# ── Vrew 자막(srt) 시각: 장면 시작(초)과 장면 안 문장 시작(초) ───
START = {
    "S00": 0.0, "S01": 27.17, "S02": 64.97, "S03": 115.88, "S04": 152.98, "S05": 175.29,
    "S06": 260.35, "S07": 314.19, "S08": 371.5, "S09": 410.53, "S10": 482.0, "S11": 533.39,
    "S12": 571.86,
}  # fmt: skip
END = 614.1
CUES = {
    "S00": [0.0, 4.24, 9.84, 13.04, 19.92, 24.32],
    "S01": [0.0, 3.28, 7.36, 11.76, 19.28, 25.2, 29.6, 33.84],
    "S02": [0.0, 3.36, 7.92, 13.44, 18.32, 25.04, 28.72, 32.0, 37.6, 40.4, 46.24],
    "S03": [0.0, 4.48, 8.48, 12.72, 18.16, 23.88, 27.48, 33.0],
    "S04": [0.0, 4.72, 10.4, 13.68, 18.34],
    "S05": [0.0, 5.92, 10.32, 15.2, 18.4, 26.0, 29.92, 35.6, 40.64, 49.28, 54.16, 59.2, 63.01, 67.57, 72.39,
            75.91, 82.87],
    "S06": [0.0, 4.72, 10.24, 16.96, 21.28, 26.4, 34.72, 43.39, 50.27],
    "S07": [0.0, 6.4, 13.6, 18.0, 22.16, 28.88, 32.72, 38.64, 44.48, 51.44],
    "S08": [0.0, 6.96, 11.2, 18.32, 23.68, 29.86, 34.9],
    "S09": [0.0, 5.68, 11.76, 17.2, 20.4, 27.04, 33.49, 42.13, 46.93, 55.25, 61.52, 66.16],
    "S10": [0.0, 4.0, 10.24, 15.36, 19.12, 23.6, 28.08, 34.04, 40.23, 44.83, 48.62],
    "S11": [0.0, 3.52, 9.92, 13.6, 20.48, 26.51, 30.92],
    "S12": [0.0, 5.04, 10.88, 17.12, 20.88, 24.03, 28.91, 37.63],
}  # fmt: skip
ORDER = list(START)


def scene_end(key):
    i = ORDER.index(key)
    return START[ORDER[i + 1]] if i + 1 < len(ORDER) else END


# ── demian_llm 에서 실제 값 계산 ─────────────────────────────
V, N = d.V, d.N
EX, EX_NEXT, EX_TOK = d.EX, d.EX_NEXT, d.START  # ·the → ·egg
NEXT_TOK = d.id_to_tok[EX_NEXT]
N_WORDS, N_TOKENS = len(d.WORDS), len(d.tokens)
N_PARAMS = V * V + V
NONZERO = sum(len(c) for c in d.counts.values())
FORKS = [i for i in d.model if max(d.model[i].values()) < 0.9]
SPLITS = d.split_words()

NET0 = d.BigramParams()
W0, B0 = NET0.W.copy(), NET0.b.copy()
Z0 = W0[EX] + B0
P0 = d.softmax(Z0)
LOSS0 = NET0.forward()[2]
DW0, DB0, _, _ = NET0.grads()

SNAP_K = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
SNAP, WSNAP, GNORM, EGG_P, EGG_W = {}, {}, {}, {}, {}
_net = d.BigramParams()
for _k in range(501):
    if _k in SNAP_K:
        SNAP[_k] = _net.table()
        WSNAP[_k] = _net.W.copy()
        _gw, _gb, _, _ = _net.grads()
        GNORM[_k] = float(np.sqrt((_gw**2).sum() + (_gb**2).sum()))
        EGG_P[_k] = float(SNAP[_k][EX, EX_NEXT])
        EGG_W[_k] = float(_net.W[EX, EX_NEXT])
    if _k < 500:
        _net.step()
HISTORY = _net.history + [_net.forward()[2]]  # 0~500회 Loss
TABLE500 = SNAP[500]
PAIR_LOSS0 = -np.log(
    np.maximum(d.softmax(W0[d.X] + B0)[np.arange(N), d.Y], 1e-12)
)  # 113쌍 각각의 Loss


def w(tok):
    return tok.replace("·", "")


# ── 그리기 도구 ──────────────────────────────────────────────
def ko(s, size=32, color=C_TEXT, max_w=2 * SAFE_X):
    """한글 = 맑은 고딕(Malgun Gothic). 문장 속 영어 단어와 숫자는 Consolas 로"""
    words = {}
    if KO_FONT != MONO_FONT:  # 글꼴이 둘일 때만 영어·숫자를 따로 지정
        words = {
            m.group(0): MONO_FONT
            for m in re.finditer(r"[A-Za-z][A-Za-z\-]*|[0-9][0-9.,%]*", s)
        }
    t = Text(s, font=KO_FONT, font_size=size, color=color, t2f=words)
    if t.width > max_w:
        t.scale_to_fit_width(max_w)
    return t


def mono(s, size=32, color=C_TEXT):
    return Text(s, font=MONO_FONT, font_size=size, color=color)


def tex(latex, plain, size=40, color=C_TEXT, colors=None):
    """수식 (LaTeX 가 없으면 plain 글자로). colors = {수식 조각: 색} → 그림과 같은 색으로"""
    if USE_LATEX:
        try:
            return MathTex(
                latex, font_size=size, color=color, tex_to_color_map=colors or {}
            )
        except Exception:
            pass
    return mono(plain, int(size * 0.85), color)


def en(s, size=40, color=C_TEXT, max_w=2 * SAFE_X):
    """영어 = Consolas (화면 전체 영어 글꼴 통일)"""
    t = Text(s, font=MONO_FONT, font_size=size, color=color)
    if t.width > max_w:
        t.scale_to_fit_width(max_w)
    return t


def fit(m, wmax=None, hmax=None):
    if wmax and m.width > wmax:
        m.scale_to_fit_width(wmax)
    if hmax and m.height > hmax:
        m.scale_to_fit_height(hmax)
    return m


def role_color(tok, role):
    if not w(tok).isalpha():
        return C_PUNCT
    return C_PIECE if role in ("FIRST", "CONT") else C_WHOLE


def token_box(tok, role="WHOLE", size=32, color=None):
    """token 상자. 글자를 '기준선(baseline)'에 맞춰 놓아서
    ·the(위로 솟은 글자)와 ·egg(아래로 내려간 글자)가 같은 높이에 보이게 한다"""
    c = color or role_color(tok, role)
    # 양끝에 보이지 않는 '|' 를 붙이면 모든 token 의 글자 높이(위~아래)가 같아진다
    t = mono("|" + tok + "|", size)
    t[0].set_opacity(0)
    t[-1].set_opacity(0)
    inner = VGroup(*t[1:-1])
    box = RoundedRectangle(corner_radius=0.08, width=inner.width + 0.3, height=size / 32 * 0.68,
                           stroke_color=c, stroke_width=3, fill_color=c, fill_opacity=0.22)  # fmt: skip
    t.move_to(box)
    t.shift(RIGHT * (box.get_x() - inner.get_x()))  # 보이는 글자를 가로 가운데로
    return VGroup(box, t)


def flow(items, max_w=2 * SAFE_X, gap=0.18, row_gap=0.3, top=TOP_Y):
    rows, row, wsum = [], [], 0
    for it in items:
        need = it.width + (gap if row else 0)
        if row and wsum + need > max_w:
            rows.append(row)
            row, wsum, need = [], 0, it.width
        row.append(it)
        wsum += need
    if row:
        rows.append(row)
    y = top
    for row in rows:
        h = max(it.height for it in row)
        x = -(sum(it.width for it in row) + gap * (len(row) - 1)) / 2
        for it in row:
            it.move_to([x + it.width / 2, y - h / 2, 0], aligned_edge=UP)
            x += it.width + gap
        y -= h + row_gap
    return VGroup(*items)


def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _image(arr, height, width=None):
    img = ImageMobject(np.clip(arr, 0, 255).astype(np.uint8))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.height = height
    if width is not None:
        img.stretch_to_fit_width(width)
    img.set_z_index(-1)  # 테두리·강조 표시가 항상 그림 위에 보이게
    return img


def prob_image(P, height=4.0, only=None, dim_rows=None):
    """확률 Matrix 그림 (0 = 어두움, 1 = 노랑)"""
    v = np.sqrt(np.clip(P, 0, 1))[..., None]
    arr = _hex(C_CELL) * (1 - v) + _hex(C_ACCENT) * v
    if only is not None:
        arr = np.where(only[..., None], _hex(C_ACCENT), _hex(C_CELL))
    if dim_rows is not None:
        arr[dim_rows] *= 0.3
    return _image(arr, height)


def signed_image(M, height=4.0, scale=None, pos=C_POS, neg=C_NEG, width=None):
    """+ 값은 pos 색, − 값은 neg 색"""
    s = scale or (np.abs(M).max() + 1e-12)
    v = np.clip(M / s, -1, 1)[..., None]
    base = _hex(C_CELL)
    arr = np.where(
        v > 0, base * (1 - v) + _hex(pos) * v, base * (1 + v) + _hex(neg) * (-v)
    )
    return _image(arr, height, width)


def frame_of(img, color=C_DIM, sw=2, grid=True):
    """Matrix 테두리: [0] 바깥선 · [1] 수학 행렬 괄호 [ ] · [2] 칸 격자(희미하게)"""
    wd, ht = img.width, img.height
    L, Rr, T, B = (
        img.get_left()[0],
        img.get_right()[0],
        img.get_top()[1],
        img.get_bottom()[1],
    )
    border = Rectangle(
        width=wd, height=ht, stroke_color=color, stroke_width=sw
    ).move_to(img)
    g, tick = 0.14, min(0.28, wd * 0.08)
    lb = VMobject(stroke_color=color, stroke_width=sw + 2).set_points_as_corners(
        [
            [L - g + tick, T + g, 0],
            [L - g, T + g, 0],
            [L - g, B - g, 0],
            [L - g + tick, B - g, 0],
        ]
    )
    rb = VMobject(stroke_color=color, stroke_width=sw + 2).set_points_as_corners(
        [
            [Rr + g - tick, T + g, 0],
            [Rr + g, T + g, 0],
            [Rr + g, B - g, 0],
            [Rr + g - tick, B - g, 0],
        ]
    )
    lines = VGroup()
    if grid:
        rows = 1 if ht < wd / 5 else V
        for i in range(1, V):
            x = L + wd * i / V
            lines.add(
                Line(
                    [x, T, 0],
                    [x, B, 0],
                    stroke_width=0.6,
                    stroke_color=C_TEXT,
                    stroke_opacity=0.10,
                )
            )
        for i in range(1, rows):
            y = T - ht * i / rows
            lines.add(
                Line(
                    [L, y, 0],
                    [Rr, y, 0],
                    stroke_width=0.6,
                    stroke_color=C_TEXT,
                    stroke_opacity=0.10,
                )
            )
    return VGroup(border, VGroup(lb, rb), lines)


def cell_rect(img, row=None, col=None, color=C_ACCENT, sw=4):
    """Matrix 안의 강조: 가로줄(row) 띠 · 세로줄(col) 띠 · 한 칸(row, col) (빛나는 네모)"""
    cw, ch = img.width / V, img.height / V
    left, top = img.get_left()[0], img.get_top()[1]
    if row is not None and col is not None:
        c = [left + cw * (col + 0.5), top - ch * (row + 0.5), 0]
        s_ = max(cw, ch)
        glow = Square(
            s_ * 3.2, stroke_width=0, fill_color=color, fill_opacity=0.18
        ).move_to(c)
        core = Square(
            s_ * 1.5,
            stroke_color=color,
            stroke_width=sw,
            fill_color=color,
            fill_opacity=0.55,
        ).move_to(c)
        return VGroup(glow, core)
    if row is not None:
        return Rectangle(width=img.width, height=ch * 1.15, stroke_color=color, stroke_width=sw * 0.6,
                         fill_color=color, fill_opacity=0.28).move_to([img.get_x(), top - ch * (row + 0.5), 0])  # fmt: skip
    return Rectangle(width=cw * 1.15, height=img.height, stroke_color=color, stroke_width=sw * 0.6,
                     fill_color=color, fill_opacity=0.28).move_to([left + cw * (col + 0.5), img.get_y(), 0])  # fmt: skip


def band_label(band, text, color, side):
    """강조 띠 안쪽 끝에 붙는 작은 이름표 (row 띠 → 오른쪽 끝, col 띠 → 아래 끝)"""
    t = mono(text, 24, C_BG)
    bg = RoundedRectangle(corner_radius=0.06, width=t.width + 0.2, height=t.height + 0.14, stroke_width=0,
                          fill_color=color, fill_opacity=1)  # fmt: skip
    tag = VGroup(bg, t.move_to(bg))
    if side == "row":
        tag.move_to(band.get_right() + LEFT * (tag.width / 2 + 0.06))
    else:
        tag.move_to(band.get_bottom() + UP * (tag.height / 2 + 0.06))
    return tag


def cell_center(img, row, col):
    cw, ch = img.width / V, img.height / V
    return np.array(
        [img.get_left()[0] + cw * (col + 0.5), img.get_top()[1] - ch * (row + 0.5), 0]
    )


def bars(
    values,
    width=10.0,
    base_y=0.0,
    center_x=0.0,
    scale=1.0,
    colors=None,
    highlight=None,
    hcolor=C_ACCENT,
):
    """막대 (양수 위, 음수 아래)"""
    n = len(values)
    bw = width / n
    g = VGroup()
    for i, v in enumerate(values):
        h = max(abs(v) * scale, 0.012)
        c = (
            hcolor
            if i == highlight
            else (colors[i] if colors else (C_GOOD if v >= 0 else C_NEG))
        )
        r = Rectangle(
            width=bw * 0.78, height=h, stroke_width=0, fill_color=c, fill_opacity=0.92
        )
        r.move_to(
            [
                center_x - width / 2 + bw * (i + 0.5),
                base_y + (h / 2 if v >= 0 else -h / 2),
                0,
            ]
        )
        g.add(r)
    return g


def line_text(text, size=36, color=C_TEXT, font=None):
    """한 줄 글자. 양끝의 보이지 않는 '|' 덕분에 어떤 글자든 위·아래 높이가 같다
    → 나란히 놓으면 크기·높이가 똑같이 맞는다"""
    t = Text("|" + text + "|", font=font or MONO_FONT, font_size=size, color=color)
    t[0].set_opacity(0)
    t[-1].set_opacity(0)
    return t


def bubble(text, fill, opacity, size=36, slant="NORMAL"):
    """말풍선. 질문·대답 모두 같은 글자 크기, 같은 높이, 같은 기준선(baseline)"""
    t = Text(
        "|" + text + "|", font=MONO_FONT, font_size=size, color=C_TEXT, slant=slant
    )
    t[0].set_opacity(0)
    t[-1].set_opacity(0)
    glyphs = VGroup(*t[1:-1])
    b = RoundedRectangle(
        corner_radius=0.3,
        width=glyphs.width + 0.8,
        height=1.0,
        stroke_width=0,
        fill_color=fill,
        fill_opacity=opacity,
    )
    t.move_to(b)
    t.shift(RIGHT * (b.get_x() - glyphs.get_x()))
    return VGroup(b, t), glyphs


def dim(group, k):
    """group 을 k 배 흐리게 (k=1 이면 원래대로). 각 부분의 원래 투명도 비율을 지킨다.
    (set_opacity 는 상자 안까지 꽉 칠해 버리고, 숨겨 둔 글자도 드러내므로 쓰지 않는다)"""
    anims = []
    for m in group.family_members_with_points():
        if "_base_op" not in vars(m):
            m._base_op = (m.get_fill_opacity(), m.get_stroke_opacity())
        f, st = m._base_op
        anims.append(m.animate.set_fill(opacity=f * k).set_stroke(opacity=st * k))
    return AnimationGroup(*anims)


def node(name, color, wd=2.7, ht=0.95, size=34):
    """주인공 상자 (Parameter, Probability, Loss, Gradient …)"""
    b = RoundedRectangle(corner_radius=0.16, width=wd, height=ht, stroke_color=color, stroke_width=4,
                         fill_color=color, fill_opacity=0.15)  # fmt: skip
    return VGroup(b, fit(en(name, size, color), wd - 0.3).move_to(b))


def ko_node(name, color, wd=3.4, ht=0.95, size=32):
    b = RoundedRectangle(corner_radius=0.16, width=wd, height=ht, stroke_color=color, stroke_width=4,
                         fill_color=color, fill_opacity=0.15)  # fmt: skip
    return VGroup(b, fit(ko(name, size, color), wd - 0.3).move_to(b))


def crack_line(center, color=C_ACCENT, size=1.0):
    pts = [np.array([center[0] + dx * size, center[1] + dy * size, 0])
           for dx, dy in [(-0.9, 0.1), (-0.5, -0.15), (-0.2, 0.15), (0.15, -0.1), (0.5, 0.15), (0.9, -0.05)]]  # fmt: skip
    return VMobject(stroke_color=color, stroke_width=6).set_points_as_corners(pts)


def egg(scale=1.0):
    """2D 알 (테두리 + 아주 옅은 채움)"""
    return Ellipse(width=2.0 * scale, height=2.6 * scale, stroke_color=C_TEXT, stroke_width=5,
                   fill_color=C_TEXT, fill_opacity=0.15)  # fmt: skip


def bird(scale=1.0, color=C_TEXT):
    """2D 새 (날개 두 개) + 아주 옅은 채움"""
    left = Arc(
        radius=0.5,
        start_angle=PI / 6,
        angle=2 * PI / 3,
        stroke_color=color,
        stroke_width=6,
    )
    right = left.copy().flip(UP).next_to(left, RIGHT, buff=-0.08)
    g = VGroup(left, right)
    g.set_fill(color, opacity=0.15)
    return g.scale(scale)


def loss_surface_fn(u, v):
    """3D 'Loss 언덕' (설명용 그릇 모양). 높이 0.3 ~ 2.9 안에 들어오게 (화면 밖으로 안 넘침)"""
    return (
        0.12 * (u - 0.4) ** 2
        + 0.2 * (v + 0.3) ** 2
        + 0.12 * math.sin(1.5 * u) * math.cos(1.2 * v)
        + 0.3
    )


def surface_grad(u, v):
    e = 1e-4
    gu = (loss_surface_fn(u + e, v) - loss_surface_fn(u - e, v)) / (2 * e)
    gv = (loss_surface_fn(u, v + e) - loss_surface_fn(u, v - e)) / (2 * e)
    return np.array([gu, gv])


# ── 장면 공통 ────────────────────────────────────────────────
class Base(ThreeDScene):
    """Vrew 문장 시각(CUES)에 맞춰 진행. 프레임 수로 시간을 세서 밀리지 않게 한다"""

    def setup(self):
        super().setup()
        self.f = 0  # 지금까지 만든 프레임 수
        self.key = None

    # 시간 관리 ------------------------------------------------
    def play(self, *anims, run_time=None, **kw):
        if run_time is None:  # Manim 내부 호출 (wait 등) — 프레임은 hold_until 이 센다
            super().play(*anims, **kw)
            return
        k = max(1, round(run_time * FPS))
        super().play(*anims, run_time=(k - 0.5) / FPS, **kw)  # 정확히 k 프레임
        self.f += k
        self.check_layout()

    # 겹침 검사 ------------------------------------------------
    def check_layout(self):
        """제목과 겹치거나 여백(안전 영역) 밖으로 나간 그림을 콘솔에 알려 준다 (같은 알림은 한 번만)"""
        if getattr(self, "in3d", False) or self.key is None:
            return
        seen = self.__dict__.setdefault("_warned", set())
        head = self.head
        for m in self.mobjects:
            if m is head or isinstance(m, ValueTracker) or not m.get_family():
                continue
            try:
                L, Rr = m.get_left()[0], m.get_right()[0]
                T, B = m.get_top()[1], m.get_bottom()[1]
            except Exception:
                continue
            if Rr - L < 1e-3 and T - B < 1e-3:
                continue
            name = (
                getattr(m, "text", None)
                or getattr(m, "original_text", None)
                or type(m).__name__
            )
            name = str(name)[:20]
            probs = []
            if head is not None:
                hL, hR, hT, hB = (
                    head.get_left()[0],
                    head.get_right()[0],
                    head.get_top()[1],
                    head.get_bottom()[1],
                )
                if L < hR + 0.05 and Rr > hL - 0.05 and T > hB - 0.05 and B < hT + 0.05:
                    probs.append("제목과 겹침")
            if T > SAFE_TOP + 0.02 or B < SAFE_BOTTOM - 0.02:
                probs.append("위·아래 여백 밖")
            if L < -SAFE_X - 0.05 or Rr > SAFE_X + 0.05:
                probs.append("좌우 여백 밖")
            for pb in probs:
                tag = (self.key, getattr(self, "cur_cue", 0), pb, name)
                if tag not in seen:
                    seen.add(tag)
                    print(f"[겹침] {self.key} 문장 {tag[1]}: '{name}' {pb}")

    def hold_until(self, frame):
        k = frame - self.f
        if k > 0:
            super().wait((k - 0.5) / FPS, frozen_frame=False)
            self.f += k

    def hold(self, sec):
        self.hold_until(self.f + round(sec * FPS))

    @staticmethod
    def fr(sec_abs):
        return round(sec_abs * FPS)

    def cue(self, i, lead=0.25):
        """이 장면 i번째 문장이 시작되기 lead 초 전까지 기다린다"""
        self.cur_cue = i
        target = self.fr(START[self.key] + CUES[self.key][i] - lead)
        if self.f > target:
            print(
                f"[알림] {self.key} 문장 {i}: 그림이 {(self.f - target) / FPS:.1f}초 늦음"
            )
        self.hold_until(target)

    def begin(self, key, title):
        if self.key is None:  # 장면 하나만 만들 때: 그 장면 시작 시각부터 센다
            self.f = self.fr(START[key])
        self.key = key
        self.head = None
        self.cur_cue = 0
        if title:
            h = fit(ko(title, size=TITLE_SIZE), SAFE_X - TITLE_X)
            h.move_to(
                [TITLE_X + h.width / 2, TITLE_Y, 0]
            )  # 처음부터 왼쪽 8% 자리에 고정
            self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.6)
            self.head = h

    def finish(self):
        end_f = self.fr(scene_end(self.key))
        fade = round(0.5 * FPS)
        if self.f > end_f - fade:
            print(
                f"[알림] {self.key} 장면이 {(self.f - end_f + fade) / FPS:.1f}초 길어요"
            )
        self.hold_until(end_f - fade)
        shown = [m for m in self.mobjects if not isinstance(m, ValueTracker)]
        if shown:
            self.play(
                *[FadeOut(m) for m in shown], run_time=max(end_f - self.f, 1) / FPS
            )
        self.hold_until(end_f)
        self.clear()

    # ════════════════════════════════════════════════════════
    # S00 오프닝
    # ════════════════════════════════════════════════════════
    def s00(self):
        self.begin("S00", "")
        title = ko("데미안 LLM", size=80).move_to(UP * 1.0)
        sub = en("Building a tiny language model", 44, C_DIM).next_to(
            title, DOWN, buff=0.35
        )
        self.play(Write(title), FadeIn(sub), run_time=2.0)

        self.cue(1)  # 거대 언어 모델
        rng = random.Random(1)
        nums = VGroup(*[mono(f"{rng.uniform(-1, 1):+.2f}", 26, C_DIM)
                        .move_to([rng.uniform(-6, 6), rng.uniform(-2.9, 2.9), 0]) for _ in range(70)])  # fmt: skip
        llm = VGroup(
            en("LLM", 96, C_ACCENT),
            ko("Large Language Model · 거대 언어 모델", 34, C_DIM),
        ).arrange(DOWN, buff=0.3)
        self.play(
            FadeOut(VGroup(title, sub)),
            LaggedStart(*[FadeIn(n) for n in nums], lag_ratio=0.02),
            run_time=1.8,
        )
        self.play(*[n.animate.move_to(ORIGIN).scale(0.2).set_opacity(0) for n in nums], FadeIn(llm, scale=0.6),
                  run_time=1.6)  # fmt: skip
        self.remove(nums)

        self.cue(2)  # 뼈대는 단순
        self.play(llm.animate.scale(0.55).move_to(UP * 2.2), run_time=0.8)

        self.cue(3)  # 데미안 단락, 단어 94개
        e = egg(0.9).move_to(LEFT * 3.2 + DOWN * 0.4)
        n94 = VGroup(mono(str(N_WORDS), 96, C_ACCENT), ko("단어", 36, C_DIM)).arrange(
            DOWN, buff=0.15
        )
        n94.move_to(RIGHT * 3.2 + DOWN * 0.4)
        self.play(FadeIn(e, scale=0.7), FadeIn(n94, shift=UP * 0.3), run_time=1.0)

        self.cue(4)  # 조각내고, 숫자로 바꾸고, 학습
        self.play(FadeOut(VGroup(e, n94)), run_time=0.5)
        names = [
            "원문",
            "토큰화",
            "어휘사전",
            "인코딩",
            "확률표",
            "학습",
            "생성",
            "대화",
        ]
        steps = VGroup(
            *[ko_node(n, C_WHOLE, wd=1.42, ht=0.85, size=28) for n in names]
        ).arrange(RIGHT, buff=0.22)
        fit(steps, 2 * SAFE_X).move_to(DOWN * 0.3)
        self.play(
            LaggedStart(*[FadeIn(s, shift=RIGHT * 0.2) for s in steps], lag_ratio=0.12),
            run_time=2.2,
        )
        self.play(
            steps[5][0].animate.set_fill(BLUE_C, 0.45),
            Circumscribe(steps[5], color=C_ACCENT),
            run_time=1.0,
        )

        self.cue(5)  # 스스로 문장
        self.play(
            steps[6][0].animate.set_fill(C_GOOD, 0.45),
            Indicate(steps[6], color=C_GOOD),
            run_time=1.0,
        )
        self.finish()

    # ════════════════════════════════════════════════════════
    # S01 원문
    # ════════════════════════════════════════════════════════
    def s01(self):
        self.begin("S01", "0. 원문")
        e = egg(1.1).move_to(DOWN * 0.3)
        self.play(FadeIn(e, scale=0.8), run_time=0.8)

        self.cue(1)  # GPT 는 엄청나게 많은 글
        pages = VGroup(*[Rectangle(width=0.55, height=0.75, stroke_color=C_DIM, stroke_width=1.5, fill_color=C_DIM,
                                   fill_opacity=0.1) for _ in range(60)]).arrange_in_grid(rows=5, cols=12, buff=0.12)  # fmt: skip
        pages.move_to(DOWN * 0.5)
        gpt = en("GPT", 60, C_DIM).next_to(pages, UP, buff=0.25)
        self.play(
            FadeOut(e),
            LaggedStart(*[FadeIn(p) for p in pages], lag_ratio=0.01),
            FadeIn(gpt),
            run_time=1.5,
        )

        self.cue(2)  # 우리 모델은 이 짧은 글 하나
        one = pages[0]
        self.play(
            FadeOut(VGroup(*pages[1:])),
            FadeOut(gpt),
            one.animate.move_to(DOWN * 0.3).scale(2.2),
            run_time=1.2,
        )

        self.cue(3)  # 509바이트 영어 글
        hl = {wd: C_ACCENT for wd, _ in SPLITS}
        lines = textwrap.wrap(d.TEXT, 50)
        body = VGroup(
            *[
                Text(ln, font=MONO_FONT, font_size=26, color=C_TEXT, t2c=hl)
                for ln in lines
            ]
        )
        body.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        fit(body, 2 * SAFE_X - 0.4, (TOP_Y - SAFE_BOTTOM) - 1.5)
        body.move_to([0, (TOP_Y + SAFE_BOTTOM) / 2 - 0.4, 0])
        self.play(FadeOut(one), run_time=0.5)  # 종이를 먼저 치우고 (겹치지 않게)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in body], lag_ratio=0.2),
            run_time=2.5,
        )

        self.cue(4)  # 509바이트 = 509자, 94단어
        stats = (
            VGroup(
                VGroup(
                    mono(f"{len(d.TEXT.encode())}", 48, C_ACCENT),
                    ko("바이트 = 글자", 26, C_DIM),
                ).arrange(DOWN, buff=0.06),
                VGroup(mono(f"{N_WORDS}", 48, C_ACCENT), ko("단어", 26, C_DIM)).arrange(
                    DOWN, buff=0.06
                ),
            )
            .arrange(RIGHT, buff=1.2)
            .move_to([0, TOP_Y - 0.35, 0])
        )
        self.play(
            body.animate.shift(DOWN * 0.45),
            FadeIn(stats, shift=DOWN * 0.2),
            run_time=1.0,
        )

        self.cue(5)  # 이 글이 세상의 전부
        frame = SurroundingRectangle(body, color=C_ACCENT, buff=0.18, stroke_width=4)
        self.play(Create(frame), run_time=1.0)

        self.cue(6)  # 없는 단어는 말할 수 없음
        outside = ko("?", 90, C_POS).move_to(
            [min(frame.get_right()[0] + 0.5, SAFE_X - 0.3), 0, 0]
        )
        self.play(FadeIn(outside, scale=1.5), run_time=0.6)
        self.play(Indicate(outside, color=C_POS), run_time=0.8)

        self.cue(7)  # 모델이 읽을 수 있는 모양으로
        self.play(FadeOut(outside), frame.animate.set_stroke(C_WHOLE), run_time=0.8)
        self.finish()

    # ════════════════════════════════════════════════════════
    # S02 토큰화
    # ════════════════════════════════════════════════════════
    def s02(self):
        self.begin("S02", "1. 토큰화")
        k, dots = 0, 0
        while k < N_TOKENS and dots < 2:
            dots += d.tokens[k] == "."
            k += 1
        toks, roles = d.tokens[:k], d.TOK_ROLE[:k]
        sent = fit(
            Text(d.join_text(toks), font=MONO_FONT, font_size=36, color=C_TEXT),
            2 * SAFE_X,
        ).move_to(UP * 0.8)
        q = ko("?", 80, C_DIM).next_to(sent, DOWN, buff=0.4)
        self.play(FadeIn(sent), FadeIn(q), run_time=0.8)

        self.cue(1)  # token 조각으로 자름
        boxes = [token_box(t, r, 34) for t, r in zip(toks, roles)]
        grid = flow(boxes, top=1.9)
        n_glyph = sum(len(w(t)) for t in toks)
        self.play(FadeOut(q), run_time=0.3)
        if len(sent.submobjects) == n_glyph:
            i, moves = 0, []
            for t, b in zip(toks, boxes):
                n = len(w(t))
                moves.append(
                    ReplacementTransform(VGroup(*sent.submobjects[i : i + n]), b)
                )
                i += n
            self.play(LaggedStart(*moves, lag_ratio=0.1), run_time=2.8)
            self.remove(sent)
            self.add(*boxes)
        else:
            self.play(
                FadeOut(sent),
                LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.06),
                run_time=2.0,
            )

        self.cue(2)  # GPT-4o Tokenizer
        tk = VGroup(
            en("GPT-4o Tokenizer", 44, C_ACCENT), mono(f"({d.ENC_NAME})", 28, C_DIM)
        ).arrange(DOWN, buff=0.1)
        tk.move_to(DOWN * 1.6)
        self.play(FadeIn(tk, shift=UP * 0.2), run_time=0.8)

        self.cue(3)  # BPE
        self.play(FadeOut(tk), dim(grid, 0.3), run_time=0.6)
        bpe = en("BPE", 72, C_PIECE).move_to(
            DOWN * (0.6 + 0.4)
        )  # 5% 더 아래 (위 token 줄과 떨어지게)
        self.play(Write(bpe), run_time=0.8)

        self.cue(4)  # 두 조각 합치기 수십만 번
        stages = [
            ["·", "r", "e", "b", "o", "r", "n"],
            ["·r", "e", "b", "or", "n"],
            ["·re", "b", "orn"],
            ["·reb", "orn"],
        ]
        last = [p for wd, p in SPLITS if wd.lower().endswith("reborn")]
        if last:
            stages[-1] = list(last[0])

        def row(parts):
            return (
                VGroup(*[token_box(p, "FIRST", 40) for p in parts])
                .arrange(RIGHT, buff=0.14)
                .move_to(DOWN * 1.4)
            )

        r = row(stages[0])
        self.play(
            bpe.animate.scale(0.6).move_to(LEFT * 4.8 + DOWN * 0.3),
            FadeIn(r),
            run_time=0.8,
        )
        for st in stages[1:]:
            self.play(Transform(r, row(st)), run_time=1.0)
            self.hold(0.4)

        self.cue(5)  # 자주 쓰는 단어 = token 1개
        self.play(FadeOut(r), FadeOut(bpe), dim(grid, 1), run_time=0.6)
        whole = [
            b
            for b, t, rl in zip(boxes, toks, roles)
            if rl == "WHOLE" and w(t).isalpha()
        ][:4]
        self.play(
            LaggedStart(*[Indicate(b, color=C_WHOLE) for b in whole], lag_ratio=0.2),
            run_time=1.6,
        )

        self.cue(6)  # 드문 단어는 여러 조각
        self.play(dim(grid, 0.3), run_time=0.4)

        self.cue(7)  # Abraxas → Ab | rax | as
        split_grp = VGroup()
        if SPLITS:
            wd, parts = SPLITS[0]
            src = mono(wd, 56).move_to(LEFT * 3.4 + DOWN * 1.3)
            pieces = (
                VGroup(*[token_box(p, "FIRST", 48) for p in parts])
                .arrange(RIGHT, buff=0.15)
                .move_to(RIGHT * 2.4 + DOWN * 1.3)
            )
            arr = Arrow(
                src.get_right(),
                pieces.get_left(),
                buff=0.25,
                stroke_width=5,
                color=C_DIM,
            )
            self.play(FadeIn(src), run_time=0.5)
            self.play(GrowArrow(arr), TransformFromCopy(src, pieces), run_time=1.2)
            split_grp = VGroup(src, pieces, arr)

        self.cue(8)  # 띄어쓰기도 token
        self.play(FadeOut(split_grp), dim(grid, 1), run_time=0.5)
        spaced = [b for b, t in zip(boxes, toks) if t.startswith("·")][:5]
        self.play(
            LaggedStart(
                *[Circumscribe(b, color=C_ACCENT) for b in spaced], lag_ratio=0.15
            ),
            run_time=1.6,
        )

        self.cue(9)  # The ≠ ·the
        a, b = token_box("The", size=48), token_box("·the", size=48)
        ida = mono(str(d.REAL_ID.get("The", "")), 30, C_DIM)
        idb = mono(str(d.REAL_ID.get("·the", "")), 30, C_DIM)
        neq = tex(r"\neq", "≠", 80, C_POS)
        pair = VGroup(
            VGroup(a, ida).arrange(DOWN, buff=0.15),
            neq,
            VGroup(b, idb).arrange(DOWN, buff=0.15),
        )
        pair.arrange(RIGHT, buff=0.8).move_to(DOWN * 1.6)
        self.play(dim(grid, 0.3), FadeIn(pair), run_time=0.8)

        self.cue(10)  # 94 단어 → 114 token
        self.play(FadeOut(pair), dim(grid, 1), run_time=0.5)
        cnt = VGroup(mono(str(N_WORDS), 72, C_TEXT), ko("단어", 30, C_DIM)).arrange(
            DOWN, buff=0.08
        )
        cnt2 = VGroup(
            mono(str(N_TOKENS), 72, C_ACCENT), en("token", 34, C_DIM)
        ).arrange(DOWN, buff=0.08)
        arrow = Arrow(LEFT, RIGHT, stroke_width=6, color=C_DIM)
        VGroup(cnt, arrow, cnt2).arrange(RIGHT, buff=0.6).move_to(DOWN * 1.8)
        self.play(FadeIn(cnt), run_time=0.5)
        self.play(GrowArrow(arrow), FadeIn(cnt2, shift=LEFT * 0.3), run_time=0.8)
        self.finish()

    # ════════════════════════════════════════════════════════
    # S03 어휘사전
    # ════════════════════════════════════════════════════════
    def s03(self):
        self.begin("S03", "2. 어휘사전")
        allb = VGroup(*[token_box(t, r, 20) for t, r in zip(d.tokens, d.TOK_ROLE)])
        flow(list(allb), gap=0.08, row_gap=0.12, top=TOP_Y)
        fit(allb, None, TOP_Y - SAFE_BOTTOM - 0.2)
        allb.align_to(UP * TOP_Y, UP)
        self.play(
            LaggedStart(*[FadeIn(b) for b in allb], lag_ratio=0.005), run_time=1.5
        )

        self.cue(1)  # 마침표 12번, the 7번
        top = d.freq.most_common(2)
        hits = VGroup()
        for (tok, _), col in zip(top, [C_ACCENT, C_WHOLE]):
            for b, t in zip(allb, d.tokens):
                if t == tok:
                    hits.add(
                        SurroundingRectangle(b, color=col, buff=0.03, stroke_width=3)
                    )
        self.play(LaggedStart(*[Create(h) for h in hits], lag_ratio=0.03), run_time=1.6)
        labels = VGroup(*[VGroup(token_box(tok, size=36, color=col), mono(f"× {c}", 44, col)).arrange(RIGHT, buff=0.25)
                          for (tok, c), col in zip(top, [C_ACCENT, C_WHOLE])]).arrange(RIGHT, buff=1.2)  # fmt: skip
        bgl = Rectangle(
            width=labels.width + 0.6,
            height=labels.height + 0.4,
            stroke_width=0,
            fill_color=C_BG,
            fill_opacity=0.92,
        )
        lab = VGroup(bgl, labels).move_to(DOWN * 0.2)
        labels.move_to(bgl)
        self.play(FadeIn(lab, scale=0.9), run_time=0.6)

        self.cue(2)  # 서로 다른 token 58개
        self.play(FadeOut(lab), FadeOut(hits), run_time=0.4)
        cells = [
            token_box(d.id_to_tok[i], d.ROLE_OF.get(d.id_to_tok[i], "WHOLE"), 24)
            for i in range(V)
        ]
        VTOP = (
            TOP_Y - 0.55
        )  # 번호 줄 + 테두리가 들어갈 자리를 위에 남김 (제목과 안 겹치게)
        vocab = flow(
            cells, gap=0.12, row_gap=0.42, top=VTOP
        )  # 줄 사이를 넓혀 번호가 들어갈 틈
        fit(vocab, None, VTOP - (SAFE_BOTTOM + 1.2))
        vocab.align_to(UP * VTOP, UP)
        # 번호(0~57)는 미리 자리를 잡아 두고, 나중(문장 6)에 보여 줌
        nums = VGroup(
            *[
                mono(str(i), 18, C_ACCENT).next_to(c, UP, buff=0.04).align_to(c, LEFT)
                for i, c in enumerate(cells)
            ]
        )
        self.play(
            FadeOut(allb),
            LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.015),
            run_time=1.8,
        )
        big = VGroup(
            mono(str(V), 80, C_ACCENT), ko("서로 다른 token", 30, C_DIM)
        ).arrange(RIGHT, buff=0.3)
        big.move_to([0, SAFE_BOTTOM + 0.55, 0])
        self.play(FadeIn(big, shift=UP * 0.2), run_time=0.6)

        self.cue(3)  # 어휘사전
        vtitle = (
            VGroup(en("Vocabulary", 44, C_ACCENT), ko("= 어휘사전", 34, C_DIM))
            .arrange(RIGHT, buff=0.3)
            .move_to(big)
        )
        vbox = SurroundingRectangle(
            VGroup(vocab, nums), color=C_ACCENT, buff=0.15, fill_opacity=0
        )  # 번호까지 감싸는 테두리
        self.play(FadeOut(big), FadeIn(vtitle), Create(vbox), run_time=0.9)

        self.cue(4)  # 확률표의 가로줄과 세로줄
        dimmed = VGroup(vocab, vbox, vtitle)
        self.play(
            FadeOut(dimmed), run_time=0.5
        )  # 겹치지 않게 어휘사전은 잠시 치움 (번호 붙일 때 다시 나옴)
        G = 4.4  # 58×58 칸 격자
        gx, gy = 0.6, CONTENT_Y - 0.1
        gL, gT = gx - G / 2, gy + G / 2
        grid = VGroup(
            Square(G, stroke_color=C_WHOLE, stroke_width=3).move_to([gx, gy, 0])
        )
        for i in range(1, V):
            grid.add(
                Line(
                    [gL, gT - G * i / V, 0],
                    [gL + G, gT - G * i / V, 0],
                    stroke_width=0.8,
                    stroke_color=C_DIM,
                    stroke_opacity=0.45,
                )
            )
            grid.add(
                Line(
                    [gL + G * i / V, gT, 0],
                    [gL + G * i / V, gT - G, 0],
                    stroke_width=0.8,
                    stroke_color=C_DIM,
                    stroke_opacity=0.45,
                )
            )
        rl = ko(f"가로줄 {V}", 34, C_WHOLE).next_to(grid, LEFT, buff=0.35)
        cl = ko(f"세로줄 {V}", 34, C_WHOLE).next_to(grid, UP, buff=0.2)
        self.play(Create(grid), run_time=1.2)
        rband = Rectangle(
            width=G,
            height=G / V * 1.2,
            stroke_width=0,
            fill_color=C_WHOLE,
            fill_opacity=0.6,
        ).move_to([gx, gT - G * 4.5 / V, 0])
        cband = Rectangle(
            width=G / V * 1.2,
            height=G,
            stroke_width=0,
            fill_color=C_WHOLE,
            fill_opacity=0.6,
        ).move_to([gL + G * 39.5 / V, gy, 0])
        self.play(FadeIn(rl), FadeIn(rband), run_time=0.6)
        self.play(FadeIn(cl), FadeIn(cband), run_time=0.6)
        grid.add(rband, cband)

        self.cue(5)  # GPT 어휘 약 20만 개
        self.play(FadeOut(VGroup(grid, rl, cl)), run_time=0.4)
        cmp = (
            VGroup(
                VGroup(en("GPT-4o", 40, C_DIM), mono("≈ 200,000", 56, C_TEXT)).arrange(
                    DOWN, buff=0.15
                ),
                VGroup(
                    ko("우리 모델", 34, C_ACCENT), mono(str(V), 56, C_ACCENT)
                ).arrange(DOWN, buff=0.15),
            )
            .arrange(RIGHT, buff=2.0)
            .move_to([0, CONTENT_Y, 0])
        )
        self.play(FadeIn(cmp), run_time=0.8)

        self.cue(6)  # 0번부터 57번까지 번호
        vbox.set_fill(opacity=0)  # 상자 안이 노랗게 칠해지지 않도록 다시 한 번 비움
        self.play(FadeOut(cmp), FadeIn(dimmed), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(n) for n in nums], lag_ratio=0.02), run_time=2.0)

        self.cue(7)  # 번호는 다음 단계에서
        self.play(
            Indicate(nums[0], scale_factor=1.6),
            Indicate(nums[-1], scale_factor=1.6),
            run_time=1.0,
        )
        self.finish()

    # ════════════════════════════════════════════════════════
    # S04 인코딩
    # ════════════════════════════════════════════════════════
    def s04(self):
        self.begin("S04", "3. 인코딩")
        n = 12
        toks = d.tokens[:n]
        boxes = VGroup(
            *[token_box(t, r, 34) for t, r in zip(toks, d.TOK_ROLE[:n])]
        ).arrange(RIGHT, buff=0.14)
        fit(boxes, 2 * SAFE_X).move_to(UP * 1.3)
        self.play(
            LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.05), run_time=1.0
        )
        nums = VGroup()
        for b, t in zip(boxes, toks):
            nb = VGroup(b[0].copy(), mono(str(d.vocab[t]), 36, C_ACCENT).move_to(b[0]))
            nums.add(nb)
        nums.next_to(boxes, DOWN, buff=0.8)
        arrows = VGroup(*[Arrow(b.get_bottom(), nb.get_top(), buff=0.06, stroke_width=3, color=C_DIM,
                                max_tip_length_to_length_ratio=0.3) for b, nb in zip(boxes, nums)])  # fmt: skip

        self.cue(1)  # 각 조각 → 번호
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), TransformFromCopy(b, nb)) for a, b, nb in zip(arrows, boxes, nums)],
                              lag_ratio=0.12), run_time=3.0)  # fmt: skip

        self.cue(2)  # 같은 token = 같은 번호
        cnt = {}
        for t in toks:
            cnt[t] = cnt.get(t, 0) + 1
        rep = [
            t
            for t, c in sorted(cnt.items(), key=lambda kv: -kv[1])
            if c > 1 and w(t).isalpha()
        ][:2]
        marks = VGroup()
        for t, col in zip(rep, [C_ACCENT, C_WHOLE]):
            for i, tt in enumerate(toks):
                if tt == t:
                    marks.add(
                        SurroundingRectangle(
                            VGroup(boxes[i], nums[i]),
                            color=col,
                            buff=0.06,
                            stroke_width=4,
                        )
                    )
        self.play(
            LaggedStart(*[Create(m) for m in marks], lag_ratio=0.15), run_time=1.2
        )

        self.cue(3)  # 숫자 114개 배열
        self.play(
            FadeOut(marks),
            FadeOut(arrows),
            FadeOut(boxes),
            nums.animate.move_to(UP * 1.7),
            run_time=0.8,
        )
        seq = " ".join(str(i) for i in d.ids)
        body = VGroup(
            *[mono(line, 30, C_ACCENT) for line in textwrap.wrap(seq, 56)]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        fit(body, 2 * SAFE_X, 3.4).move_to(DOWN * 0.9)
        self.play(
            LaggedStart(*[FadeIn(b, shift=LEFT * 0.3) for b in body], lag_ratio=0.15),
            run_time=1.6,
        )

        self.cue(4)  # 모델이 보는 것은 숫자뿐
        self.play(FadeOut(nums), body.animate.move_to(UP * 0.2), run_time=0.8)
        self.play(Circumscribe(body, color=C_ACCENT), run_time=1.2)
        self.finish()

    # ════════════════════════════════════════════════════════
    # S05 확률표 — 주인공: Probability, Matrix
    # ════════════════════════════════════════════════════════
    def s05(self):
        self.begin("S05", "4. 확률표")
        cur = token_box(EX_TOK, size=56).move_to(LEFT * 2.0 + UP * 0.2)
        qm = ko("?", 110, C_ACCENT).next_to(cur, RIGHT, buff=1.2)
        arr = Arrow(
            cur.get_right(), qm.get_left(), buff=0.15, stroke_width=6, color=C_DIM
        )
        self.play(FadeIn(cur), GrowArrow(arr), FadeIn(qm, scale=1.4), run_time=1.2)

        self.cue(1)  # 직접 세기
        self.play(FadeOut(VGroup(cur, arr, qm)), run_time=0.5)

        self.cue(2)  # 이웃한 두 token 쌍 113
        n = 11
        cells = fit(
            VGroup(*[mono(str(i), 44, C_ACCENT) for i in d.ids[:n]]).arrange(
                RIGHT, buff=0.55
            ),
            2 * SAFE_X,
        ).move_to(UP * 0.9)
        win = SurroundingRectangle(
            VGroup(cells[0], cells[1]), color=C_WHOLE, buff=0.14, stroke_width=5
        )
        cv = ValueTracker(1)
        counter = always_redraw(lambda: VGroup(mono(f"{int(cv.get_value())}", 72, C_WHOLE), ko("쌍", 34, C_DIM))
                                .arrange(RIGHT, buff=0.15).move_to(DOWN * 1.4))  # fmt: skip
        self.play(FadeIn(cells), Create(win), run_time=0.8)
        self.add(counter)
        slides = [Transform(win, SurroundingRectangle(VGroup(cells[i], cells[i + 1]), color=C_WHOLE, buff=0.14, stroke_width=5))
                  for i in range(1, n - 1)]  # fmt: skip
        self.play(
            Succession(*slides), cv.animate.set_value(N), run_time=3.2, rate_func=linear
        )

        self.cue(3)  # the 는 7번
        self.remove(counter)
        self.play(FadeOut(VGroup(cells, win)), run_time=0.4)
        srcs = VGroup(
            *[token_box(EX_TOK, size=36) for _ in range(d.freq[EX_TOK])]
        ).arrange(DOWN, buff=0.12)
        fit(srcs, None, TOP_Y - SAFE_BOTTOM - 0.5).move_to(LEFT * 5.0 + DOWN * 0.25)
        self.play(
            LaggedStart(*[FadeIn(s, shift=RIGHT * 0.2) for s in srcs], lag_ratio=0.12),
            run_time=1.2,
        )
        seven = mono(f"× {d.freq[EX_TOK]}", 56, C_TEXT).next_to(srcs, RIGHT, buff=0.3)
        self.play(FadeIn(seven), run_time=0.4)

        self.cue(4)  # 다음에 egg·world 2번씩 …
        cnt = d.counts[EX]
        total = sum(cnt.values())
        items = cnt.most_common()
        # 다음 단어 상자들: 오른쪽 끝을 x=1.6 에 맞춰 세움 → 막대가 모두 같은 곳에서 시작
        outs = VGroup(*[token_box(d.id_to_tok[j], size=36) for j, _ in items]).arrange(
            DOWN, buff=0.35, aligned_edge=RIGHT
        )
        outs.move_to([0, CONTENT_Y, 0])
        outs.shift(RIGHT * (1.6 - outs.get_right()[0]))
        tallies = VGroup(
            *[
                mono("|" * c, 44, C_DIM).next_to(o, RIGHT, buff=0.35)
                for o, (_, c) in zip(outs, items)
            ]
        )
        self.play(FadeOut(seven), FadeIn(outs), run_time=0.6)
        links, k = [], 0
        for o_i, (_, c) in enumerate(items):
            for _ in range(c):
                links.append(Arrow(srcs[k].get_right(), outs[o_i].get_left(), buff=0.1, stroke_width=3, color=C_DIM,
                                   max_tip_length_to_length_ratio=0.05))  # fmt: skip
                k += 1
        self.play(LaggedStart(*[GrowArrow(a) for a in links], lag_ratio=0.25),
                  LaggedStart(*[FadeIn(t) for t in tallies], lag_ratio=0.3), run_time=3.0)  # fmt: skip

        self.cue(5)  # 횟수를 비율로 = Probability
        self.play(FadeOut(VGroup(*links)), FadeOut(srcs), run_time=0.5)
        ptitle = VGroup(
            en("Probability", 64, C_GOOD, max_w=4.6),
            ko("확률 = 횟수 ÷ 전체", 34, C_TEXT),
        ).arrange(DOWN, buff=0.15)
        ptitle.move_to(LEFT * 4.0 + UP * 1.0)
        self.play(Write(ptitle[0]), FadeIn(ptitle[1]), run_time=1.2)

        self.cue(6)  # egg 2/7 ≈ 29%
        pe = cnt[EX_NEXT] / total
        frac = VGroup(
            mono(f"P({w(NEXT_TOK)}|{w(EX_TOK)})", NUM_SIZE, C_ACCENT),
            mono(f"= {cnt[EX_NEXT]}/{total} ≈ {pe:.0%}", NUM_SIZE, C_ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)  # 두 줄 (글자 크기 그대로)
        fit(frac, 4.6).move_to(
            [-3.9, -0.85, 0]
        )  # 왼쪽 칸 (x ≤ -1.6): 오른쪽 단어 상자와 안 겹침
        self.play(Write(frac), Indicate(outs[0], color=C_ACCENT), run_time=1.4)

        self.cue(7)  # 합 = 100%
        probs = [c / total for _, c in items]
        pb = VGroup()
        for i, (p, o) in enumerate(zip(probs, outs)):
            b = Rectangle(
                width=max(p * 9.0, 0.05),
                height=0.5,
                stroke_width=0,
                fill_color=C_ACCENT if i == 0 else C_DIM,
                fill_opacity=0.9,
            )
            pb.add(b.next_to(o, RIGHT, buff=0.35))
        plab = VGroup(
            *[
                mono(f"{p:.0%}", 32, C_TEXT).next_to(b, RIGHT, buff=0.15)
                for p, b in zip(probs, pb)
            ]
        )
        self.play(
            *[ReplacementTransform(t, b) for t, b in zip(tallies, pb)],
            FadeIn(plab),
            run_time=1.2,
        )
        # 확률을 한 줄로 이어 붙이면 = 100% (왼쪽 아래 칸에만)
        stack = VGroup(*[Rectangle(width=max(p * 3.6, 0.05), height=0.5, stroke_width=0, fill_color=C_ACCENT if i == 0 else C_DIM,
                                   fill_opacity=0.9) for i, p in enumerate(probs)]).arrange(RIGHT, buff=0.04)  # fmt: skip
        stack.move_to([-6.2 + stack.width / 2, -2.3, 0])
        sum100 = mono("= 100%", 40, C_GOOD).next_to(stack, RIGHT, buff=0.25)
        self.play(TransformFromCopy(pb, stack), FadeIn(sum100), run_time=1.4)

        self.cue(8)  # 58×58 = 3,364칸 표
        img = prob_image(d.COUNT_P, height=4.9).move_to([-2.2, CONTENT_Y, 0])
        fr = frame_of(img)
        row = cell_rect(img, row=EX, color=C_WHOLE)
        self.play(
            FadeOut(VGroup(ptitle, frac, outs, plab, pb, sum100)),
            ReplacementTransform(stack, row),
            run_time=0.9,
        )
        self.play(FadeIn(img), Create(fr), run_time=1.4)
        RCX = 3.45  # 오른쪽 칸 가운데
        size = VGroup(
            mono(f"{V} × {V} = {V * V:,}", NUM_SIZE, C_TEXT), ko("칸", 30, C_DIM)
        ).arrange(RIGHT, buff=0.2)
        size.move_to([RCX, 1.6, 0])
        self.play(Write(size), run_time=1.0)

        self.cue(9)  # 가로줄 = 현재, 세로줄 = 다음
        col = cell_rect(img, col=EX_NEXT, color=C_ACCENT)
        ar = Arrow(
            img.get_corner(UP + LEFT) + LEFT * 0.5,
            img.get_corner(DOWN + LEFT) + LEFT * 0.5,
            buff=0,
            stroke_width=5,
            color=C_WHOLE,
        )
        arl = ko("현재", 30, C_WHOLE).next_to(
            ar, DOWN, buff=0.1
        )  # 화살표 아래 (왼쪽 여백 밖으로 잘리지 않게)
        ac = Arrow(
            img.get_corner(UP + LEFT) + UP * 0.42,
            img.get_corner(UP + RIGHT) + UP * 0.42,
            buff=0,
            stroke_width=5,
            color=C_ACCENT,
        )
        acl = ko("다음", 30, C_ACCENT).next_to(
            ac, RIGHT, buff=0.12
        )  # 제목과 겹치지 않게 오른쪽
        rtag = band_label(row, w(EX_TOK), C_WHOLE, "row")
        ctag = band_label(col, w(NEXT_TOK), C_ACCENT, "col")
        self.play(GrowArrow(ar), FadeIn(arl), FadeIn(rtag), run_time=0.8)
        self.play(GrowArrow(ac), FadeIn(acl), FadeIn(col), FadeIn(ctag), run_time=0.8)
        hit = cell_rect(img, row=EX, col=EX_NEXT, color=C_ACCENT, sw=5)
        zoom = VGroup(token_box(EX_TOK, size=32), mono("→", 34, C_DIM), token_box(NEXT_TOK, size=32),
                      mono(f"= {pe:.0%}", NUM_SIZE, C_ACCENT)).arrange(RIGHT, buff=0.2).move_to([RCX, 0.4, 0])  # fmt: skip
        link = DashedLine(hit.get_right(), zoom.get_left(), color=C_ACCENT)
        self.play(
            Create(hit),
            Create(link),
            FadeIn(zoom),
            Flash(hit.get_center(), color=C_ACCENT),
            run_time=1.2,
        )

        self.cue(10)  # 0 아닌 칸 95개 3%
        mask = d.COUNT_P > 0
        lit = prob_image(d.COUNT_P, height=img.height, only=mask).move_to(img)
        nz = VGroup(mono(f"{NONZERO}", NUM_SIZE, C_ACCENT), mono(f"/ {V * V:,}", NUM_SIZE, C_DIM),
                    mono(f"≈ {NONZERO / (V * V):.0%}", NUM_SIZE, C_ACCENT)).arrange(RIGHT, buff=0.2).move_to([RCX, -0.8, 0])  # fmt: skip
        self.play(
            FadeOut(VGroup(row, col, hit, link, rtag, ctag)),
            FadeIn(lit),
            FadeIn(nz),
            run_time=1.0,
        )
        self.play(FadeOut(lit), run_time=0.6)
        self.play(FadeIn(lit), run_time=0.6)

        self.cue(11)  # 나머지 칸은 0
        zero = ko("나머지 = 0", 36, C_DIM).next_to(nz, DOWN, buff=0.35)
        self.play(FadeOut(lit), FadeIn(zero), run_time=0.8)

        self.cue(12)  # Probability Matrix = 확률표
        pm = fit(en("Probability Matrix", 56, C_GOOD), 5.4).move_to([RCX, 1.6, 0])
        self.play(
            FadeOut(VGroup(size, zoom)), run_time=0.4
        )  # 먼저 치우고 (겹치지 않게)
        self.play(Write(pm), fr[:2].animate.set_stroke(C_GOOD), run_time=1.2)

        self.cue(13)  # 세어서 만든 정답표
        st = ko("정답표", 52, C_POS)
        stamp = VGroup(
            st, SurroundingRectangle(st, color=C_POS, buff=0.18, stroke_width=6)
        ).rotate(-0.15)
        stamp.move_to([RCX, -2.35, 0])
        self.play(FadeOut(VGroup(nz, zero)), FadeIn(stamp, scale=1.8), run_time=0.7)

        self.cue(14)  # 진짜 LLM 은 세지 않음
        no = (
            VGroup(ko("세기", 40, C_DIM), ko("✗", 54, C_POS))
            .arrange(RIGHT, buff=0.25)
            .move_to([RCX, -0.05, 0])
        )
        self.play(FadeIn(no), run_time=0.6)

        self.cue(15)  # Parameter 를 고쳐 가며 스스로
        par = node("Parameter", C_PARAM, wd=3.4).move_to([RCX, -0.05, 0])
        target = cell_center(img, EX, EX_NEXT)
        pcell = cell_rect(
            img, row=EX, col=EX_NEXT, color=C_PARAM, sw=5
        )  # Matrix 안의 그 칸
        a2 = CurvedArrow(
            par.get_left() + LEFT * 0.05,
            target + RIGHT * 0.12 + UP * 0.05,
            angle=0.5,
            stroke_width=6,
            color=C_PARAM,
        )
        self.play(ReplacementTransform(no, par), run_time=0.7)
        self.play(Create(a2), run_time=0.7)
        self.play(
            FadeIn(pcell, scale=2.5),
            Flash(target, color=C_PARAM, line_length=0.25),
            run_time=0.6,
        )

        self.cue(16)  # 이것이 학습
        learn = ko("= 학습", 44, C_ACCENT).next_to(par, DOWN, buff=0.35)
        self.play(
            FadeIn(learn, shift=UP * 0.2),
            Indicate(par, color=C_ACCENT),
            Indicate(pcell, color=C_ACCENT, scale_factor=1.6),
            run_time=0.9,
        )
        self.finish()

    # ════════════════════════════════════════════════════════
    # S06 초기화 — 주인공: Weight, Bias, Parameter
    # ════════════════════════════════════════════════════════
    def s06(self):
        self.begin("S06", "5-1. 학습(초기화)")
        wn = node("Weight", C_W, wd=3.2, size=44).move_to(LEFT * 2.2 + UP * 0.4)
        bn = node("Bias", C_B, wd=3.2, size=44).move_to(RIGHT * 2.2 + UP * 0.4)
        self.play(
            FadeIn(wn, shift=RIGHT * 0.3), FadeIn(bn, shift=LEFT * 0.3), run_time=1.0
        )

        self.cue(1)  # Weight 는 58×58 Matrix
        H = 4.4  # 위 '58×58' 글자가 제목과 겹치지 않게
        wimg = signed_image(W0, height=H, scale=1.5, pos=C_W, neg=C_NEG).move_to(
            [-2.9, CONTENT_Y + 0.3, 0]
        )
        wfr = frame_of(wimg, C_W, 3)
        wl = en("W", 64, C_W).next_to(wimg, LEFT, buff=0.5)
        dims = mono(f"{V} × {V}", NUM_SIZE, C_W).next_to(wimg, UP, buff=0.12)
        self.play(
            ReplacementTransform(wn, wl),
            FadeOut(bn),
            FadeIn(wimg),
            Create(wfr),
            run_time=1.4,
        )
        self.play(Write(dims), run_time=0.7)

        self.cue(2)  # 가로줄 현재, 세로줄 다음
        ar = Arrow(wimg.get_corner(UP + RIGHT) + RIGHT * 0.45, wimg.get_corner(DOWN + RIGHT) + RIGHT * 0.45, buff=0, stroke_width=5,
                   color=C_WHOLE)  # fmt: skip
        al = ko("현재 단어", 28, C_WHOLE).next_to(ar, RIGHT, buff=0.12)
        self.play(GrowArrow(ar), FadeIn(al), run_time=0.8)
        ac = Arrow(wimg.get_corner(DOWN + LEFT) + DOWN * 0.42, wimg.get_corner(DOWN + RIGHT) + DOWN * 0.42, buff=0, stroke_width=5,
                   color=C_ACCENT)  # fmt: skip
        acl = ko("다음 단어", 28, C_ACCENT).next_to(ac, DOWN, buff=0.08)
        self.play(GrowArrow(ac), FadeIn(acl), run_time=0.8)

        self.cue(3)  # the 가로줄과 egg 세로줄이 만나는 칸
        rr = cell_rect(wimg, row=EX, color=C_WHOLE)
        cc = cell_rect(wimg, col=EX_NEXT, color=C_ACCENT)
        rlab = band_label(rr, w(EX_TOK), C_WHOLE, "row")
        clab = band_label(cc, w(NEXT_TOK), C_ACCENT, "col")
        self.play(FadeOut(dims), FadeIn(rr), FadeIn(rlab), run_time=0.7)
        self.play(FadeIn(cc), FadeIn(clab), run_time=0.7)

        self.cue(4)  # 그 칸 = the→egg 의 Weight
        hit = cell_rect(wimg, row=EX, col=EX_NEXT, color=C_ACCENT, sw=5)
        sq = Square(
            1.4,
            stroke_color=C_ACCENT,
            stroke_width=5,
            fill_color=C_W,
            fill_opacity=0.35,
        ).move_to(RIGHT * 3.9 + UP * 1.2)
        val = mono(f"{W0[EX, EX_NEXT]:+.2f}", 40, C_TEXT).move_to(sq)
        cap = VGroup(
            en("Weight", 36, C_W),
            token_box(EX_TOK, size=28),
            mono("→", 28, C_DIM),
            token_box(NEXT_TOK, size=28),
        )
        cap.arrange(RIGHT, buff=0.15).next_to(sq, DOWN, buff=0.3)
        self.play(
            FadeOut(VGroup(ar, al)),
            Create(hit),
            Flash(hit.get_center(), color=C_ACCENT),
            run_time=0.7,
        )
        self.play(TransformFromCopy(hit, sq), FadeIn(val), FadeIn(cap), run_time=1.0)

        self.cue(5)  # Bias = 아래 한 줄 58개
        self.play(
            FadeOut(VGroup(rr, cc, rlab, clab, hit, sq, val, cap, ac, acl)),
            run_time=0.5,
        )
        bimg = signed_image(
            B0[None, :] + 1e-9,
            height=H / V * 2.2,
            scale=1.5,
            pos=C_B,
            neg=C_NEG,
            width=wimg.width,
        )
        bimg.next_to(wimg, DOWN, buff=0.42)
        bfr = frame_of(bimg, C_B, 3)
        bl = en("b", 64, C_B).next_to(bimg, LEFT, buff=0.5)
        bcnt = mono(f"{V}", 40, C_B).next_to(bimg, RIGHT, buff=0.45)
        self.play(
            FadeIn(bimg, shift=UP * 0.3),
            Create(bfr),
            FadeIn(bl),
            FadeIn(bcnt),
            run_time=1.2,
        )
        self.play(Indicate(bfr, color=C_B), run_time=0.8)

        self.cue(6)  # 3,364 + 58 = 3,422 = Parameter
        eq = fit(mono(f"{V * V:,} + {V} = {N_PARAMS:,}", NUM_SIZE, C_TEXT), 5.0)
        eq.move_to(RIGHT * 3.9 + UP * 1.7)
        self.play(Write(eq), run_time=1.4)
        brace = Brace(VGroup(wfr, bfr), RIGHT, buff=0.15, color=C_PARAM)
        pn = node("Parameter", C_PARAM, wd=3.6, size=44).move_to(
            RIGHT * 3.9 + DOWN * 0.1
        )
        pcount = mono(f"{N_PARAMS:,}", 56, C_PARAM).next_to(pn, DOWN, buff=0.25)
        self.play(
            FadeOut(bcnt), Create(brace), FadeIn(pn, shift=LEFT * 0.3), run_time=1.0
        )
        self.play(TransformFromCopy(eq, pcount), run_time=0.9)
        self.play(
            wfr[:2].animate.set_stroke(C_PARAM),
            bfr[:2].animate.set_stroke(C_PARAM),
            run_time=0.6,
        )

        self.cue(7)  # 초기 값 ≈ 0 = 모름
        vals = np.concatenate([W0.ravel(), B0])
        hist, _ = np.histogram(vals, bins=21, range=(-2.1, 2.1))
        hb = bars(
            list(hist / hist.max() * 1.9),
            width=4.6,
            base_y=-2.85,
            center_x=3.9,
            colors=[C_PARAM] * len(hist),
        )
        z0 = VGroup(
            Line([3.9, -2.85, 0], [3.9, -0.75, 0], color=C_ACCENT, stroke_width=3),
            mono("0", 34, C_ACCENT).move_to([3.9, -0.55, 0]),
        )
        self.play(FadeOut(VGroup(eq, pcount)), pn.animate.move_to(RIGHT * 3.9 + UP * 1.6),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in hb], lag_ratio=0.04), run_time=1.4)  # fmt: skip
        self.play(Create(z0), run_time=0.6)
        unk = ko("≈ 0  →  아직 모름", 34, C_DIM).move_to(RIGHT * 3.9 + UP * 0.55)
        self.play(FadeIn(unk), run_time=0.6)

        self.cue(8)  # Parameter = 확률표의 재료
        self.play(FadeOut(VGroup(hb, z0, unk)), run_time=0.5)
        psq = Square(
            1.6,
            stroke_color=C_GOOD,
            stroke_width=4,
            fill_color=C_GOOD,
            fill_opacity=0.15,
        ).move_to(RIGHT * 3.9 + DOWN * 1.2)
        pm = VGroup(psq, en("P", 60, C_GOOD).move_to(psq))
        pml = en("Probability Matrix", 30, C_GOOD, max_w=4.4).next_to(
            pm, DOWN, buff=0.12
        )
        a = Arrow(
            pn.get_bottom(), pm.get_top(), buff=0.12, stroke_width=6, color=C_PARAM
        )
        self.play(GrowArrow(a), FadeIn(pm), FadeIn(pml), run_time=1.0)
        self.finish()

    # ════════════════════════════════════════════════════════
    # S07 순전파 — 주인공: Probability, Loss
    # ════════════════════════════════════════════════════════
    def s07(self):
        self.begin("S07", "5-2. 학습(순전파)")
        DY = -0.8  # 그래프 내용을 화면 높이의 10% 만큼 아래로
        DY2 = -0.6  # Loss 수식·그래프는 아래 여백을 넘지 않게 7.5% 만큼
        chain = VGroup(
            node("Parameter", C_PARAM), node("Probability", C_GOOD), node("Loss", C_POS)
        ).arrange(RIGHT, buff=1.2)
        chain.move_to(UP * 0.4)
        ar = VGroup(
            *[
                Arrow(
                    chain[i].get_right(),
                    chain[i + 1].get_left(),
                    buff=0.1,
                    stroke_width=6,
                    color=C_DIM,
                )
                for i in range(2)
            ]
        )
        self.play(FadeIn(chain[0]), run_time=0.5)
        self.play(GrowArrow(ar[0]), FadeIn(chain[1]), run_time=0.7)
        self.play(GrowArrow(ar[1]), FadeIn(chain[2]), run_time=0.7)

        self.cue(1)  # 현재 단어 가로줄 + Bias 줄 → 점수 58개
        hdr = VGroup(chain, ar)
        self.play(
            hdr.animate.scale(0.55).move_to([0, TOP_Y - 0.1 - 0.4, 0]), run_time=0.7
        )  # 화면 높이 5% (0.4) 더 아래: 제목과 안 겹치게
        wimg = signed_image(W0, height=2.0, scale=1.5, pos=C_W, neg=C_NEG).move_to(
            LEFT * 5.15 + UP * (0.6 + DY)
        )
        wfr = frame_of(wimg, C_W, 2)
        rr = cell_rect(wimg, row=EX, color=C_WHOLE, sw=4)
        self.play(FadeIn(wimg), Create(wfr), FadeIn(rr), run_time=0.8)
        BW, BX, sc = 7.8, 1.5, 0.5  # 막대 영역: x = -2.4 ~ 5.4
        LX = -3.4  # W·b 이름표 자리 (W 그림과 막대 사이)
        wrow = bars(
            list(W0[EX]),
            width=BW,
            base_y=1.0 + DY,
            center_x=BX,
            scale=sc,
            colors=[C_W if v >= 0 else C_NEG for v in W0[EX]],
        )
        wl = tex(r"W[\text{" + w(EX_TOK) + "}]", f"W[{w(EX_TOK)}]", 38, C_W).move_to(
            [LX, 1.0 + DY, 0]
        )
        fit(wl, 1.3)
        self.play(ReplacementTransform(rr.copy(), wrow), FadeIn(wl), run_time=1.2)
        brow = bars(
            list(B0 + 1e-3),
            width=BW,
            base_y=-0.3 + DY,
            center_x=BX,
            scale=sc,
            colors=[C_B] * V,
        )
        bl = tex("+\\,b", "+ b", 44, C_B).move_to([LX, -0.3 + DY, 0])
        self.play(FadeIn(brow), FadeIn(bl), run_time=0.8)
        # 점수 z 가 나오면 W 그림은 할 일을 마쳤으니 치운다 (왼쪽을 이름표 자리로)
        LX2 = -4.4
        zrow = bars(
            list(Z0),
            width=BW,
            base_y=1.2 + DY,
            center_x=BX,
            scale=sc,
            colors=[C_TEXT if v >= 0 else C_DIM for v in Z0],
        )
        zl = (
            VGroup(tex("z", "z", 48, C_TEXT), ko("점수 58개", 28, C_DIM))
            .arrange(RIGHT, buff=0.2)
            .move_to([LX2, 1.2 + DY, 0])
        )
        self.play(ReplacementTransform(VGroup(wrow, brow), zrow), ReplacementTransform(VGroup(wl, bl), zl),
                  FadeOut(Group(wimg, wfr, rr)), run_time=1.2)  # fmt: skip

        self.cue(2)  # 점수 → Probability (softmax)
        pmax = float(P0.max())
        prow = bars(
            list(P0),
            width=BW,
            base_y=-1.9 + DY,
            center_x=BX,
            scale=2.0 / pmax,
            colors=[C_GOOD] * V,
        )
        prow.set_fill(opacity=0.45)  # 다른 단어 확률은 흐리게 → egg 만 밝게 강조
        sm = VGroup(Arrow(UP * 0.35, DOWN * 0.35, buff=0, stroke_width=5, color=C_GOOD),
                    tex(r"\mathrm{softmax}", "softmax", 40, C_GOOD)).arrange(RIGHT, buff=0.15).move_to([LX2, 0.2 + DY, 0])  # fmt: skip
        pl = (
            VGroup(en("Probability", 38, C_GOOD, max_w=3.4), ko("확률 58개", 28, C_DIM))
            .arrange(DOWN, buff=0.1)
            .move_to([LX2, -0.9 + DY, 0])
        )
        self.play(FadeIn(sm), run_time=0.6)
        self.play(TransformFromCopy(zrow, prow), FadeIn(pl), run_time=1.4)
        s100 = mono("Σ = 100%", 36, C_GOOD).next_to(pl, DOWN, buff=0.3)
        self.play(FadeIn(s100), Indicate(chain[1], color=C_GOOD), run_time=0.8)

        self.cue(3)  # 정답은 egg
        self.play(FadeOut(VGroup(zrow, zl, sm)), prow.animate.shift(UP * 1.2), pl.animate.move_to([LX2, 0.7 + DY, 0]),
                  s100.animate.move_to([LX2, -0.3 + DY, 0]), run_time=0.6)  # fmt: skip
        star = ko("★", 44, C_ACCENT).next_to(prow[EX_NEXT], DOWN, buff=0.1)
        tag = token_box(NEXT_TOK, size=30).next_to(star, DOWN, buff=0.08)
        self.play(
            prow[EX_NEXT].animate.set_fill(YELLOW, 1),
            FadeIn(star, scale=1.6),
            FadeIn(tag),
            run_time=0.8,
        )

        self.cue(4)  # 1.8% = 찍은 수준
        pegg = mono(f"{P0[EX_NEXT]:.1%}", 60, C_ACCENT)
        guess = mono(f"≈ 1/{V}", 40, C_DIM)
        callout = (
            VGroup(pegg, guess).arrange(RIGHT, buff=0.4).next_to(tag, LEFT, buff=0.4)
        )  # 아래 여백을 넘지 않게 egg 왼쪽에
        callout.set_x(
            min(
                max(callout.get_x(), -2.4 + callout.width / 2),
                SAFE_X - callout.width / 2,
            )
        )
        self.play(
            FadeIn(pegg, shift=UP * 0.2),
            Flash(prow[EX_NEXT].get_top(), color=C_ACCENT),
            run_time=0.9,
        )
        self.play(FadeIn(guess), run_time=0.5)

        self.cue(5)  # 틀린 정도 = Loss
        self.play(FadeOut(VGroup(prow, pl, s100, star, tag, guess)), pegg.animate.move_to(LEFT * 3.9 + UP * (1.3 + DY2)),
                  Indicate(chain[2], color=C_POS), run_time=0.9)  # fmt: skip
        lossf = fit(tex(r"\text{Loss}=-\log P(\text{" + w(NEXT_TOK) + "})", f"Loss = −log P({w(NEXT_TOK)})", 54, C_TEXT,
                        colors={r"\text{Loss}": C_POS, r"P(\text{" + w(NEXT_TOK) + "})": C_GOOD}), 5.6)  # fmt: skip
        lossf.move_to(LEFT * 3.7 + UP * (-0.1 + DY2))
        self.play(Write(lossf), run_time=1.2)
        lnow = mono(f"= {-math.log(P0[EX_NEXT]):.2f}", 48, C_POS).next_to(
            lossf, DOWN, buff=0.3
        )
        self.play(FadeIn(lnow), run_time=0.6)

        self.cue(6)  # 확률 ↑ → Loss ↓ (그래프)
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 4.5, 1], x_length=5.6, y_length=4.0, tips=False,
                  axis_config={"color": C_DIM, "stroke_width": 3}).move_to(RIGHT * 3.2 + UP * (-0.5 + DY2))  # fmt: skip
        curve = ax.plot(
            lambda x: -math.log(x), x_range=[0.012, 1], color=C_POS, stroke_width=6
        )
        xl = en("P", 40, C_GOOD).next_to(ax.x_axis, RIGHT, buff=0.12)
        yl = en("Loss", 36, C_POS).next_to(ax.y_axis, UP, buff=0.08)
        pv = ValueTracker(float(P0[EX_NEXT]))
        dot = always_redraw(
            lambda: Dot(
                ax.c2p(pv.get_value(), -math.log(pv.get_value())),
                radius=0.13,
                color=C_ACCENT,
            )
        )
        vline = always_redraw(lambda: DashedLine(ax.c2p(pv.get_value(), 0), ax.c2p(pv.get_value(), -math.log(pv.get_value())),
                                                 color=C_DIM))  # fmt: skip
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
        self.play(Create(curve), run_time=1.0)
        self.add(vline, dot)
        self.play(pv.animate.set_value(0.97), run_time=2.0)
        self.play(pv.animate.set_value(float(P0[EX_NEXT])), run_time=1.2)

        self.cue(7)  # 113쌍 평균
        self.remove(vline, dot)
        self.play(FadeOut(VGroup(ax, curve, xl, yl, lossf, lnow, pegg)), run_time=0.6)
        # 113쌍의 Loss → 하나의 평균 숫자로 모으기 (말하려는 것 한 가지: 지금 Loss 4.15, 목표 0.76)
        lmax = float(PAIR_LOSS0.max())
        base = SAFE_BOTTOM + 0.4
        pairs = bars(
            list(PAIR_LOSS0),
            width=2 * SAFE_X - 0.6,
            base_y=base,
            center_x=0,
            scale=2.2 / lmax,
            colors=[C_POS] * N,
        )
        pairs.set_fill(opacity=0.55)
        nlab = (
            VGroup(mono(f"{N}", 44, C_POS), ko("쌍의 Loss", 30, C_DIM))
            .arrange(RIGHT, buff=0.15)
            .next_to(pairs, UP, buff=0.3)
        )
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in pairs], lag_ratio=0.01),
            FadeIn(nlab),
            run_time=2.0,
        )
        cur = (
            VGroup(en("Loss", 48, C_POS), mono(f"{LOSS0:.2f}", 96, C_POS))
            .arrange(DOWN, buff=0.2)
            .move_to([0, CONTENT_Y + 0.2, 0])
        )
        avg_tag = ko("평균", 30, C_DIM).next_to(cur, UP, buff=0.2)
        self.play(
            ReplacementTransform(pairs, cur[1]),
            FadeOut(nlab),
            FadeIn(cur[0]),
            FadeIn(avg_tag),
            run_time=1.2,
        )

        self.cue(8)  # 4.15 = 정답 확률 평균 1.6%
        cur_p = ko(f"정답 확률 ≈ {math.exp(-LOSS0):.1%}", 32, C_DIM).next_to(
            cur, DOWN, buff=0.3
        )
        self.play(FadeIn(cur_p, shift=UP * 0.1), run_time=0.7)

        self.cue(9)  # 목표 0.76 (47%)
        left_g = VGroup(avg_tag, cur, cur_p)
        goal = VGroup(ko("목표", 30, C_DIM), en("Loss", 48, C_POS), mono(f"{d.MIN_LOSS:.2f}", 96, C_POS),
                      ko(f"정답 확률 ≈ {math.exp(-d.MIN_LOSS):.0%}", 32, C_DIM)).arrange(DOWN, buff=0.2)  # fmt: skip
        self.play(left_g.animate.move_to([-3.4, CONTENT_Y, 0]), run_time=0.8)
        goal.move_to([3.4, CONTENT_Y, 0])
        arrow = Arrow(
            [-1.2, CONTENT_Y, 0],
            [1.2, CONTENT_Y, 0],
            buff=0,
            stroke_width=8,
            color=C_POS,
        )
        self.play(GrowArrow(arrow), FadeIn(goal, shift=LEFT * 0.3), run_time=1.0)
        self.finish()

    # ════════════════════════════════════════════════════════
    # S08 역전파 — 주인공: Gradient (3D Loss 언덕 포함)
    # ════════════════════════════════════════════════════════
    def s08(self):
        self.begin("S08", "5-3. 학습(역전파)")
        pn = node("Parameter", C_PARAM, wd=3.4, size=40).move_to(LEFT * 3.4 + UP * 0.2)
        cnt = mono(f"{N_PARAMS:,}", 48, C_PARAM).next_to(pn, DOWN, buff=0.25)
        q = (
            VGroup(*[ko(s, 60, C_ACCENT) for s in ["↑", "↓", "?"]])
            .arrange(RIGHT, buff=0.4)
            .move_to(RIGHT * 2.6 + UP * 0.2)
        )
        self.play(FadeIn(pn), FadeIn(cnt), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(x, scale=1.5) for x in q], lag_ratio=0.25),
            run_time=1.0,
        )

        self.cue(1)  # 역전파
        chain = VGroup(
            node("Parameter", C_PARAM), node("Probability", C_GOOD), node("Loss", C_POS)
        ).arrange(RIGHT, buff=1.3)
        chain.move_to(UP * 0.8)
        fwd = VGroup(
            *[
                Arrow(
                    chain[i].get_right(),
                    chain[i + 1].get_left(),
                    buff=0.1,
                    stroke_width=5,
                    color=C_DIM,
                )
                for i in range(2)
            ]
        )
        self.play(
            FadeOut(VGroup(q, cnt)),
            ReplacementTransform(pn, chain[0]),
            FadeIn(chain[1:]),
            FadeIn(fwd),
            run_time=1.0,
        )
        back = CurvedArrow(chain[2].get_bottom() + DOWN * 0.1, chain[0].get_bottom() + DOWN * 0.1, angle=-PI / 3,
                           stroke_width=7, color=C_GRAD)  # fmt: skip
        bl = VGroup(
            en("Backpropagation", 40, C_GRAD, max_w=7.0), ko("역전파", 32, C_TEXT)
        ).arrange(RIGHT, buff=0.3)
        bl.move_to([0, back.get_bottom()[1] - 0.4, 0])
        self.play(Create(back), FadeIn(bl), run_time=1.4)

        self.cue(2)  # Loss 에서 거꾸로 올라가며
        self.play(Indicate(chain[2], color=C_POS), run_time=0.6)
        self.play(Indicate(chain[1], color=C_GOOD), run_time=0.6)
        self.play(Indicate(chain[0], color=C_PARAM), run_time=0.6)

        self.cue(3)  # Gradient = 기울기 → 2D Loss 곡선 위의 공
        self.play(FadeOut(VGroup(chain, fwd, back, bl)), run_time=0.5)
        lab = VGroup(
            line_text("Gradient", 44, C_GRAD),
            line_text("= 기울기", 40, C_TEXT, KO_FONT),
        ).arrange(RIGHT, buff=0.25)
        lab.move_to([TITLE_X + lab.width / 2, TITLE_Y - 0.8, 0])  # 제목 바로 아래

        def L(x):  # 설명용 Loss 곡선 (U 자)
            return 0.32 * (x - 0.6) ** 2 + 0.35

        def dL(x):
            return 0.64 * (x - 0.6)

        ax = Axes(x_range=[-3, 4, 1], y_range=[0, 4.2, 1], x_length=8.4, y_length=4.2, tips=True,
                  axis_config={"color": C_FAINT, "stroke_width": 3, "include_ticks": False}).move_to([0.4, CONTENT_Y - 0.45, 0])  # fmt: skip
        xl = (
            line_text("Parameter", 30, C_PARAM)
            .next_to(ax.x_axis, DOWN, buff=0.15)
            .align_to(ax.x_axis, RIGHT)
        )
        yl = (
            line_text("Loss", 30, C_POS)
            .next_to(ax.y_axis, LEFT, buff=0.15)
            .align_to(ax.y_axis, UP)
        )
        curve = ax.plot(L, x_range=[-2.8, 3.9], color=C_TEXT, stroke_width=5)
        self.play(FadeIn(lab), Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
        self.play(Create(curve), run_time=0.8)

        xv = ValueTracker(-2.4)  # 공의 위치

        def tangent():
            x = xv.get_value()
            h = 0.9 / math.sqrt(1 + dL(x) ** 2)  # 접선 길이를 일정하게
            return Line(
                ax.c2p(x - h, L(x) - h * dL(x)),
                ax.c2p(x + h, L(x) + h * dL(x)),
                color=C_GRAD,
                stroke_width=6,
            )

        tan_line = always_redraw(tangent)
        ball = always_redraw(
            lambda: Dot(
                ax.c2p(xv.get_value(), L(xv.get_value())), radius=0.17, color=YELLOW
            )
        )
        self.play(FadeIn(ball), Create(tan_line), run_time=0.6)
        # 공이 기울기 반대쪽(내리막)으로 굴러감 → 바닥에서 접선이 평평해짐(기울기 0)
        self.play(xv.animate.set_value(0.6), run_time=2.2)
        self.cue(4, lead=0.4)  # the 가로줄: egg ↑ 나머지 ↓
        self.remove(tan_line, ball)
        self.play(FadeOut(VGroup(ax, xl, yl, curve, lab)), run_time=0.5)

        y = np.zeros(V)
        y[EX_NEXT] = 1.0
        gvec = P0 - y
        BW, BX = 9.4, 0.9
        shown = np.sign(-gvec) * np.sqrt(
            np.abs(gvec)
        )  # 작은 '내리기' 막대도 보이게 (√ 로 키움)
        gs = 2.0 / abs(shown).max()
        GY = (
            CONTENT_Y - 0.15
        )  # 막대 기준선: 위(egg)·아래(나머지)·안내문이 화면 가운데에 고르게 오도록
        gb = bars(
            list(shown),
            width=BW,
            base_y=GY,
            center_x=BX,
            scale=gs,
            colors=[C_GOOD if v <= 0 else C_POS for v in gvec],
        )
        base_line = Line(
            [BX - BW / 2, GY, 0], [BX + BW / 2, GY, 0], stroke_width=2, color=C_DIM
        )
        gl = (
            VGroup(en("Gradient", 38, C_GRAD, max_w=1.9), mono(w(EX_TOK), 32, C_WHOLE))
            .arrange(DOWN, buff=0.1)
            .move_to([-SAFE_X + 1.0, GY, 0])
        )
        egg_tag = token_box(NEXT_TOK, size=30).next_to(gb[EX_NEXT], UP, buff=0.1)
        hint2 = VGroup(VGroup(ko("▲", 40, C_GOOD), token_box(NEXT_TOK, size=30), ko("올리기", 30, C_GOOD)).arrange(RIGHT, buff=0.15),
                       VGroup(ko("▼", 40, C_POS), ko("나머지 57개 내리기", 30, C_POS)).arrange(RIGHT, buff=0.15)
                       ).arrange(RIGHT, buff=1.0).move_to(DOWN * 2.55)  # fmt: skip
        self.play(Create(base_line), LaggedStart(*[GrowFromEdge(b, DOWN if -gvec[i] >= 0 else UP) for i, b in enumerate(gb)],
                                                 lag_ratio=0.01), FadeIn(gl), run_time=1.4)  # fmt: skip
        self.play(
            FadeIn(egg_tag),
            FadeIn(hint2),
            Indicate(gb[EX_NEXT], color=C_GOOD, scale_factor=1.3),
            run_time=1.0,
        )

        self.cue(5)  # W 행렬·b 줄과 똑같이 생김
        self.play(FadeOut(VGroup(gb, gl, hint2, egg_tag, base_line)), run_time=0.5)
        H = 3.7  # 위 이름표가 제목과 겹치지 않게 (그림 윗변 y ≈ 1.65)
        wimg = signed_image(W0, height=H, scale=1.5, pos=C_W, neg=C_NEG).move_to(
            [-3.4, CONTENT_Y, 0]
        )
        gimg = signed_image(DW0, height=H, pos=C_POS, neg=C_GOOD).move_to(
            [3.4, CONTENT_Y, 0]
        )
        bimg = signed_image(
            B0[None, :] + 1e-9,
            height=H / V * 2.2,
            scale=1.5,
            pos=C_B,
            neg=C_NEG,
            width=wimg.width,
        )
        bimg.next_to(wimg, DOWN, buff=0.4)
        gbimg = signed_image(
            DB0[None, :], height=H / V * 2.2, pos=C_POS, neg=C_GOOD, width=gimg.width
        )
        gbimg.next_to(gimg, DOWN, buff=0.4)
        lw = tex("W,\\ b", "W, b", 48, C_PARAM).next_to(wimg, UP, buff=0.15)
        lg = tex(r"\nabla W,\ \nabla b", "∇W, ∇b", 48, C_GRAD).next_to(
            gimg, UP, buff=0.15
        )
        same = ko("같은 모양", 30, C_DIM).move_to([0, CONTENT_Y, 0])
        frs = VGroup(frame_of(wimg, C_W), frame_of(bimg, C_B))
        gfrs = VGroup(frame_of(gimg, C_GRAD), frame_of(gbimg, C_GRAD))
        self.play(FadeIn(wimg), FadeIn(bimg), Create(frs), FadeIn(lw), run_time=1.0)
        self.play(
            TransformFromCopy(Group(wimg, bimg), Group(gimg, gbimg)),
            Create(gfrs),
            FadeIn(lg),
            FadeIn(same),
            run_time=1.6,
        )

        self.cue(6)  # Parameter 3,422개마다 방향 하나씩
        rr = cell_rect(gimg, row=EX, color=C_ACCENT, sw=3)
        mark = cell_rect(gimg, row=EX, col=EX_NEXT, color=C_ACCENT, sw=5)
        self.play(
            Create(rr),
            Create(mark),
            Flash(mark.get_center(), color=C_GOOD),
            run_time=0.8,
        )
        tot = (
            VGroup(ko("방향", 30, C_DIM), mono(f"× {N_PARAMS:,}", 40, C_GRAD))
            .arrange(DOWN, buff=0.1)
            .move_to([0, CONTENT_Y - 1.1, 0])
        )
        self.play(FadeIn(tot), Indicate(gfrs, color=C_GRAD), run_time=1.0)
        self.finish()

    # ════════════════════════════════════════════════════════
    # S09 업데이트 — 주인공: Parameter 와 Probability 갱신
    # ════════════════════════════════════════════════════════
    def s09(self):
        self.begin("S09", "5-4. 학습(업데이트)")
        H = 3.8
        wimg = signed_image(W0, height=H, scale=1.5, pos=C_W, neg=C_NEG).move_to(
            [-4.4, CONTENT_Y - 0.25, 0]
        )
        gimg = signed_image(-DW0, height=H, pos=C_GOOD, neg=C_POS).move_to(
            [0, CONTENT_Y - 0.25, 0]
        )
        w1 = signed_image(WSNAP[1], height=H, scale=1.5, pos=C_W, neg=C_NEG).move_to(
            [4.4, CONTENT_Y - 0.25, 0]
        )
        # 세 이름표: 같은 글꼴·같은 크기·같은 높이
        l1 = line_text("Parameter", 36, C_PARAM).next_to(wimg, UP, buff=0.15)
        l2 = line_text("Gradient", 36, C_GRAD).next_to(gimg, UP, buff=0.15)
        l3 = line_text("New Parameter", 36, C_ACCENT).next_to(w1, UP, buff=0.15)
        for lb in (l2, l3):
            lb.set_y(l1.get_y())
        plus = mono("+", 60, C_DIM).move_to((wimg.get_right() + gimg.get_left()) / 2)
        eqs = mono("=", 60, C_DIM).move_to((gimg.get_right() + w1.get_left()) / 2)
        self.play(FadeIn(wimg), FadeIn(l1), run_time=0.6)
        self.play(FadeIn(plus), FadeIn(gimg), FadeIn(l2), run_time=0.8)
        self.play(FadeIn(eqs), FadeIn(w1, shift=LEFT * 0.4), FadeIn(l3), run_time=1.4)

        self.cue(1)  # the 가로줄 egg 칸 Weight ↑, 나머지 ↓
        BIGW = 4.6
        w1b = signed_image(
            WSNAP[1], height=BIGW, scale=1.5, pos=C_W, neg=C_NEG
        ).move_to([-3.4, CONTENT_Y - 0.2, 0])
        l3b = ko("Parameter (Weight)", 34, C_ACCENT).next_to(w1b, UP, buff=0.12)
        self.play(
            FadeOut(Group(wimg, gimg, plus, eqs, l1, l2)),
            FadeTransform(w1, w1b),
            FadeTransform(l3, l3b),
            run_time=0.8,
        )
        rr = cell_rect(w1b, row=EX, color=C_WHOLE, sw=3)
        hit = cell_rect(w1b, row=EX, col=EX_NEXT, color=C_ACCENT, sw=5)
        wv = ValueTracker(EGG_W[0])
        RX = 3.2

        def readout(label, tracker, fmt, color, y):
            return always_redraw(lambda: VGroup(label.copy(), mono(fmt.format(tracker.get_value()), 50, color))
                                 .arrange(RIGHT, buff=0.25).move_to([RX, y, 0]))  # fmt: skip

        wlabel = VGroup(
            en("Weight", 34, C_W),
            token_box(EX_TOK, size=26),
            mono("→", 26, C_DIM),
            token_box(NEXT_TOK, size=26),
        ).arrange(RIGHT, buff=0.1)
        wnum = readout(wlabel, wv, "{:+.2f}", C_ACCENT, 1.7)
        self.play(Create(rr), Create(hit), run_time=0.6)
        self.add(wnum)
        self.play(
            wv.animate.set_value(EGG_W[10]),
            Flash(hit.get_center(), color=C_ACCENT),
            run_time=1.6,
        )

        self.cue(2)  # Parameter 바뀌면 egg 확률 ↑
        pv = ValueTracker(EGG_P[0])
        plabel = VGroup(
            en("Probability", 34, C_GOOD), token_box(NEXT_TOK, size=26)
        ).arrange(RIGHT, buff=0.12)
        pnum = readout(plabel, pv, "{:.1%}", C_GOOD, 0.6)
        a1 = Arrow([RX, 1.35, 0], [RX, 0.95, 0], buff=0, stroke_width=5, color=C_DIM)
        self.play(GrowArrow(a1), run_time=0.3)
        self.add(pnum)
        self.play(pv.animate.set_value(EGG_P[10]), run_time=1.6)

        self.cue(3)  # 확률 ↑ → Loss ↓
        lv = ValueTracker(HISTORY[0])
        lnum = readout(en("Loss", 38, C_POS), lv, "{:.2f}", C_POS, -0.5)
        a2 = Arrow([RX, 0.25, 0], [RX, -0.15, 0], buff=0, stroke_width=5, color=C_DIM)
        self.play(GrowArrow(a2), run_time=0.3)
        self.add(lnum)
        self.play(lv.animate.set_value(HISTORY[10]), run_time=1.4)

        self.cue(4)  # Loss ↓ → Gradient 작아짐
        gv = ValueTracker(GNORM[0])
        a3 = Arrow([RX, -0.85, 0], [RX, -1.25, 0], buff=0, stroke_width=5, color=C_DIM)
        gbar = always_redraw(lambda: VGroup(
            en("Gradient", 36, C_GRAD),
            Rectangle(width=max(gv.get_value() / GNORM[0] * 3.2, 0.03), height=0.38, stroke_width=0, fill_color=C_GRAD, fill_opacity=0.9),
        ).arrange(RIGHT, buff=0.25).move_to([RX, -1.6, 0]))  # fmt: skip
        self.play(GrowArrow(a3), run_time=0.3)
        self.add(gbar)
        self.play(gv.animate.set_value(GNORM[100]), run_time=2.0)
        small = ko("고치는 양도 작아짐", 30, C_DIM).move_to([RX, -2.4, 0])
        self.play(FadeIn(small), run_time=0.5)

        self.cue(5)  # 500번 되풀이 — 고리
        self.remove(wnum, pnum, lnum, gbar)
        self.play(FadeOut(Group(w1b, l3b, rr, hit, a1, a2, a3, small)), run_time=0.5)
        ring = VGroup(node("Parameter", C_PARAM, wd=2.9), node("Probability", C_GOOD, wd=2.9), node("Loss", C_POS, wd=2.9),
                      node("Gradient", C_GRAD, wd=2.9))  # fmt: skip
        for m, p in zip(
            ring,
            [
                UP * 1.55,
                RIGHT * 3.7 + DOWN * 0.35,
                DOWN * 2.2,
                LEFT * 3.7 + DOWN * 0.35,
            ],
        ):
            m.move_to(p)
        arcs = VGroup()
        for i in range(4):
            s, e = ring[i].get_center(), ring[(i + 1) % 4].get_center()
            arcs.add(
                CurvedArrow(
                    s + (e - s) * 0.3,
                    e - (e - s) * 0.3,
                    angle=-PI / 4,
                    stroke_width=5,
                    color=C_DIM,
                )
            )
        self.play(
            LaggedStart(*[FadeIn(m, scale=0.8) for m in ring], lag_ratio=0.15),
            run_time=1.0,
        )
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), run_time=1.0)
        kv = ValueTracker(0)
        ktxt = always_redraw(lambda: VGroup(mono(f"{int(kv.get_value())}", 72, C_ACCENT), ko("번", 32, C_DIM)).arrange(RIGHT, buff=0.12)
                             .move_to(DOWN * 0.35))  # fmt: skip
        self.add(ktxt)
        dot = Dot(radius=0.13, color=C_ACCENT).move_to(arcs[0].get_start())
        self.play(Succession(*[MoveAlongPath(dot, a) for _ in range(3) for a in arcs]), kv.animate.set_value(500), run_time=3.6,
                  rate_func=linear)  # fmt: skip

        self.cue(6)  # Loss 10번 2.78 · 100번 0.91 · 500번 0.78
        self.remove(ktxt)
        self.play(FadeOut(VGroup(ring, arcs, dot)), run_time=0.5)
        ax = Axes(x_range=[0, 500, 100], y_range=[0, 4.5, 1], x_length=7.2, y_length=4.4, tips=False,
                  axis_config={"color": C_DIM, "stroke_width": 3}).move_to(LEFT * 2.2 + DOWN * 0.4)  # fmt: skip
        xl = ko("학습 횟수", 28, C_DIM).next_to(ax.x_axis, DOWN, buff=0.15)
        yl = en("Loss", 36, C_POS).next_to(ax.y_axis, UP, buff=0.08)
        goal = DashedLine(
            ax.c2p(0, d.MIN_LOSS), ax.c2p(500, d.MIN_LOSS), color=C_GOOD, stroke_width=4
        )
        curve = ax.plot_line_graph(
            list(range(501)),
            HISTORY,
            add_vertex_dots=False,
            line_color=C_POS,
            stroke_width=6,
        )
        ball = Dot(ax.c2p(0, HISTORY[0]), radius=0.15, color=C_ACCENT)
        self.play(
            Create(ax), FadeIn(xl), FadeIn(yl), Create(goal), FadeIn(ball), run_time=0.8
        )
        mini = prob_image(SNAP[0], height=2.6).move_to(RIGHT * 4.5 + DOWN * 0.3)
        mfr = frame_of(mini, C_GOOD, 3)
        mlab = en("Probability Matrix", 28, C_GOOD, max_w=3.4).next_to(
            mini, UP, buff=0.1
        )
        self.play(FadeIn(mini), Create(mfr), FadeIn(mlab), run_time=0.5)
        self.play(
            Create(curve),
            MoveAlongPath(ball, curve["line_graph"]),
            run_time=3.0,
            rate_func=linear,
        )
        tags = VGroup()
        for n_, off in ((10, (0.95, 0.3)), (100, (0.75, 0.4)), (500, (-0.7, 0.45))):
            p = ax.c2p(n_, HISTORY[n_])
            tags.add(
                VGroup(
                    Dot(p, radius=0.08, color=C_ACCENT),
                    mono(f"{n_}번 {HISTORY[n_]:.2f}", 30, C_ACCENT).move_to(
                        p + np.array([off[0], off[1], 0])
                    ),
                )
            )
        for t_, k_ in zip(tags, (10, 100, 500)):
            nxt = prob_image(SNAP[k_], height=2.6).move_to(mini)
            self.play(FadeIn(t_), FadeTransform(mini, nxt), run_time=0.6)
            mini = nxt

        self.cue(7)  # 업데이트된 Parameter 로 확률표 다시 계산
        self.play(
            FadeOut(VGroup(ax, xl, yl, goal, curve, ball, tags, mfr, mlab)),
            FadeOut(mini),
            run_time=0.5,
        )
        BIG = 5.0
        pm = prob_image(SNAP[0], height=BIG).move_to([-2.7, CONTENT_Y, 0])
        pfr = frame_of(pm, C_GOOD, 4)
        plab = en("Probability Matrix", 44, C_GOOD, max_w=5.0).move_to(
            RIGHT * 3.7 + UP * 1.8
        )
        kv2 = ValueTracker(0)
        ktag = always_redraw(lambda: VGroup(ko("학습", 30, C_DIM), mono(f"{int(kv2.get_value())}", 56, C_ACCENT), ko("번", 30, C_DIM))
                             .arrange(RIGHT, buff=0.15).move_to(RIGHT * 3.7 + UP * 0.9))  # fmt: skip
        self.play(FadeIn(pm), Create(pfr), Write(plab), run_time=1.0)
        self.add(ktag)
        for k_ in SNAP_K[1:]:
            nxt = prob_image(SNAP[k_], height=BIG).move_to(pm)
            self.play(
                FadeTransform(pm, nxt),
                kv2.animate.set_value(k_),
                run_time=0.28 if k_ < 100 else 0.4,
            )
            pm = nxt

        self.cue(8)  # egg 1.8% → 28.3%
        hit = cell_rect(pm, row=EX, col=EX_NEXT, color=C_ACCENT, sw=5)
        ev = ValueTracker(EGG_P[0])
        etxt = always_redraw(lambda: VGroup(token_box(EX_TOK, size=30), mono("→", 30, C_DIM), token_box(NEXT_TOK, size=30),
                                            mono(f"{ev.get_value():.1%}", 54, C_ACCENT)).arrange(RIGHT, buff=0.15)
                             .move_to(RIGHT * 3.7 + DOWN * 0.3))  # fmt: skip
        link = DashedLine(hit.get_right(), [1.2, -0.3, 0], color=C_ACCENT)
        self.play(
            Create(hit),
            Create(link),
            Flash(hit.get_center(), color=C_ACCENT),
            run_time=0.8,
        )
        self.add(etxt)
        self.play(ev.animate.set_value(EGG_P[500]), run_time=2.0)

        self.cue(9)  # 정답표 28.6% 와 거의 같음
        ans = VGroup(
            ko("정답표", 32, C_POS), mono(f"{d.COUNT_P[EX, EX_NEXT]:.1%}", 54, C_POS)
        ).arrange(RIGHT, buff=0.2)
        ans.move_to(RIGHT * 3.9 + DOWN * 1.7)
        approx = mono("≈", 60, C_ACCENT).move_to(RIGHT * 3.9 + DOWN * 1.0)
        self.play(FadeIn(approx), FadeIn(ans, shift=UP * 0.2), run_time=0.9)

        self.cue(10)  # 확률표 전체도 정답표 모양
        self.remove(ktag, etxt)
        self.play(FadeOut(VGroup(plab, hit, link, approx, ans)), run_time=0.4)
        H2 = 4.6
        left = prob_image(TABLE500, height=H2).move_to([-3.4, CONTENT_Y - 0.25, 0])
        right = prob_image(d.COUNT_P, height=H2).move_to([3.4, CONTENT_Y - 0.25, 0])
        lt = ko("학습 500번 (고쳐서)", 34, C_GOOD).next_to(left, UP, buff=0.15)
        rt = ko("정답표 (세어서)", 34, C_POS).next_to(right, UP, buff=0.15)
        mid = mono("≈", 90, C_ACCENT).move_to([0, CONTENT_Y - 0.25, 0])
        self.play(FadeTransform(pm, left), FadeOut(pfr), run_time=1.0)
        self.play(
            FadeIn(right),
            FadeIn(lt),
            FadeIn(rt),
            Create(frame_of(left, C_GOOD, 3)),
            Create(frame_of(right, C_POS, 3)),
            run_time=1.0,
        )
        self.play(FadeIn(mid, scale=1.5), run_time=0.6)

        self.cue(11)  # 세지 않고 고쳐서 찾아냄
        self.play(Circumscribe(left, color=C_ACCENT), run_time=1.4)
        self.finish()

    # ════════════════════════════════════════════════════════
    # S10 생성
    # ════════════════════════════════════════════════════════
    def chain_boxes(self, words, trace, top_y, size=28):
        items = []
        for i, wd in enumerate(words):
            items.append(token_box(wd, d.ROLE_OF.get(wd, "WHOLE"), size))
            if i < len(trace):
                items.append(mono(f"{trace[i][3]:.0%}", 22, C_DIM))
        return flow(items, gap=0.12, row_gap=0.2, top=top_y)

    def s10(self):
        self.begin("S10", "6. 생성")
        tr = []
        words = d.generate(EX_TOK, TABLE500, length=10, seed=7, trace=tr)
        cur = token_box(EX_TOK, size=48).move_to(LEFT * 4.6 + UP * 0.6)
        self.play(FadeIn(cur), run_time=0.6)

        self.cue(1)  # 다음 단어 확률
        cands = tr[0][1]
        rows = VGroup()
        for k_, (tok, p) in enumerate(cands):
            y = 1.8 - k_ * 0.66
            lab = fit(mono(tok, 32), 2.2).move_to([-1.2, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(
                width=max(p * 7, 0.04),
                height=0.44,
                stroke_width=0,
                fill_color=C_GOOD,
                fill_opacity=0.9,
            )
            bar.move_to([-1.0 + bar.width / 2, y, 0])
            rows.add(
                VGroup(
                    lab, bar, mono(f"{p:.0%}", 28, C_DIM).next_to(bar, RIGHT, buff=0.12)
                )
            )
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.1), *[FadeIn(r[0]) for r in rows],
                  *[FadeIn(r[2]) for r in rows], run_time=1.2)  # fmt: skip

        self.cue(2)  # 주사위로 하나 뽑기
        die = VGroup(
            Square(0.9, stroke_color=C_TEXT, stroke_width=4),
            *[
                Dot(radius=0.07, color=C_TEXT).move_to(p)
                for p in (UP * 0.22 + LEFT * 0.22, ORIGIN, DOWN * 0.22 + RIGHT * 0.22)
            ],
        )
        die.move_to(RIGHT * 4.8 + UP * 0.6)
        self.play(FadeIn(die), run_time=0.3)
        self.play(die.animate.rotate(PI), run_time=0.6)
        self.play(die.animate.rotate(PI / 2), run_time=0.4)
        pick = [i for i, (t, _) in enumerate(cands) if t == tr[0][2]]
        if pick:
            self.play(
                Indicate(rows[pick[0]], color=C_ACCENT, scale_factor=1.15), run_time=0.8
            )

        self.cue(3)  # 뽑힌 단어가 다음 출발점
        self.play(FadeOut(VGroup(rows, die)), run_time=0.4)
        nxt = token_box(tr[0][2], size=48).move_to(cur.get_center() + RIGHT * 3.0)
        a = Arrow(
            cur.get_right(), nxt.get_left(), buff=0.12, stroke_width=6, color=C_DIM
        )
        self.play(GrowArrow(a), FadeIn(nxt, shift=LEFT * 0.3), run_time=0.8)
        self.play(nxt.animate.move_to(cur), FadeOut(cur), FadeOut(a), run_time=0.8)

        self.cue(4)  # 학습 전: 엉터리
        self.play(FadeOut(nxt), run_time=0.3)
        tr0 = []
        w0 = d.generate(EX_TOK, SNAP[0], length=10, seed=7, trace=tr0)
        D10 = -0.64  # 화면 높이 8% 아래로
        l0 = ko("학습 0번", 32, C_POS).move_to([-SAFE_X + 1.0, TOP_Y - 0.1 + D10, 0])
        c0 = self.chain_boxes(w0, tr0, top_y=TOP_Y - 0.5 + D10)
        self.play(
            FadeIn(l0),
            LaggedStart(*[FadeIn(m) for m in c0], lag_ratio=0.05),
            run_time=2.0,
        )

        self.cue(5)  # 500번 학습: 원문다운
        l1 = ko("학습 500번", 32, C_GOOD).move_to([-SAFE_X + 1.1, -0.2 + D10, 0])
        c1 = self.chain_boxes(words, tr, top_y=-0.6 + D10)
        self.play(
            FadeIn(l1),
            LaggedStart(*[FadeIn(m) for m in c1], lag_ratio=0.05),
            run_time=2.0,
        )

        self.cue(6)  # 시도 3개 중 평균 확률 최고
        self.play(FadeOut(VGroup(l0, c0, l1, c1)), run_time=0.5)
        runs = []
        for s in (1, 2, 3):
            t3 = []
            ws = d.generate(EX_TOK, TABLE500, length=8, seed=s, trace=t3)
            runs.append((ws, float(np.mean([np.log(t[3]) for t in t3]))))
        best = max(range(3), key=lambda a_: runs[a_][1])
        lines = VGroup()
        for a_, (ws, sc) in enumerate(runs):
            txt = fit(mono(d.join_text(ws), 30, C_BG if a_ == best else C_TEXT), 9.2)
            pr = mono(f"{math.exp(sc):.0%}", 34, C_BG if a_ == best else C_GOOD)
            rowg = VGroup(txt, pr).arrange(RIGHT, buff=0.5)
            if a_ == best:
                bg = RoundedRectangle(corner_radius=0.1, width=rowg.width + 0.5, height=rowg.height + 0.35, stroke_width=0,
                                      fill_color=C_TEXT, fill_opacity=1).move_to(rowg)  # fmt: skip
                rowg = VGroup(bg, rowg)
            lines.add(rowg)
        lines.arrange(DOWN, buff=0.45).move_to(DOWN * 0.1)
        self.play(
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in lines], lag_ratio=0.3),
            run_time=1.5,
        )

        self.cue(7)  # 갈림길 15 / 58
        self.play(FadeOut(lines), run_time=0.4)
        non_fork = [i for i in range(V) if i not in FORKS]
        img = prob_image(d.COUNT_P, height=5.0, dim_rows=non_fork).move_to(
            [-2.6, CONTENT_Y, 0]
        )
        info = (
            VGroup(mono(f"{len(FORKS)}", 80, C_ACCENT), mono(f"/ {V}", 44, C_DIM))
            .arrange(RIGHT, buff=0.2)
            .move_to(RIGHT * 3.8 + UP * 0.8)
        )
        self.play(FadeIn(img), Create(frame_of(img)), FadeIn(info), run_time=1.0)

        self.cue(8)  # 나머지는 거의 하나로 정해짐
        one = (
            VGroup(mono(f"{V - len(FORKS)}", 56, C_DIM), mono("→ 1", 44, C_DIM))
            .arrange(RIGHT, buff=0.2)
            .move_to(RIGHT * 3.8 + DOWN * 0.4)
        )
        self.play(FadeIn(one), run_time=0.6)

        self.cue(9)  # 통째로 외움
        memo = ko("통째로 외움", 40, C_POS).move_to(RIGHT * 3.8 + DOWN * 1.6)
        self.play(FadeIn(memo, scale=1.3), run_time=0.6)

        self.cue(10)  # Overfitting
        of = en("Overfitting", 52, C_POS, max_w=4.8).next_to(memo, DOWN, buff=0.25)
        self.play(FadeIn(of), Circumscribe(of, color=C_POS), run_time=1.2)
        self.finish()

    # ════════════════════════════════════════════════════════
    # S11 대화
    # ════════════════════════════════════════════════════════
    QUESTION = "Who is the bird?"

    def s11(self):
        self.begin("S11", "7. 대화")
        q = self.QUESTION
        # 두 문장을 크게, 한 글자씩 써지듯(스트리밍) 보여 줌
        s1 = Text("데미안 원문에 없는\n" f'"{q}" 와 같은\n' "질문이 들어오면,\nLLM은 답변을 못합니다.",
                  font=KO_FONT, font_size=46, color=C_TEXT, line_spacing=1.3,
                  t2c={f'"{q}"': C_ACCENT, "LLM": C_WHOLE, "못합니다": C_POS})  # fmt: skip
        s2 = Text("이 모델은 대답하는 게 아니라\n이어 쓰기만 합니다.", font=KO_FONT, font_size=46, color=C_TEXT, line_spacing=1.3,
                  t2c={"이어 쓰기만": C_ACCENT})  # fmt: skip
        body = VGroup(s1, s2).arrange(DOWN, aligned_edge=LEFT, buff=0.75)
        fit(body, 2 * SAFE_X - 0.6, (TITLE_Y - 0.6) - (SAFE_BOTTOM + 0.2))
        body.move_to([0, CONTENT_Y, 0])

        self.cue(1)  # 첫 문장
        self.play(AddTextLetterByLetter(s1, rate_func=linear), run_time=6.0)

        self.cue(4)  # 둘째 문장 (이어 쓰기)
        self.play(AddTextLetterByLetter(s2, rate_func=linear), run_time=3.5)

        self.cue(5)
        key = [m for m in s2.submobjects][
            len("이모델은대답하는게아니라") : len("이모델은대답하는게아니라")
            + len("이어쓰기만")
        ]
        if key:
            self.play(
                Indicate(VGroup(*key), color=C_ACCENT, scale_factor=1.15), run_time=1.0
            )
        self.finish()

    # ════════════════════════════════════════════════════════
    # S12 마무리
    # ════════════════════════════════════════════════════════
    def s12(self):
        self.begin("S12", "정리")
        names = ["원문", "토큰화", "어휘사전", "인코딩", "확률표"]
        steps = (
            VGroup(*[ko_node(n, C_WHOLE, wd=1.8, ht=0.85, size=30) for n in names])
            .arrange(RIGHT, buff=0.4)
            .move_to(UP * 1.4)
        )
        arrs = VGroup(
            *[
                Arrow(
                    steps[i].get_right(),
                    steps[i + 1].get_left(),
                    buff=0.04,
                    stroke_width=4,
                    color=C_DIM,
                )
                for i in range(4)
            ]
        )
        self.play(
            LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.12),
            Create(arrs),
            run_time=1.6,
        )

        self.cue(1)  # 측정 · 방향 · 고치기 수백 번
        ring = VGroup(node("Parameter", C_PARAM, wd=2.7), node("Probability", C_GOOD, wd=2.7), node("Loss", C_POS, wd=2.7),
                      node("Gradient", C_GRAD, wd=2.7)).arrange(RIGHT, buff=0.55)  # fmt: skip
        fit(ring, 2 * SAFE_X).move_to(DOWN * 0.8)
        self.play(dim(steps, 0.4), dim(arrs, 0.4), run_time=0.4)

        self.cue(2)  # Parameter → Probability → Loss → Gradient
        ra = VGroup(
            *[
                Arrow(
                    ring[i].get_right(),
                    ring[i + 1].get_left(),
                    buff=0.05,
                    stroke_width=5,
                    color=C_DIM,
                )
                for i in range(3)
            ]
        )
        back = CurvedArrow(ring[3].get_bottom() + DOWN * 0.08, ring[0].get_bottom() + DOWN * 0.08, angle=-PI / 4, stroke_width=5,
                           color=C_GRAD)  # fmt: skip
        for i in range(4):
            self.play(
                FadeIn(ring[i], scale=0.8),
                *([GrowArrow(ra[i - 1])] if i > 0 else []),
                run_time=0.7,
            )
        self.play(Create(back), run_time=0.8)

        self.cue(3)  # 스스로 문장
        dot = Dot(radius=0.12, color=C_GRAD).move_to(back.get_start())
        self.play(MoveAlongPath(dot, back), run_time=1.2)
        self.remove(dot)

        self.cue(4)  # 진짜 GPT 도 같은 고리
        loop = VGroup(ring, ra, back)
        self.play(
            FadeOut(VGroup(steps, arrs)),
            loop.animate.scale(0.8).move_to(UP * 1.3),
            run_time=0.8,
        )
        # 왼쪽 Demian LLM / 오른쪽 GPT: 같은 줄은 같은 크기·같은 높이
        rows = [("Demian LLM", "GPT", 36), ("1", "∞", 56), ("앞 단어", "긴 문맥 (Transformer)", 28),
                (f"{N_PARAMS:,}", "175,000,000,000", 40), ("Parameter", "Parameter (GPT-3)", 28)]  # fmt: skip
        row_y = [-0.05, -0.8, -1.5, -2.1, -2.65]
        cmp_l, cmp_r = VGroup(), VGroup()
        for (lt, rt, sz), y in zip(rows, row_y):
            lc = C_ACCENT if sz != 28 else C_DIM
            rc = C_TEXT if sz != 28 else C_DIM
            cmp_l.add(line_text(lt, sz, lc, KO_FONT).move_to([-3.2, y, 0]))
            cmp_r.add(fit(line_text(rt, sz, rc, KO_FONT), 6.0).move_to([3.2, y, 0]))
        self.play(FadeIn(cmp_l[0]), FadeIn(cmp_r[0]), run_time=0.6)

        self.cue(5)  # 긴 문맥
        self.play(FadeIn(cmp_l[1:3]), FadeIn(cmp_r[1:3]), run_time=0.8)

        self.cue(6)  # 1,750억 개
        self.play(FadeIn(cmp_l[3:]), FadeIn(cmp_r[3:]), run_time=0.8)
        self.play(Indicate(cmp_r[3], color=C_ACCENT), run_time=0.8)

        self.cue(7)  # 알을 깨고 나온 작은 새
        self.play(FadeOut(VGroup(loop, cmp_l, cmp_r)), run_time=0.5)
        e = egg(0.9).move_to(DOWN * 0.3)
        self.play(FadeIn(e), run_time=0.5)
        cr = crack_line(e.get_center(), size=0.8)
        self.play(Create(cr), run_time=0.5)
        b = bird(1.0).move_to(e)
        self.play(FadeOut(e, scale=1.3), FadeOut(cr), FadeIn(b), run_time=0.6)
        free = en("The bird is free", 56, C_ACCENT).move_to(DOWN * 2.2)
        self.play(
            b.animate.move_to(UP * 1.9 + RIGHT * 3.6).scale(0.7),
            FadeIn(free),
            run_time=2.0,
        )
        self.finish()


# ── 실행할 장면들 ────────────────────────────────────────────
class S00Opening(Base):
    def construct(self):
        self.s00()


class S01Text(Base):
    def construct(self):
        self.s01()


class S02Tokenize(Base):
    def construct(self):
        self.s02()


class S03Vocab(Base):
    def construct(self):
        self.s03()


class S04Encode(Base):
    def construct(self):
        self.s04()


class S05Table(Base):
    def construct(self):
        self.s05()


class S06Init(Base):
    def construct(self):
        self.s06()


class S07Forward(Base):
    def construct(self):
        self.s07()


class S08Backward(Base):
    def construct(self):
        self.s08()


class S09Update(Base):
    def construct(self):
        self.s09()


class S10Generate(Base):
    def construct(self):
        self.s10()


class S11Chat(Base):
    def construct(self):
        self.s11()


class S12Outro(Base):
    def construct(self):
        self.s12()


class Full(Base):
    """전체 10분 14초 (Vrew 목소리와 같은 길이)"""

    def construct(self):
        for k in ORDER:
            getattr(self, k.lower())()
