"""
demian_llm.py — 『데미안』 문장으로 보는 LLM 학습 과정 (Matplotlib 대시보드)

  [0 원문] → [1 토큰화] → [2 어휘사전] → [3 인코딩] → [4 확률표]
  → [5-1 학습(초기화)] → [5-2 학습(순전파)] → [5-3 학습(역전파)] → [5-4 학습(업데이트)]
  → [6 생성] → [7 대화]

  · 화면 1920×1080. 그림·단추는 Matplotlib(맷플롯립)으로 그립니다. 창 아래 툴바는 숨겼습니다.
  · 0단계 입력창: 영문만 최대 512바이트까지 직접 타이핑할 수 있습니다 (한글 입력 불가).
    처음에는 『데미안』 이야기를 512바이트 안으로 다시 쓴 영어 문장이 채워져 있습니다.
    고친 뒤 [적용]을 누르면 1~7단계가 새 문장으로 다시 계산됩니다.
  · 토큰화: 실제 GPT가 쓰는 BPE 토크나이저 (OpenAI tiktoken 라이브러리)
      - GPT-4o 의 토크나이저(o200k_base)를 씁니다
      - 띄어쓰기도 토큰에 포함되며 화면에는 '·' 로 표시합니다 (" bird" → "·bird")
      - 처음 쓰는 토크나이저는 어휘 파일(수 MB)을 한 번 내려받아 캐시에 둡니다 (인터넷 필요)
  · 창은 작업 표시줄을 피해 화면에 맞춰 커지고, F11 키로 전체 화면을 켜고 끕니다.

[준비]  pip install matplotlib numpy tiktoken
[실행]  python demian_llm.py
"""

import random
import re
import textwrap
from collections import Counter, defaultdict

import matplotlib as mpl
import numpy as np

mpl.rcParams["toolbar"] = "None"  # 창 아래 툴바 숨기기
for _k in [k for k in mpl.rcParams if k.startswith("keymap.")]:
    mpl.rcParams[_k] = []  # 기본 단축키(q=닫기 등) 끄기
from matplotlib import font_manager  # noqa: E402

_have = {f.name for f in font_manager.fontManager.ttflist}
KOREAN = [
    f
    for f in (
        "Malgun Gothic",
        "NanumGothic",
        "AppleGothic",
        "Noto Sans CJK KR",
    )
    if f in _have
]  # 설치된 한글 글꼴만 사용
mpl.rcParams["font.family"] = KOREAN + ["DejaVu Sans"]
mpl.rcParams["font.serif"] = ["DejaVu Serif"] + KOREAN
mpl.rcParams["font.monospace"] = ["DejaVu Sans Mono"] + KOREAN
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.widgets import Button, TextBox  # noqa: E402

# ════════════════════════════════════════════════════════════════════
# 원문 (기본값) — 『데미안』 이야기를 512바이트 안으로 다시 쓴 영어 문장
# ════════════════════════════════════════════════════════════════════
MAX_BYTES = 512
DEFAULT_TEXT = (
    "The bird struggles out of the egg. The egg is the world. Whoever wants to be born must "
    "first destroy a world. The bird flies to God, and the name of that God is Abraxas. I was"
    " unhappy and uncertain. The thoughts are not forbidden. The transformation was painful "
    "and beautiful. My old world was destroyed, and I was reborn. We are experiments of "
    "nature, unfinished and incomplete. The destruction of the egg is the beginning of birth."
    " The egg is broken, the bird is free, and the world is new. I discovered it."
)

# ════════════════════════════════════════════════════════════════════
# 토큰화 — 실제 GPT 모델이 쓰는 BPE 토크나이저 (OpenAI tiktoken 라이브러리)
#   BPE(Byte Pair Encoding): 글자(바이트)에서 시작해, 학습 데이터에서 자주 붙어 나오는
#   두 조각을 하나로 합치는 '병합 규칙'을 수만~수십만 번 쌓아 만든 어휘로 글을 자른다.
#   · 자주 쓰는 단어는 통째로 1토큰, 드문 단어는 여러 조각   (Abraxas → Ab | rax | as)
#   · 띄어쓰기도 토큰에 들어간다: " bird" 를 화면에서는 "·bird" 로 표시
#   · 대·소문자를 구분한다: "The" 와 "·the" 는 다른 토큰
# ════════════════════════════════════════════════════════════════════
ENC_NAME, MODEL_NAME = "o200k_base", "GPT-4o"  # 쓰는 토크나이저
SPACE, NEWLINE = "·", "↵"  # 화면에 보이는 띄어쓰기·줄바꿈 표시

try:
    import tiktoken

    HAVE_TIKTOKEN = True
except ImportError:
    HAVE_TIKTOKEN = False

_ENCODERS = {}


def get_encoder(name):
    """tiktoken 인코더 (처음 한 번 인터넷에서 어휘 파일을 받아 캐시에 둔다)"""
    if not HAVE_TIKTOKEN:
        return None
    if name not in _ENCODERS:
        try:
            _ENCODERS[name] = tiktoken.get_encoding(name)
        except Exception as ex:  # 인터넷이 안 될 때 등
            print(f"tiktoken '{name}' 어휘를 불러오지 못했습니다:", ex)
            _ENCODERS[name] = None
    return _ENCODERS[name]


def tokenizer_label():
    enc = get_encoder(ENC_NAME)
    if enc is None:
        return "단어 단위 (tiktoken 없음)"
    return f"{MODEL_NAME} Tokenizer ({ENC_NAME}, Vocabulary {enc.n_vocab:,}개)"


# tiktoken 이 없을 때: GPT-2 의 '미리 자르기' 규칙만 써서 단어 단위로 자른다
FALLBACK_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def ascii_only(s):
    """한글 등 영문(ASCII) 밖의 글자는 지운다"""
    return "".join(ch for ch in s if ord(ch) < 128)


def clip_bytes(s, limit=MAX_BYTES):
    """limit 바이트를 넘으면 잘라 낸다 (영문 1글자 = 1바이트)"""
    return s.encode("utf-8")[:limit].decode("utf-8", "ignore")


def to_disp(piece):
    return piece.replace(" ", SPACE).replace("\n", NEWLINE)


def tokenize(text, name=None):
    """문장 → [(토큰, 실제 GPT 토큰 번호)].  토큰의 띄어쓰기는 '·' 로 바꿔 둔다"""
    enc = get_encoder(name or ENC_NAME)
    if enc is None:
        return [(to_disp(p), None) for p in FALLBACK_RE.findall(text)]
    return [
        (
            to_disp(
                enc.decode_single_token_bytes(i).decode("utf-8", "replace")
            ),
            i,
        )
        for i in enc.encode(text)
    ]


def join_text(toks):
    """토큰 → 문장: 그냥 이어 붙이고 '·' 를 띄어쓰기로 되돌린다"""
    return "".join(toks).replace(SPACE, " ").replace(NEWLINE, "\n").strip()


def token_roles(toks):
    """한 단어가 여러 토큰으로 나뉘었는지 표시: FIRST(첫 조각) · CONT(이어지는 조각) · WHOLE(통째)"""
    roles = ["WHOLE"] * len(toks)
    for i in range(1, len(toks)):
        t, prev = toks[i], toks[i - 1]
        if (t[0].isalpha() or t[0] == "'") and prev[-1:].isalpha():
            roles[i] = "CONT"
            if roles[i - 1] == "WHOLE":
                roles[i - 1] = "FIRST"
    return roles


# ════════════════════════════════════════════════════════════════════
# 1~4단계 계산 — 원문이 바뀌면 build() 로 전부 다시 계산
# ════════════════════════════════════════════════════════════════════
TOP_K = 30  # 행렬 그림에 보여 줄 토큰 수 (많이 나온 순)
LR = 10.0  # 학습률 η (처음 값)
TRAIN = {
    "lr": LR,
    "opt": "SGD",
}  # 5-4단계 단추로 바꾸는 값: 학습률 η, 옵티마이저
LRS = (1, 10, 30, 100)
OPTIMIZERS = (("SGD", "SGD"), ("Momentum", "Momentum"), ("Adam", "Adam"))
INIT_SIGMA, INIT_SEED = 0.5, 42  # 초기화: W 의 표준편차, 난수 씨앗


def build(text):
    global \
        TEXT, \
        WORDS, \
        tokens, \
        REAL_ID, \
        TOK_ROLE, \
        ROLE_OF, \
        vocab, \
        id_to_tok, \
        freq, \
        ids, \
        counts
    global model, V, X, Y, N, COUNT_P, MIN_LOSS, SUB, EX, EX_NEXT, START
    text = clip_bytes(ascii_only(text))
    pairs = tokenize(text)
    toks = [t for t, _ in pairs]
    if len(set(toks)) < 2:
        raise ValueError("영어 단어가 2개 이상 있어야 계산할 수 있어요.")
    TEXT, WORDS, tokens = text, re.findall(r"[A-Za-z]+", text), toks
    REAL_ID = {t: i for t, i in pairs}  # 토큰 → 실제 GPT 토큰 번호
    TOK_ROLE = token_roles(toks)
    ROLE_OF = {}
    for t, r in zip(toks, TOK_ROLE):
        if r != "WHOLE" or t not in ROLE_OF:
            ROLE_OF[t] = r
    # 2단계: 어휘사전 — 실제 GPT 토큰 번호 순서로 0번부터 다시 번호를 붙인다
    order = sorted(
        set(toks),
        key=lambda t: (REAL_ID.get(t) is None, REAL_ID.get(t) or 0, t),
    )
    vocab = {tok: i for i, tok in enumerate(order)}
    id_to_tok = {i: tok for tok, i in vocab.items()}
    freq = Counter(tokens)
    ids = [vocab[t] for t in tokens]  # 3단계
    counts = defaultdict(Counter)  # 4단계
    for cur, nxt in zip(ids, ids[1:]):
        counts[cur][nxt] += 1
    model = {}
    for cur, c in counts.items():
        total = sum(c.values())
        model[cur] = {nxt: n / total for nxt, n in c.items()}
    # 5단계(학습)용
    V = len(vocab)
    X, Y = np.array(ids[:-1]), np.array(ids[1:])
    N = len(X)
    COUNT_P = np.zeros((V, V))
    for i, row in model.items():
        for j, p in row.items():
            COUNT_P[i, j] = p
    MIN_LOSS = float(-np.mean(np.log(COUNT_P[X, Y])))
    SUB = sorted(sorted(range(V), key=lambda i: -freq[id_to_tok[i]])[:TOP_K])

    # 설명에 쓸 예시: 가장 많이 나온 단어 토큰과 그 다음에 가장 자주 온 토큰
    def word_start(t):  # '·the' 처럼 단어 하나로 된 토큰
        return t.strip(SPACE).isalpha() and ROLE_OF.get(t) != "CONT"

    cands = (
        [i for i in counts if word_start(id_to_tok[i]) and len(counts[i]) >= 2]
        or [i for i in counts if word_start(id_to_tok[i])]
        or list(counts)
    )
    EX = max(
        cands,
        key=lambda i: (freq[id_to_tok[i]], id_to_tok[i].startswith(SPACE)),
    )
    EX_NEXT = counts[EX].most_common(1)[0][0]
    START = id_to_tok[EX]


def softmax(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


class BigramParams:
    """파라미터: 가중치 W (V×V) + 바이어스 b (V).  z = W[현재 토큰] + b,  p = softmax(z)"""

    def __init__(self):
        self.reset()

    def reset(self):
        rng = np.random.default_rng(INIT_SEED)
        self.W = rng.normal(0, INIT_SIGMA, (V, V))
        self.b = np.zeros(V)
        self.updates = 0
        self.history = []
        self.state = {}  # Momentum·Adam 이 기억하는 값 (속도, 평균)

    def forward(self):
        Z = self.W[X] + self.b  # N×V 점수(logit)
        P = softmax(Z)  # N×V 확률
        loss = float(-np.mean(np.log(np.maximum(P[np.arange(N), Y], 1e-12))))
        return Z, P, loss

    def table(self):
        return softmax(
            self.W + self.b
        )  # V×V: 행 = 현재 토큰, 열 = 다음 토큰 확률

    def grads(self):
        _, P, loss = self.forward()
        dZ = P.copy()
        dZ[np.arange(N), Y] -= 1  # ∂L/∂z = (p − 정답) / N
        dZ /= N
        dW = np.zeros_like(self.W)
        np.add.at(dW, X, dZ)  # 같은 현재 토큰의 기울기는 같은 행에 모임
        db = dZ.sum(axis=0)
        return dW, db, dZ, loss

    def step(self):
        """업데이트 한 번.  옵티마이저:
        SGD      W ← W − η·g
        Momentum v ← 0.9·v + g,  W ← W − η·v                 (관성: 이전 방향을 이어 감)
        Adam     m ← 평균(g), s ← 평균(g²),  W ← W − (η/100)·m/√s  (파라미터마다 보폭 자동 조절)"""
        dW, db, _, loss = self.grads()
        lr, opt, st = TRAIN["lr"], TRAIN["opt"], self.state
        t = self.updates + 1
        for name, g in (("W", dW), ("b", db)):
            if opt == "Momentum":
                st["v" + name] = 0.9 * st.get("v" + name, 0) + g
                delta = lr * st["v" + name]
            elif opt == "Adam":
                st["m" + name] = 0.9 * st.get("m" + name, 0) + 0.1 * g
                st["s" + name] = 0.999 * st.get("s" + name, 0) + 0.001 * g * g
                m_hat = st["m" + name] / (1 - 0.9**t)
                s_hat = st["s" + name] / (1 - 0.999**t)
                delta = lr * 0.01 * m_hat / (np.sqrt(s_hat) + 1e-8)
            else:
                delta = lr * g
            setattr(self, name, getattr(self, name) - delta)
        self.updates += 1
        self.history.append(loss)
        return loss


def generate(start_tok, table, length=14, seed=7, trace=None):
    """6·7단계. 학습한 파라미터로 만든 확률표 table = softmax(W + b) 로 다음 토큰을 뽑는다.
    trace 에 (현재 토큰, 상위 후보 [(토큰, 확률)], 선택한 토큰, 선택한 토큰의 확률) 을 기록"""
    rng = random.Random(seed)
    cur = vocab[start_tok]
    out = [start_tok]
    for _ in range(length):
        row = table[cur]
        nxt = rng.choices(range(V), weights=row)[0]
        if trace is not None:
            top = np.argsort(-row)[:5]
            trace.append(
                (
                    id_to_tok[cur],
                    [(id_to_tok[j], float(row[j])) for j in top],
                    id_to_tok[nxt],
                    float(row[nxt]),
                )
            )
        cur = nxt
        out.append(id_to_tok[cur])
    return out


def top_text():
    """가장 많이 나온 토큰 설명 (2단계)"""
    mx = max(freq.values())
    tops = [t for t, c in freq.items() if c == mx]
    name = {".": "마침표(.)", ",": "쉼표(,)"}
    return (
        ", ".join(name.get(t, f"'{t}'") for t in tops)
        + f"가 {mx}번으로 가장 많이 나옴"
    )


def count_example():
    """예시 토큰으로 확률 계산 보여 주기 (4단계)"""
    cnt = counts[EX]
    tot = sum(cnt.values())
    top = cnt.most_common(4)
    more = " …" if len(cnt) > 4 else ""
    return (
        f"  '{id_to_tok[EX]}' 다음 → "
        + ", ".join(f"{id_to_tok[n]} {c}번" for n, c in top)
        + more
        + f"\n      (합계 {tot}번) → "
        + ", ".join(f"{c / tot:.0%}" for _, c in top)
        + more
    )


def find_token(word):
    """사용자가 친 단어 → 어휘사전 토큰들 (예: 'the' → ['·the'],  'Transformers' → ['Transform', 'ers']).
    모든 조각을 배웠을 때만 돌려주고, 못 찾으면 None"""
    w = word.strip()
    if not w:
        return None
    seqs = [
        [t for t, _ in tokenize(x)]
        for x in dict.fromkeys(
            (
                " " + w,
                w,
                " " + w.capitalize(),
                w.capitalize(),
                " " + w.lower(),
                w.lower(),
            )
        )
    ]
    for seq in seqs:  # ① 한 토큰으로 딱 맞음
        if len(seq) == 1 and seq[0] in vocab:
            return seq
    low = w.lower()
    same = [
        t for t in vocab if t.strip(SPACE).lower() == low
    ]  # ② 대·소문자만 다름
    if same:
        return [max(same, key=lambda t: freq[t])]
    for seq in seqs:  # ③ 여러 조각, 모두 배운 조각
        if seq and all(t in vocab for t in seq):
            return seq
    return None


def word_examples(n=6):
    """시작 단어로 쓸 만한 토큰 예시 (많이 나온 순)"""
    out = [
        t
        for t, _ in freq.most_common()
        if t.strip(SPACE).isalpha() and ROLE_OF.get(t) != "CONT"
    ]
    return [t.strip(SPACE) for t in out[:n]]


def split_words():
    """여러 토큰으로 나뉜 단어 [(단어, [조각들])] (등장 순서, 중복 없이)"""
    out, seen, cur = [], set(), []
    for t, r in zip(tokens + [""], TOK_ROLE + ["WHOLE"]):
        if r == "CONT":
            cur.append(t)
            continue
        if len(cur) > 1:
            w = join_text(cur)
            if w not in seen:
                seen.add(w)
                out.append((w, cur))
        cur = [t] if r == "FIRST" else []
    return out


build(DEFAULT_TEXT)


# ════════════════════════════════════════════════════════════════════
# 화면 (Matplotlib)
# ════════════════════════════════════════════════════════════════════
FIG_W, FIG_H, DPI = 19.2, 10.8, 100  # 1920 × 1080 픽셀
NAVY, ORANGE, TEAL, SKY, BG = (
    "#1d3557",
    "#e76f51",
    "#2a9d8f",
    "#a8dadc",
    "#f1f3f6",
)
PLAIN, PLAIN_BAR = (
    ("#f4f6f9", "#c5ced8"),
    "#b9c2cc",
)  # 통째 토큰 (색 구분 없음)
ROLE_COLOR = {
    "FIRST": ("#fde3da", ORANGE),
    "CONT": ("#d6efe9", TEAL),
    "WHOLE": PLAIN,
}
ROLE_BAR = {"FIRST": ORANGE, "CONT": TEAL, "WHOLE": PLAIN_BAR}
ROLE_NAME = {
    "FIRST": "단어의 첫 조각",
    "CONT": "이어지는 조각",
    "WHOLE": "통째 토큰",
}
STEPS = [
    ("0", "원문"),
    ("1", "토큰화"),
    ("2", "어휘사전"),
    ("3", "인코딩"),
    ("4", "확률표"),
    ("5-1", "학습(초기화)"),
    ("5-2", "학습(순전파)"),
    ("5-3", "학습(역전파)"),
    ("5-4", "학습(업데이트)"),
    ("6", "생성"),
    ("7", "대화"),
]
GEN_STEP, CHAT_STEP = 9, 10
LEFT = [0.008, 0.045, 0.334, 0.805]  # 왼쪽 설명 영역 [x, y, 폭, 높이]
RIGHT = [0.350, 0.045, 0.642, 0.805]  # 오른쪽 시각화 영역
INPUT_RECT = [
    RIGHT[0] + 0.012,
    0.255,
    RIGHT[2] - 0.024,
    0.545,
]  # 0단계 입력창 (그림 좌표)


def wrap(text, width=60):
    """한글은 2칸, 영문은 1칸으로 세어 줄바꿈"""
    out = []
    for para in text.split("\n"):
        line, w = "", 0
        ind = len(para) - len(para.lstrip())
        for ch in para:
            cw = 2 if ord(ch) > 0x2E80 else 1
            if w + cw > width and line:
                cut = line.rfind(" ")
                if (
                    ch != " " and line[-1:] != " " and cut > ind
                ):  # 단어 중간에서 끊지 않음
                    out.append(
                        line[:cut]
                    )  # 영어 단어·숫자는 중간에서 끊지 않음
                    line = " " * ind + line[cut + 1 :]
                else:
                    out.append(line)
                    line = " " * ind
                w = sum(2 if ord(c) > 0x2E80 else 1 for c in line)
            line += ch
            w += cw
        out.append(line)
    return "\n".join(out)


def short(t, n=8):
    """그림의 눈금 글자가 길면 줄인다 (·understanding → ·underst…)"""
    return t if len(t) <= n else t[: n - 1] + "…"


def fit_fs(text, width, base=14):
    """칸 폭(픽셀)에 맞게 글자 크기 줄이기"""
    need = len(text) * base * 0.62 * DPI / 72
    return base if need <= width - 8 else max(7.5, base * (width - 8) / need)


class _TextHolder:
    """그림 위 글자도 단계가 바뀔 때 함께 지우기 위한 작은 껍데기"""

    def __init__(self, text):
        self.text = text

    def remove(self):
        self.text.remove()


class Dashboard:
    def __init__(self):
        self.cur = 0
        self.unlocked = 0
        self.seed = 1
        self.start_word = [START]  # 6단계 시작 토큰들 (여러 조각일 수 있음)
        self.row_sel = None
        self.dyn = []  # 단계마다 새로 만드는 그림 영역
        self.net = BigramParams()  # 5단계(학습)의 파라미터
        self.timer = None
        self.left = 0
        self.widgets = []  # 단계마다 새로 만드는 입력칸·단추
        self.tk_inputs = []  # 단계마다 새로 만드는 Tk 입력칸 (6·7단계)
        self.past = []  # 5-4단계: 이전 설정의 손실 곡선 [(이름, 기록)]
        self.draft = TEXT  # 0단계 입력창 내용 (적용 전 초안)
        self.tk_text = None  # Tk 입력창 (TkAgg 창일 때만)
        self.notice = ""
        self.warn = ""

        self.fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("데미안 LLM")
        except Exception:
            pass
        f = self.fig

        # ── 머리 ───────────────────────────────────────────────
        f.patches.append(
            Rectangle(
                (0, 0.94),
                1,
                0.06,
                transform=f.transFigure,
                color=NAVY,
                zorder=-1,
            )
        )
        f.text(
            0.012,
            0.97,
            "데미안 LLM",
            color="white",
            fontsize=21,
            fontweight="bold",
            va="center",
        )

        # ── 단계 단추 줄 ────────────────────────────────────────
        self.btns = []
        x = 0.008
        for i, (num, name) in enumerate(STEPS):
            ax = f.add_axes([x, 0.868, 0.0675, 0.056])
            two = "(" in name  # 5-1 학습(초기화) → 두 줄
            b = Button(
                ax,
                f"{num} {name.replace('(', chr(10) + '(')}"
                if two
                else f"{num} {name}",
            )
            b.label.set_fontsize(11.5 if two else 13)
            b.label.set_linespacing(1.15)
            b.label.set_fontweight("bold")
            b.on_clicked(lambda _e, k=i: self.click(k))
            self.btns.append(b)
            if i < len(STEPS) - 1:
                f.text(
                    x + 0.0712,
                    0.896,
                    "▶",
                    color="#9aa5b1",
                    fontsize=10,
                    ha="center",
                    va="center",
                )
            x += 0.0775
        ax = f.add_axes([0.872, 0.868, 0.12, 0.056])
        self.next_btn = Button(
            ax, "다음 단계 ▶", color=ORANGE, hovercolor="#d65a3a"
        )
        self.next_btn.label.set_fontsize(15)
        self.next_btn.label.set_fontweight("bold")
        self.next_btn.label.set_color("white")
        self.next_btn.on_clicked(lambda _e: self.next())

        # ── 왼쪽·오른쪽 바탕 ─────────────────────────────────────
        for rect, label in ((LEFT, "설명"), (RIGHT, "시각화")):
            f.patches.append(
                Rectangle(
                    (rect[0], rect[1]),
                    rect[2],
                    rect[3],
                    transform=f.transFigure,
                    facecolor="white",
                    edgecolor="#c5ced8",
                    lw=1,
                    zorder=-1,
                )
            )
            f.text(
                rect[0] + 0.006,
                rect[1] + rect[3] - 0.004,
                label,
                fontsize=11,
                color="#8a96a3",
                va="top",
            )
        self.title_txt = f.text(
            LEFT[0] + 0.012,
            0.815,
            "",
            fontsize=19,
            fontweight="bold",
            color=NAVY,
            va="top",
        )
        self.desc_txt = f.text(
            LEFT[0] + 0.012,
            0.765,
            "",
            fontsize=13.5,
            va="top",
            linespacing=1.65,
            color="#222",
        )

        # ── 상태 줄 ────────────────────────────────────────────
        f.patches.append(
            Rectangle(
                (0, 0),
                1,
                0.032,
                transform=f.transFigure,
                color="#e3e8ef",
                zorder=-1,
            )
        )
        self.status = f.text(
            0.012, 0.016, "", fontsize=11, va="center", color="#333"
        )

        f.canvas.mpl_connect("key_press_event", self.on_key)
        f.canvas.mpl_connect("button_press_event", self.on_click)
        self.show(0)

    # ── 창 크기: 작업 표시줄과 겹치지 않게 ───────────────────────────
    def setup_window(self):
        """창을 '최대화'로 띄운다. 최대화는 작업 표시줄을 뺀 영역만 쓰므로 겹치지 않는다.
        창이 1920×1080보다 작아지면 on_resize 가 화면 전체를 같은 비율로 줄인다."""
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)
        win = getattr(self.fig.canvas.manager, "window", None)
        if win is None or not hasattr(win, "attributes"):  # Tk 창이 아닐 때
            return

        def maximize():
            try:
                win.state("zoomed")  # Windows · macOS
            except Exception:
                try:
                    win.attributes("-zoomed", True)  # Linux
                except Exception:
                    pass

        def toggle_full(_e=None):  # F11: 전체 화면(방송용)
            win.attributes("-fullscreen", not win.attributes("-fullscreen"))

        def leave_full(_e=None):
            win.attributes("-fullscreen", False)

        win.bind("<F11>", toggle_full)
        win.bind("<Escape>", leave_full)
        win.after(100, maximize)  # 창이 뜬 다음에 최대화

    def on_resize(self, _e=None):
        """창 크기에 맞춰 dpi 를 바꿔, 1920×1080 화면 배치를 통째로 늘이거나 줄인다"""
        w, h = self.fig.bbox.width, self.fig.bbox.height
        if w < 50 or h < 50:
            return
        new = min(w / FIG_W, h / FIG_H)
        if abs(new - self.fig.dpi) < 0.5:
            return
        try:
            self.fig._set_dpi(new, forward=False)
        except AttributeError:
            self.fig.dpi = new
        self.fig.set_size_inches(w / new, h / new, forward=False)
        self.fig.canvas.draw_idle()

    # ── 단계 이동 ────────────────────────────────────────────────
    def on_key(self, e):
        if any(
            isinstance(w, TextBox) and w.capturekeystrokes
            for w in self.widgets
        ):
            return
        if e.key == "right":
            self.next()
        elif e.key == "left" and self.cur > 0:
            self.click(self.cur - 1)

    def click(self, k):
        if 0 <= k <= self.unlocked + 1:
            self.show(k)

    def next(self):
        if self.cur < len(STEPS) - 1:
            self.show(self.cur + 1)

    def style(self, b, bg, fg, hover=None):
        b.color, b.hovercolor = bg, hover or bg
        b.ax.set_facecolor(bg)
        b.label.set_color(fg)

    def refresh_buttons(self):
        for i, b in enumerate(self.btns):
            if i == self.cur:
                self.style(b, NAVY, "white")
            elif i <= self.unlocked:
                self.style(b, SKY, NAVY, "#8ecbd0")
            elif i == self.unlocked + 1:
                self.style(b, "white", ORANGE, "#fde3da")
            else:
                self.style(b, "#e6e6e6", "#aaaaaa")
        last = self.cur == len(STEPS) - 1
        self.style(
            self.next_btn,
            "#bbbbbb" if last else ORANGE,
            "white",
            None if last else "#d65a3a",
        )

    def clear(self):
        if self.tk_text is not None:
            self.draft = self.get_draft()
            self.tk_text.place_forget()
        for ax in self.dyn:
            ax.remove()
        for w in self.widgets:
            w.ax.remove()
        for e in self.tk_inputs:
            e.destroy()
        self.dyn, self.widgets, self.tk_inputs = [], [], []
        self.flow_txt = {}

    def stop_timer(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    def show(self, k):
        self.stop_timer()
        self.cur = k
        self.unlocked = max(self.unlocked, k)
        self.row_sel = None
        self.clear()
        self.refresh_buttons()
        title, fn = [
            (f"[0단계] 원문 입력 (영문 최대 {MAX_BYTES}바이트)", self.s0),
            ("[1단계] 토큰화 — 실제 GPT의 BPE 방식", self.s1),
            ("[2단계] 어휘사전", self.s2),
            ("[3단계] 인코딩", self.s3),
            ("[4단계] 확률표 — 세어서 구하기", self.s4),
            ("[5-1단계] 학습 ① 초기화", self.p_init),
            ("[5-2단계] 학습 ② 순전파와 손실", self.p_forward),
            ("[5-3단계] 학습 ③ 역전파: 기울기", self.p_backward),
            ("[5-4단계] 학습 ④ 업데이트와 반복", self.p_update),
            ("[6단계] 생성", self.s5),
            ("[7단계] 모델과 대화하기", self.s6),
        ][k]
        self.title_txt.set_text(title)
        fn()
        self.status.set_color("#333")
        self.status.set_fontweight("normal")
        self.status.set_text(
            (self.notice + "    ·    " if self.notice else "")
            + f"{STEPS[k][0]}단계    ·    키보드 → ← 로도 이동"
            + "    ·    F11 전체 화면"
        )
        self.notice = ""
        self.fig.canvas.draw_idle()

    # ── 그리기 도구 ──────────────────────────────────────────────
    def set_desc(self, s, bottom=0.125, size=13.5):
        """설명란: 길면 글자를 줄여 칸 안에 넣는다"""
        s = s.strip("\n")
        fs = size
        while fs > 10.5:
            width = int(62 * 13.5 / fs)
            lines = wrap(s, width).count("\n") + 1
            if lines * fs * 1.65 * DPI / 72 <= (0.765 - bottom) * FIG_H * DPI:
                break
            fs -= 0.5
        self.desc_txt.set_fontsize(fs)
        self.desc_txt.set_text(wrap(s, int(62 * 13.5 / fs)))

    def board(self, rect=None):
        """픽셀 단위 좌표(왼쪽 위 = 0,0)를 쓰는 그림 영역"""
        rect = rect or [
            RIGHT[0] + 0.006,
            RIGHT[1] + 0.008,
            RIGHT[2] - 0.012,
            RIGHT[3] - 0.03,
        ]
        ax = self.fig.add_axes(rect)
        w, h = rect[2] * FIG_W * DPI, rect[3] * FIG_H * DPI
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        self.dyn.append(ax)
        return ax, w, h

    def chip(
        self,
        ax,
        x,
        y,
        w,
        h,
        text,
        sub=None,
        fc="#eef3f8",
        ec="#9fb3c8",
        tc="#222",
        fs=14,
        sub_color=ORANGE,
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=6",
                fc=fc,
                ec=ec,
                lw=1.3,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2 + (4 if sub is not None else 0),
            text,
            ha="center",
            va="center",
            fontsize=fit_fs(text, w, fs),
            fontweight="bold",
            color=tc,
        )
        if sub is not None:
            ax.text(
                x + 7,
                y + 5,
                sub,
                ha="left",
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color=sub_color,
                family="monospace",
            )

    def tok_chip(self, ax, x, y, w, h, tok, sub=None, fs=14, role=None):
        """여러 토큰으로 나뉜 단어의 조각만 색칠, 통째 토큰(문장부호 포함)은 회색"""
        fc, ec = ROLE_COLOR[role or ROLE_OF.get(tok, "WHOLE")]
        self.chip(ax, x, y, w, h, tok, sub=sub, fc=fc, ec=ec, fs=fs)

    def legend(self, ax, x, y, cats=("FIRST", "CONT", "WHOLE"), fs=12):
        for c in cats:
            fc, ec = ROLE_COLOR[c]
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    22,
                    18,
                    boxstyle="round,pad=0,rounding_size=4",
                    fc=fc,
                    ec=ec,
                    lw=1.3,
                )
            )
            sw = split_words()
            ex = sw[0][1] if sw else ["·Ab", "rax"]
            whole = next(
                (
                    t
                    for t, _ in freq.most_common()
                    if ROLE_OF.get(t) == "WHOLE" and t.strip(SPACE).isalpha()
                ),
                "·bird",
            )
            label = {
                "FIRST": f"단어의 첫 조각 {ex[0]}",
                "CONT": f"이어지는 조각 {ex[1]}",
                "WHOLE": f"통째 토큰 {whole}",
            }[c]
            ax.text(
                x + 30, y + 9, label, va="center", fontsize=fs, color="#333"
            )
            x += 30 + len(label) * 11 + 26

    def arrow(self, ax, x1, y1, x2, y2, color="#888", lw=1.8):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, mutation_scale=14
            ),
        )

    def widget_ax(self, x, w, y=0.055, h=0.045):
        return self.fig.add_axes([LEFT[0] + x, y, w, h])

    def teal_button(
        self, x, w, label, fn, y=0.055, color=TEAL, hover="#238a7e"
    ):
        b = Button(
            self.widget_ax(x, w, y), label, color=color, hovercolor=hover
        )
        b.label.set_color("white")
        b.label.set_fontsize(13)
        b.label.set_fontweight("bold")
        b.on_clicked(fn)
        self.widgets.append(b)
        return b

    def alert(self, text):
        """아래 상태 줄에 주황색 안내 (설명 글과 겹치지 않게)"""
        self.status.set_text("⚠  " + text)
        self.status.set_color(ORANGE)
        self.status.set_fontweight("bold")
        self.fig.canvas.draw_idle()

    def text_input(
        self,
        x,
        w,
        initial,
        submit,
        label=None,
        y=0.055,
        h=0.045,
        justify="center",
    ):
        """한 줄 입력칸. Tk 창이면 Tk 입력칸(빠름)을, 아니면 Matplotlib TextBox 를 쓴다.
        (Matplotlib TextBox 는 글자 하나 칠 때마다 화면 전체를 다시 그려서 느리다)
        돌려주는 값: (내용 읽기 함수, 내용 바꾸기 함수)"""
        if label:
            self.ftext(
                LEFT[0] + x - 0.005,
                y + h / 2,
                label,
                ha="right",
                va="center",
                fontsize=13,
            )
        canvas = self.tk_canvas()
        if canvas is not None:
            import tkinter as tk

            def ok(new):  # 영어(ASCII)만 받기
                if all(ord(c) < 128 for c in new):
                    return True
                self.alert(
                    "영어(알파벳)만 입력할 수 있어요. 한글은 입력되지 않습니다."
                )
                return False

            import tkinter.font as tkfont

            fams = set(tkfont.families())
            fam = next(
                (
                    f
                    for f in ("Segoe UI", "Malgun Gothic", "DejaVu Sans")
                    if f in fams
                ),
                "Helvetica",
            )
            e = tk.Entry(
                canvas,
                font=(fam, -20),
                justify=justify,
                relief="flat",
                bg="#f8fafc",
                highlightthickness=2,
                highlightbackground="#9fb3c8",
                highlightcolor=ORANGE,
            )
            e.insert(0, initial)
            e.configure(validate="key", validatecommand=(e.register(ok), "%P"))
            e.bind("<Return>", lambda _ev: submit())
            e.place(
                relx=LEFT[0] + x, rely=1 - (y + h), relwidth=w, relheight=h
            )
            e.focus_set()
            e.icursor("end")
            self.tk_inputs.append(e)

            def put(v):
                e.delete(0, "end")
                e.insert(0, v)

            return e.get, put
        box = TextBox(
            self.widget_ax(x, w, y, h),
            "",
            initial=initial,
            textalignment="center",
        )
        box.text_disp.set_fontsize(14)
        self.widgets.append(box)
        self.english_only(box)
        box.on_submit(lambda _t: submit())

        def put_box(v):
            box.eventson = False
            box.set_val(v)
            box.eventson = True

        return (lambda: box.text), put_box

    def english_only(self, box, msg=None):
        """입력칸에 한글 등 영문이 아닌 글자가 들어오면 지운다.
        (윈도우 한글 입력기 + Tk 조합에서 '가'가 깨진 글자로 들어오는 문제 방지)"""

        def check(text):
            clean = ascii_only(text)
            if clean != text:
                box.eventson = False  # set_val 이 '보내기'까지 일으키지 않게
                box.set_val(clean)
                box.eventson = True
                box.cursor_index = len(clean)
                box._rendercursor()
                self.alert(
                    "영어(알파벳)만 입력할 수 있어요. 한글은 입력되지 않습니다."
                )
                self.fig.canvas.draw_idle()

        box.on_text_change(check)

    def ftext(self, x, y, s, **kw):
        t = self.fig.text(x, y, s, **kw)
        self.dyn.append(_TextHolder(t))
        return t

    # ════════════════════════════════════════════════════════════
    # 0. 원문 입력
    # ════════════════════════════════════════════════════════════
    def tk_canvas(self):
        try:
            return self.fig.canvas.get_tk_widget()
        except Exception:
            return None

    def get_draft(self):
        if self.tk_text is not None:
            return self.tk_text.get("1.0", "end-1c")
        return self.draft

    def make_tk_text(self, canvas):
        import tkinter as tk
        import tkinter.font as tkfont

        fams = set(tkfont.families())
        self.mono = next(
            (
                f
                for f in (
                    "Consolas",
                    "D2Coding",
                    "DejaVu Sans Mono",
                    "Courier New",
                )
                if f in fams
            ),
            "Courier",
        )
        self.input_size = 0
        t = tk.Text(
            canvas,
            wrap="word",
            font=(self.mono, -19),
            bg="#f8fafc",
            fg="#1b2a3a",
            relief="flat",
            highlightthickness=2,
            highlightbackground="#9fb3c8",
            highlightcolor=ORANGE,
            padx=16,
            pady=12,
            undo=True,
            insertwidth=3,
            spacing1=2,
            spacing3=4,
        )

        def nbytes():
            return len(t.get("1.0", "end-1c").encode("utf-8"))

        def on_key(e):  # 한글 막기, 512바이트가 차면 더 입력하지 않음
            if e.char and ord(e.char[0]) > 127:
                self.warn = "한글(영문 외 글자)은 입력할 수 없어요"
                t.bell()
                self.update_bytes()
                return "break"
            if e.char and e.char.isprintable() and not (e.state & 0x4):
                sel = t.tag_ranges("sel")
                freed = len(t.get(*sel).encode("utf-8")) if sel else 0
                if nbytes() - freed + len(e.char.encode("utf-8")) > MAX_BYTES:
                    t.bell()
                    return "break"
            return None

        def after_change(
            _e=None,
        ):  # 붙여넣기·한글 조합 뒤에도 영문 512바이트로 맞춤
            s = t.get("1.0", "end-1c")
            fixed = clip_bytes(ascii_only(s))
            if fixed != s:
                if ascii_only(s) != s:
                    self.warn = "한글(영문 외 글자)은 입력할 수 없어요"
                pos = t.index("insert")
                t.delete("1.0", "end")
                t.insert("1.0", fixed)
                t.mark_set("insert", pos)
                t.bell()
            self.update_bytes()

        t.bind("<Key>", on_key)
        t.bind("<KeyRelease>", after_change)
        t.bind("<<Paste>>", lambda e: t.after_idle(after_change))
        t.bind("<ButtonRelease>", after_change)
        t.bind("<Configure>", lambda e: t.after_idle(self.fit_input_font))
        return t

    def fit_input_font(self):
        """512바이트(영문 512자)가 입력창에 꽉 차도록 글자 크기를 정한다"""
        import tkinter.font as tkfont

        t = self.tk_text
        w = t.winfo_width() - 2 * 16 - 10
        h = t.winfo_height() - 2 * 12 - 6
        if w < 100 or h < 60:
            return
        sample = ((DEFAULT_TEXT + " ") * 3)[:MAX_BYTES]  # 보통 영어 문장 512자
        size = 10
        for px in range(72, 9, -1):
            f = tkfont.Font(family=self.mono, size=-px)
            space, lines, cur = f.measure(" "), 1, 0
            for word in sample.split(" "):
                ww = f.measure(word)
                if cur and cur + space + ww > w:
                    lines, cur = lines + 1, ww
                else:
                    cur += (space if cur else 0) + ww
            if lines * (f.metrics("linespace") + 6) <= h:
                size = px
                break
        if size != self.input_size:
            self.input_size = size
            t.configure(font=(self.mono, -size))

    def update_bytes(self):
        s = self.get_draft()
        n = len(s.encode("utf-8"))
        if getattr(self, "byte_txt", None) is not None:
            self.byte_txt.set_text(f"{n:,} / {MAX_BYTES:,} 바이트")
            self.byte_txt.set_color(ORANGE if n >= MAX_BYTES else NAVY)
            self.byte_sub.set_text(
                self.warn
                or (
                    f"단어 {len(re.findall(r'[a-z]+', s.lower()))}개 · "
                    + ("적용 전 (원문과 다름)" if s != TEXT else "적용됨")
                )
            )
            self.byte_sub.set_color(ORANGE if self.warn else "#666")
            self.warn = ""
            self.fig.canvas.draw_idle()

    def s0(self):
        lib = (
            tokenizer_label()
            if get_encoder(ENC_NAME)
            else "※ tiktoken이 없어 단어 단위로 자름 (pip install tiktoken)"
        )
        self.set_desc(f"""
■ Training할 데이터 (최대 {MAX_BYTES}Byte)
오른쪽 입력창에 영어 문장 직접 타이핑 가능
  · 영문 1글자 = 1Byte, {MAX_BYTES}Byte가 차면 입력 중단
  · 한글 등 영문이 아닌 글자는 입력 불가

처음에는 헤세 『데미안』의 "새는 알을 깨고 나온다" 이야기를 {MAX_BYTES}Byte 안으로 다시 쓴 영어 문장이 채워져 있음
AI가 계산하기 쉽도록
  · 짧은 문장, 같은 단어(the bird, the egg, the world)를 되풀이
  · 흔한 단어(통째로 token 1개)와 드문 단어(여러 조각: Abraxas)를 함께 넣음

■ 사용법
  [적용] 입력창 내용으로 1~7단계를 다시 계산
  [데미안 원문] 처음 문장으로 되돌리기,  [지우기] 비우기

■ Tokenization 도구
  {lib}
""")
        self.teal_button(
            0.012,
            0.09,
            "적용 ▶",
            self.apply_text,
            color=ORANGE,
            hover="#d65a3a",
        )
        self.teal_button(
            0.11, 0.11, "데미안 원문", lambda _e: self.set_draft(DEFAULT_TEXT)
        )
        self.teal_button(0.228, 0.08, "지우기", lambda _e: self.set_draft(""))

        canvas = self.tk_canvas()
        x, y, w, h = INPUT_RECT
        if canvas is not None:
            if self.tk_text is None:
                self.tk_text = self.make_tk_text(canvas)
                self.tk_text.insert("1.0", self.draft)
            self.tk_text.place(
                relx=x, rely=1 - (y + h), relwidth=w, relheight=h
            )
            self.tk_text.after_idle(self.fit_input_font)
        else:  # Tk 창이 아닐 때: 그림으로 보여 주기
            ax, W, H = self.board([x, y, w, h])
            ax.add_patch(
                FancyBboxPatch(
                    (2, 2),
                    W - 4,
                    H - 4,
                    boxstyle="round,pad=0,rounding_size=8",
                    fc="#f8fafc",
                    ec="#9fb3c8",
                    lw=1.3,
                )
            )
            fs = 10
            for pt in range(40, 9, -1):  # 512자가 꽉 차는 글자 크기 찾기
                cw, lh = 0.6 * pt * DPI / 72, 1.45 * pt * DPI / 72
                lines = len(
                    textwrap.wrap(
                        ((DEFAULT_TEXT + " ") * 3)[:MAX_BYTES],
                        int((W - 40) / cw),
                    )
                )
                if lines * lh <= H - 30:
                    fs = pt
                    break
            ax.text(
                18,
                14,
                textwrap.fill(
                    self.draft, int((W - 40) / (0.6 * fs * DPI / 72))
                ),
                fontsize=fs,
                va="top",
                family="monospace",
                linespacing=1.3,
            )
        self.ftext(
            x,
            y + h + 0.012,
            "입력창 — 클릭해서 고치거나 새로 타이핑하세요",
            fontsize=12,
            color="#8a96a3",
            fontweight="bold",
        )
        self.byte_txt = self.ftext(
            x + w,
            y + h + 0.012,
            "",
            fontsize=15,
            ha="right",
            fontweight="bold",
        )
        self.byte_sub = self.ftext(
            x + w, y - 0.018, "", fontsize=11, ha="right", color="#666"
        )

        ax, W, H = self.board(
            [RIGHT[0] + 0.006, RIGHT[1] + 0.012, RIGHT[2] - 0.012, 0.165]
        )
        n_split = len(split_words())
        stats = [
            (f"{len(TEXT.encode('utf-8')):,}", "바이트 (적용된 원문)"),
            (f"{len(re.sub(r'\s', '', TEXT)):,}", "글자 수 (띄어쓰기 제외)"),
            (len(WORDS), "단어 수"),
            (len(tokens), "토큰 수 (1단계)"),
            (n_split, "여러 토큰으로 나뉜 단어"),
            (V, "어휘 수 (2단계)"),
        ]
        bw = (W - 40 - (len(stats) - 1) * 16) / len(stats)
        for i, (v, k) in enumerate(stats):
            bx = 20 + i * (bw + 16)
            ax.add_patch(
                FancyBboxPatch(
                    (bx, 10),
                    bw,
                    H - 20,
                    boxstyle="round,pad=0,rounding_size=8",
                    fc=NAVY,
                    ec="none",
                )
            )
            ax.text(
                bx + bw / 2,
                H * 0.42,
                str(v),
                fontsize=30,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
            )
            ax.text(
                bx + bw / 2,
                H * 0.78,
                k,
                fontsize=fit_fs(k, bw, 12.5),
                color=SKY,
                ha="center",
                va="center",
            )
        self.update_bytes()

    def set_draft(self, s):
        s = clip_bytes(ascii_only(s))
        self.draft = s
        if self.tk_text is not None:
            self.tk_text.delete("1.0", "end")
            self.tk_text.insert("1.0", s)
        self.update_bytes()

    def apply_text(self, _e=None):
        s = self.get_draft()
        self.status.set_text("토큰화하는 중 …")
        self.fig.canvas.draw()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        try:
            build(s)
        except ValueError as ex:
            self.notice = f"적용 못 함: {ex}"
            self.show(0)
            return
        self.draft = TEXT
        self.net = BigramParams()
        self.past = []
        self.start_word = [START]
        self.unlocked = 0
        self.notice = (
            f"적용 완료: {len(WORDS)}단어 → {len(tokens)}토큰, 어휘 {V}개"
        )
        self.show(0)

    # ════════════════════════════════════════════════════════════
    # 1. 토큰화 (실제 GPT 토크나이저)
    # ════════════════════════════════════════════════════════════
    def s1(self):
        sw = split_words()
        enc = get_encoder(ENC_NAME)
        n_role = Counter(TOK_ROLE)
        self.set_desc(
            """
■ 실제 LLM의 Tokenization: BPE (Byte Pair Encoding)
글자(Byte)에서 시작해, 엄청난 양의 Training 글에서 자주 붙어 나오는 두 조각을 하나로 합치는 규칙을 수만~수십만 번 쌓아 Vocabulary를 만듦
  · 자주 쓰는 단어는 통째로 token 1개
  · 드문 단어는 여러 조각
"""
            + "".join(f"      {w} → {' | '.join(p)}\n" for w, p in sw[:3])
            + f"""  · 띄어쓰기도 token에 포함 (· = 띄어쓰기)
  · 대·소문자 구분: "The" ≠ "·the"

■ 지금 Tokenizer: {tokenizer_label()}
OpenAI의 tiktoken 라이브러리로 실제 GPT와 똑같이 자름
카드 왼쪽 위 숫자 = 그 GPT Model의 실제 token 번호

■ 결과: {len(WORDS)}단어 → token {len(tokens)}개
  여러 조각으로 나뉜 단어 {len(sw)}개 (색칠한 token {n_role["FIRST"] + n_role["CONT"]}개)
"""
            if enc
            else f"""
■ Tokenization
※ tiktoken 라이브러리가 없어 단어 단위로 자름
명령 프롬프트에서  pip install tiktoken  을 실행한 뒤 다시 켜면 실제 GPT 방식(BPE)으로 자름

결과: {len(WORDS)}단어 → token {len(tokens)}개
"""
        )
        ax, W, H = self.board()
        self.legend(ax, 20, 6)
        ax.text(
            W - 20,
            6,
            f"토큰 전체 {len(tokens)}개",
            fontsize=14,
            fontweight="bold",
            color=NAVY,
            va="top",
            ha="right",
        )
        top = 44
        per = 12
        while True:  # 모든 토큰이 들어가도록 칸 수·높이 정하기
            rows = (len(tokens) - 1) // per + 1
            rh = min(64, (H - top - 6) / rows)
            if rh >= 40 or per >= 20:
                break
            per += 1
        tw = (W - 40) / per
        fs = 14 if rh >= 52 else 12
        for i, t in enumerate(tokens):
            r, c = divmod(i, per)
            rid = REAL_ID.get(t)
            self.tok_chip(
                ax,
                20 + c * tw + 3,
                top + r * rh,
                tw - 6,
                rh - 6,
                t,
                sub=str(rid) if (rh >= 46 and rid is not None) else None,
                fs=fs,
                role=TOK_ROLE[i],
            )

    # ════════════════════════════════════════════════════════════
    # 2. 어휘사전
    # ════════════════════════════════════════════════════════════
    def s2(self):
        vc = Counter(ROLE_OF[t] for t in vocab)
        self.set_desc(f"""
■ Vocabulary: 서로 다른 token에 번호 붙이기
token {len(tokens)}개 중 중복을 빼면 서로 다른 token은 {V}개

실제 GPT는 Vocabulary {get_encoder(ENC_NAME).n_vocab if get_encoder(ENC_NAME) else V:,}개 전체에 번호가 붙어 있음. 우리 Model은 작게 만들려고, 원문에 나온 {V}개만 골라 GPT 번호 순서대로 0번부터 다시 번호를 붙임
  예) '{id_to_tok[EX]}' : GPT 번호 {REAL_ID.get(id_to_tok[EX])} → 우리 번호 {EX}

■ 오른쪽 그래프 (많이 나온 {min(TOP_K, V)}개)
막대 길이 = 그 token이 원문에 나온 횟수
왼쪽 숫자 = 우리 번호,  괄호 = GPT 실제 번호
{top_text()}
""")
        ax = self.fig.add_axes(
            [RIGHT[0] + 0.11, RIGHT[1] + 0.05, RIGHT[2] - 0.14, RIGHT[3] - 0.1]
        )
        self.dyn.append(ax)
        names = [t for t, _ in freq.most_common(min(TOP_K, V))]
        vals = [freq[t] for t in names]
        y = list(range(len(names)))
        ax.barh(
            y, vals, color=[ROLE_BAR[ROLE_OF[t]] for t in names], height=0.72
        )
        ax.set_yticks(
            y,
            [f"{vocab[t]:>3}  {t}  ({REAL_ID.get(t, '-')})" for t in names],
            fontsize=11.5,
        )
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(
                v + max(vals) * 0.008, i, f"{v}회", va="center", fontsize=10.5
            )
        ax.set_xlim(0, max(vals) * 1.08)
        ax.set_xlabel("등장 횟수", fontsize=12)
        ax.set_title(
            f"어휘사전 {V}개 중 많이 나온 {len(names)}개  —  번호 · 토큰 · (GPT 번호) · 등장 횟수",
            fontsize=14,
            fontweight="bold",
            color=NAVY,
            loc="left",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        for c in ("FIRST", "CONT", "WHOLE"):
            ax.bar(0, 0, color=ROLE_BAR[c], label=f"{ROLE_NAME[c]} ({vc[c]})")
        ax.legend(loc="lower right", fontsize=12, frameon=False)

    # ════════════════════════════════════════════════════════════
    # 3. 인코딩
    # ════════════════════════════════════════════════════════════
    def s3(self):
        t0 = id_to_tok[EX]
        self.set_desc(f"""
■ Encoding: 문장 → 숫자 배열
각 token을 Vocabulary의 번호로 바꿈. 이제 문장은 숫자 {len(ids)}개짜리 배열

이것이 AI가 실제로 보는 모습. AI 입장에서 '{t0}'는 단어가 아니라 그냥 {EX}번

■ 확인 포인트
같은 token은 항상 같은 번호. '{t0}'는 모두 {EX}
'·' 는 띄어쓰기. 'The' 와 '·the' 는 다른 token이라 번호도 다름
""")
        ax, W, H = self.board()
        cols = 16
        cw = (W - 40) / cols
        y = 10
        for r in range(0, 48, cols):
            for c, (t, n) in enumerate(
                zip(tokens[r : r + cols], ids[r : r + cols])
            ):
                x = 20 + c * cw
                self.tok_chip(
                    ax,
                    x + 3,
                    y,
                    cw - 6,
                    50,
                    t,
                    sub=str(r + c),
                    fs=12,
                    role=TOK_ROLE[r + c],
                )
                self.arrow(ax, x + cw / 2, y + 52, x + cw / 2, y + 76)
                self.chip(
                    ax,
                    x + 3,
                    y + 78,
                    cw - 6,
                    46,
                    str(n),
                    fc=NAVY,
                    ec=NAVY,
                    tc="white",
                    fs=16,
                )
            y += 150
        ax.text(
            20,
            y + 4,
            f"AI 입력 전체 (숫자 {len(ids)}개)",
            fontsize=13,
            fontweight="bold",
            color=NAVY,
            va="top",
        )
        lines = textwrap.wrap("[" + ", ".join(str(n) for n in ids) + "]", 104)
        shown = lines[:8]
        if len(lines) > 8:
            n_shown = sum(len(re.findall(r"\d+", s)) for s in shown)
            shown.append(f"… (나머지 {len(ids) - n_shown}개 생략)")
        ax.text(
            20,
            y + 34,
            "\n".join(shown),
            fontsize=13,
            fontweight="bold",
            color=ORANGE,
            va="top",
            family="monospace",
            linespacing=1.45,
        )

    # ════════════════════════════════════════════════════════════
    # 4. 확률표 (세어서 구하기)
    # ════════════════════════════════════════════════════════════
    def s4(self):
        self.set_desc(f"""
■ Training: "이 token 다음엔 무엇이 오나" 세기
숫자 배열에서 이웃한 두 token (현재 → 다음) 쌍을 모두 셈. 그 횟수를 비율로 바꾸면 Probability Table가 됨
{count_example()}

■ 오른쪽 Matrix
전체 표는 {V}×{V} = {V * V:,}칸. 그림에는 많이 나온 token {len(SUB)}개만 잘라 표시
행 = 현재 token, 열 = 다음 token, 칸 = Probability. 대부분 비어 있음

★ Matrix의 아무 칸이나 클릭하면 그 행이 강조되고, 오른쪽에 Probability 막대 표시

실제 LLM은 이렇게 세지 않고, 숫자(Parameter)를 조금씩 고쳐 가며 Probability을 'Training'함. 다음 5단계(5-1~5-4 Training)에서 이 표를 Parameter로 직접 구함
""")
        self.draw_matrix()

    def draw_matrix(self):
        for ax in self.dyn:
            ax.remove()
        self.dyn = []
        names = [short(id_to_tok[i], 9) for i in SUB]
        n = len(names)
        M = COUNT_P[np.ix_(SUB, SUB)]
        ax = self.fig.add_axes(
            [RIGHT[0] + 0.055, RIGHT[1] + 0.012, 0.40, 0.66]
        )
        self.dyn.append(ax)
        self.ax_mat = ax
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), names, rotation=90, fontsize=10)
        ax.set_yticks(range(n), names, fontsize=10)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xticks([i - 0.5 for i in range(1, n)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(1, n)], minor=True)
        ax.grid(which="minor", color="#e3e3e3", lw=0.6)
        ax.tick_params(which="minor", length=0)
        for i in range(n):
            for j in range(n):
                if M[i][j] >= 0.01:
                    ax.text(
                        j,
                        i,
                        f"{M[i][j]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        fontweight="bold",
                        color="white" if M[i][j] > 0.5 else "#222",
                    )
        ax.set_xlabel(f"다음 토큰 (많이 나온 {n}개)", fontsize=11)
        ax.set_ylabel("현재 토큰", fontsize=11)
        if self.row_sel is not None:
            ax.add_patch(
                Rectangle(
                    (-0.5, self.row_sel - 0.5),
                    n,
                    1,
                    fill=False,
                    ec=ORANGE,
                    lw=2.5,
                )
            )
            lab = ax.get_yticklabels()[self.row_sel]
            lab.set_color(ORANGE)
            lab.set_fontweight("bold")

        side = self.fig.add_axes(
            [RIGHT[0] + 0.49, RIGHT[1] + 0.32, 0.14, 0.38]
        )
        self.dyn.append(side)
        side.spines[["top", "right"]].set_visible(False)
        if self.row_sel is None:
            side.axis("off")
            side.text(
                0.5,
                0.5,
                "← 행렬을 클릭하세요\n\n그 토큰 다음에 올\n토큰의 확률을\n보여 줍니다",
                ha="center",
                va="center",
                fontsize=14,
                color="#9aa5b1",
                fontweight="bold",
            )
            return
        tid = SUB[self.row_sel]
        t = id_to_tok[tid]
        nxt = sorted(model.get(tid, {}).items(), key=lambda x: -x[1])[:8]
        vals = [p for _, p in nxt]
        side.barh(
            range(len(vals)),
            vals,
            color=[ORANGE] + [TEAL] * (len(vals) - 1),
            height=0.6,
        )
        side.set_yticks(
            range(len(vals)),
            [short(id_to_tok[j], 10) for j, _ in nxt],
            fontsize=12,
        )
        side.invert_yaxis()
        side.set_xlim(0, 1.3)
        side.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        for k, v in enumerate(vals):
            side.text(
                v + 0.03,
                k,
                f"{v:.0%}",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
        side.set_title(
            f"'{t}' 다음에 올 토큰",
            fontsize=12.5,
            fontweight="bold",
            color=ORANGE,
            loc="left",
        )
        cnt = counts[tid]
        lines = [f"  {id_to_tok[j]} : {c}번" for j, c in cnt.most_common(8)]
        if len(cnt) > 8:
            lines.append(f"  … 외 {len(cnt) - 8}종류")
        side.text(
            0,
            -0.1,
            "원문에서 나온 횟수\n"
            + "\n".join(lines)
            + f"\n  합계 {sum(cnt.values())}번 → 비율 = 확률",
            transform=side.transAxes,
            va="top",
            fontsize=11,
            color="#555",
        )

    def on_click(self, e):
        if (
            self.cur == 4
            and e.inaxes is getattr(self, "ax_mat", None)
            and e.ydata is not None
        ):
            i = int(round(e.ydata))
            if 0 <= i < len(SUB):
                self.row_sel = i
                self.draw_matrix()
                self.fig.canvas.draw_idle()

    # ════════════════════════════════════════════════════════════
    # 5-1~5-4. 학습: 파라미터로 확률표 구하기
    # ════════════════════════════════════════════════════════════
    def heat(
        self,
        rect,
        M,
        title,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        fs=7,
        ylabel=True,
        annotate=False,
    ):
        """V×V 행렬 중 많이 나온 토큰(SUB)끼리만 잘라 그린다"""
        ax = self.fig.add_axes(rect)
        self.dyn.append(ax)
        S = M[np.ix_(SUB, SUB)]
        names = [short(id_to_tok[i]) for i in SUB]
        im = ax.imshow(S, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(SUB)), names, rotation=90, fontsize=fs)
        ax.set_yticks(
            range(len(SUB)), names if ylabel else [""] * len(SUB), fontsize=fs
        )
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        # 제목은 눈금 글자 위 고정 위치에 (긴 토큰 이름에 밀려 단추와 겹치지 않게)
        self.ftext(
            rect[0],
            rect[1] + rect[3] + 0.06,
            title,
            fontsize=12.5,
            fontweight="bold",
            color=NAVY,
        )
        if annotate:
            for i in range(len(SUB)):
                for j in range(len(SUB)):
                    if S[i][j] >= 0.1:
                        ax.text(
                            j,
                            i,
                            f"{S[i][j]:.0%}",
                            ha="center",
                            va="center",
                            fontsize=5,
                            color="white" if S[i][j] > 0.5 else "#222",
                        )
        return ax, im

    def reset_button(self, x=0.012, w=0.26, label=None):
        label = (
            label
            or f"↺ 파라미터 초기화  (지금 {self.net.updates}회 학습된 상태)"
        )

        def do(_e):
            self.stop_timer()
            self.net.reset()
            self.show(self.cur)

        self.teal_button(x, w, label, do)

    # ── 5. 초기화 ─────────────────────────────────────────────
    def param_cards(self):
        """5-1단계 왼쪽 아래: Parameters = Weights + Biases 를 단계별 카드로"""
        ax, W, H = self.board([LEFT[0] + 0.008, 0.125, LEFT[2] - 0.016, 0.335])
        cards = [
            (
                "①",
                "V (어휘 수)",
                f"{V}",
                "2단계 어휘사전의 서로 다른 token 수",
            ),
            (
                "②",
                "Weights (가중치 W)",
                f"V × V = {V} × {V} = {V * V:,}",
                f"현재 token {V} × 다음 token {V} — 칸마다 점수 1개",
            ),
            (
                "③",
                "Biases (바이어스 b)",
                f"V = {V}",
                "다음 token마다 기본 점수 1개 (입력 수와 무관)",
            ),
            (
                "④",
                "Parameters (파라미터)",
                f"{V * V:,} + {V} = {V * V + V:,}",
                "Weights + Biases — 학습으로 고치는 숫자 전체",
            ),
        ]
        fc, ec = "#e8f1f8", NAVY  # 네 카드 모두 같은 색
        gap = 14
        ch = (H - 3 * gap) / 4
        for k, (num, name, formula, why) in enumerate(cards):
            y = k * (ch + gap)
            last = False
            ax.add_patch(
                FancyBboxPatch(
                    (4, y),
                    W - 8,
                    ch,
                    boxstyle="round,pad=0,rounding_size=8",
                    fc=fc,
                    ec=ec,
                    lw=1.6,
                )
            )
            tc = "white" if last else "#1b2a3a"
            ax.text(
                18,
                y + ch * 0.32,
                f"{num} {name}",
                fontsize=12.5,
                fontweight="bold",
                color=SKY if last else ec,
                va="center",
            )
            ax.text(
                W - 18,
                y + ch * 0.36,
                formula,
                fontsize=17 if not last else 19,
                fontweight="bold",
                color=tc,
                va="center",
                ha="right",
            )
            ax.text(
                18,
                y + ch * 0.76,
                why,
                fontsize=10.5,
                color="#dfe7ef" if last else "#555",
                va="center",
            )
            if k < 3:
                self.arrow(
                    ax,
                    W / 2,
                    y + ch + 1,
                    W / 2,
                    y + ch + gap - 1,
                    "#8a96a3",
                    1.6,
                )

    def p_init(self):
        net = self.net
        ex = id_to_tok[EX]
        p_ex = softmax(net.W[EX] + net.b)
        top = np.argsort(-p_ex)[:3]
        right = ", ".join(id_to_tok[j] for j, _ in counts[EX].most_common(4))
        loss0 = (
            net.history[0] if net.history else net.forward()[2]
        )  # 학습 전(초기화 직후) 손실
        self.set_desc(
            f"""
■ parameter → 확률값 구하기

  · 어휘 수 V = {V}로 결정. {V} × {V} + {V} = {V * V + V:,}개 parameter

  · W 칸({V * V:,}개)에는 0 근처의 무작위 숫자를 넣음
  · b 칸({V}개)은 모두 0으로 둠
  · 즉 "아직 아무것도 모르는 상태"를 만드는 것
""",
            bottom=0.475,
            size=16,
        )
        self.param_cards()
        if net.updates:
            self.reset_button(0.012, 0.309)
        m = max(abs(net.W).max(), 1e-9)
        self.heat(
            [RIGHT[0] + 0.05, RIGHT[1] + 0.15, 0.36, 0.55],
            net.W,
            f"가중치 W ({V}×{V} 중 {len(SUB)}×{len(SUB)})  — 빨강 +, 파랑 −",
            cmap="RdBu_r",
            vmin=-m,
            vmax=m,
        )
        bx = self.fig.add_axes(
            [RIGHT[0] + 0.05, RIGHT[1] + 0.055, 0.36, 0.035]
        )
        self.dyn.append(bx)
        bx.imshow([net.b[SUB]], cmap="RdBu_r", vmin=-m, vmax=m, aspect="auto")
        bx.set_yticks([0], ["b"], fontsize=11, fontweight="bold")
        bx.set_xticks([])
        bx.tick_params(length=0)
        bx.set_xlabel("바이어스 b (처음엔 모두 0 = 흰색)", fontsize=10)
        side = self.fig.add_axes(
            [RIGHT[0] + 0.47, RIGHT[1] + 0.25, 0.155, 0.45]
        )
        self.dyn.append(side)
        order = np.argsort(-p_ex)[:8]
        side.barh(range(8), p_ex[order], color=TEAL, height=0.6)
        side.set_yticks(
            range(8), [short(id_to_tok[j], 10) for j in order], fontsize=11
        )
        side.invert_yaxis()
        side.set_xlim(0, max(0.05, p_ex.max() * 1.3))
        side.spines[["top", "right"]].set_visible(False)
        side.set_title(
            f"'{ex}' 다음 예측 (상위 8개)\nsoftmax(W['{ex}'] + b)",
            fontsize=12,
            fontweight="bold",
            color=NAVY,
            loc="left",
        )
        side.text(
            0,
            -0.14,
            f"처음 손실 L = {loss0:.3f}\n(모두 같은 확률일 때 ln V = {np.log(V):.3f})",
            transform=side.transAxes,
            va="top",
            fontsize=12,
            color="#333",
        )

    # ── 6. 순전파와 손실 ───────────────────────────────────────
    def flow_board(self, step, y0, h, big=False):
        """5-2·5-3·5-4 공통: 학습 한 바퀴 흐름. 지금 단계 줄은 주황, 오른쪽 숫자는 실제 값"""
        ax, W, H = self.board([LEFT[0] + 0.008, y0, LEFT[2] - 0.016, h])
        rows = [
            ("5-2", "parameters → 예측 → loss", "얼마나 틀렸나, 1개"),
            ("5-3", "gradient", f"어떻게 고칠까, {V * V + V:,}개"),
            ("5-4", "parameters ← parameters − η × gradient", "실제로 고침"),
            ("다시 5-2", "새 parameters로 loss 재기", "줄었나?"),
        ]
        arrows = ["↓ 미분", "↓ × η", "↓"]
        gap = 34 if big else 24
        rh = (H - 3 * gap) / 4
        f_tag, f_main, f_note, f_val, x_main = (
            (15, 16, 12.5, 15, 120) if big else (12.5, 12.5, 10, 12, 100)
        )
        self.flow_txt = {}
        for k, (tag, main, note) in enumerate(rows):
            y = k * (rh + gap)
            on = tag == step
            ax.add_patch(
                FancyBboxPatch(
                    (2, y),
                    W - 4,
                    rh,
                    boxstyle="round,pad=0,rounding_size=7",
                    fc="#fde3da" if on else "#f4f6f9",
                    ec=ORANGE if on else "#c5ced8",
                    lw=2 if on else 1,
                )
            )
            ax.text(
                14,
                y + rh / 2,
                tag,
                fontsize=f_tag,
                fontweight="bold",
                va="center",
                color=ORANGE if on else NAVY,
            )
            ax.text(
                x_main,
                y + rh * 0.36,
                main,
                fontsize=fit_fs(main, W - x_main - 10, f_main),
                fontweight="bold",
                va="center",
                color="#1b2a3a",
            )
            ax.text(
                x_main,
                y + rh * 0.74,
                note,
                fontsize=f_note,
                va="center",
                color="#666",
            )
            self.flow_txt[k] = ax.text(
                W - 14,
                y + rh * 0.74,
                "",
                fontsize=f_val,
                fontweight="bold",
                va="center",
                ha="right",
                color=ORANGE if on else NAVY,
            )
            if k < 3:
                ax.text(
                    x_main + 18,
                    y + rh + gap / 2,
                    arrows[k],
                    fontsize=f_note + 1,
                    fontweight="bold",
                    va="center",
                    color="#8a96a3",
                )
        self.update_flow()

    def update_flow(self):
        """흐름 상자 오른쪽의 실제 값 (5-4에서 학습하는 동안 계속 바뀜)"""
        if not getattr(self, "flow_txt", None):
            return
        dW, db, _, loss = self.net.grads()
        g = float(np.sqrt((dW**2).sum() + (db**2).sum()))
        first = self.net.history[0] if self.net.history else loss
        self.flow_txt[0].set_text(f"loss {loss:.3f}")
        self.flow_txt[1].set_text(f"‖gradient‖ {g:.4f}")
        self.flow_txt[2].set_text(f"update {self.net.updates}회")
        self.flow_txt[3].set_text(
            f"{first:.3f} → {loss:.3f}" if self.net.updates else ""
        )

    def p_forward(self):
        net = self.net
        Z, P, loss = net.forward()
        x0, y0 = X[0], Y[0]
        self.set_desc(
            f"""
■ Forward Propagation: 지금 Parameter로 다음 token 예측과 채점
  z = W[현재 token] + b  →  p = softmax(z)
  Loss L = −log p(정답) 의 평균 ({N}쌍)
""",
            bottom=0.635,
            size=16,
        )
        self.flow_board("5-2", 0.125, 0.49, big=True)
        ax, W, H = self.board(
            [
                RIGHT[0] + 0.006,
                RIGHT[1] + RIGHT[3] - 0.12,
                RIGHT[2] - 0.012,
                0.095,
            ]
        )
        cards = [
            ("현재 손실 L", f"{loss:.3f}", ORANGE),
            ("무작위로 찍을 때 ln V", f"{np.log(V):.3f}", "#8a96a3"),
            ("완벽히 배웠을 때 (최소)", f"{MIN_LOSS:.3f}", TEAL),
        ]
        cw = (W - 40) / 3
        for i, (k, v, col) in enumerate(cards):
            ax.add_patch(
                FancyBboxPatch(
                    (20 + i * cw, 4),
                    cw - 16,
                    H - 8,
                    boxstyle="round,pad=0,rounding_size=8",
                    fc=col,
                    ec="none",
                )
            )
            ax.text(
                20 + i * cw + (cw - 16) / 2,
                H * 0.4,
                v,
                fontsize=24,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
            )
            ax.text(
                20 + i * cw + (cw - 16) / 2,
                H * 0.8,
                k,
                fontsize=11,
                color="white",
                ha="center",
                va="center",
            )
        self.heat(
            [RIGHT[0] + 0.05, RIGHT[1] + 0.02, 0.27, 0.55],
            net.table(),
            "예측: softmax(W + b)",
            annotate=True,
        )
        self.heat(
            [RIGHT[0] + 0.36, RIGHT[1] + 0.02, 0.27, 0.55],
            COUNT_P,
            "목표: 4단계 확률표",
            annotate=True,
            ylabel=False,
        )
        if net.updates:
            self.reset_button()

    # ── 7. 역전파 ──────────────────────────────────────────────
    def p_backward(self):
        net = self.net
        dW, db, dZ, loss = net.grads()
        r, e = EX, EX_NEXT
        other = int(np.argmax(np.where(COUNT_P[r] == 0, dW[r], -np.inf)))
        self.set_desc(
            """
■ Backpropagation: Loss을 줄이려면 Parameter 하나하나를 어느 쪽으로 고칠지 계산
  파랑(−) = 올려야 함,  빨강(+) = 내려야 함
""",
            bottom=0.665,
            size=16,
        )
        self.flow_board("5-3", 0.125, 0.52, big=True)
        m = max(abs(dW[np.ix_(SUB, SUB)]).max(), 1e-9)
        self.heat(
            [RIGHT[0] + 0.05, RIGHT[1] + 0.06, 0.40, 0.63],
            dW,
            f"기울기 ∂L/∂W ({V}×{V} 중 {len(SUB)}×{len(SUB)})",
            cmap="RdBu_r",
            vmin=-m,
            vmax=m,
        )
        side = self.fig.add_axes(
            [RIGHT[0] + 0.50, RIGHT[1] + 0.06, 0.125, 0.63]
        )
        self.dyn.append(side)
        dbs = db[SUB]
        side.barh(
            range(len(SUB)),
            dbs,
            color=[TEAL if v < 0 else ORANGE for v in dbs],
            height=0.7,
        )
        side.set_yticks(
            range(len(SUB)), [short(id_to_tok[i]) for i in SUB], fontsize=8
        )
        side.invert_yaxis()
        side.axvline(0, color="#666", lw=0.8)
        side.spines[["top", "right"]].set_visible(False)
        side.tick_params(axis="x", labelsize=8)
        self.ftext(
            RIGHT[0] + 0.50,
            RIGHT[1] + 0.06 + 0.63 + 0.06,
            "기울기 ∂L/∂b",
            fontsize=12.5,
            fontweight="bold",
            color=NAVY,
        )
        if net.updates:
            self.reset_button()

    # ── 8. 업데이트와 반복 ─────────────────────────────────────
    def p_update(self):
        lr, opt = TRAIN["lr"], TRAIN["opt"]
        rule = {
            "SGD": "W ← W − η · ∂L/∂W",
            "Momentum": "v ← 0.9·v + ∂L/∂W,   W ← W − η · v",
            "Adam": "W ← W − (η/100) · m / √s",
        }[opt]
        self.set_desc(
            f"""
■ update: {rule}
""",
            bottom=0.72,
        )
        self.flow_board("5-4", 0.27, 0.44, big=True)
        self.left = 0

        def run(n):
            def go(_e):
                self.stop_timer()
                if n == 1:
                    self.net.step()
                    self.update_train()
                    return
                self.left = n
                self.timer = self.fig.canvas.new_timer(interval=30)
                self.timer.add_callback(self.tick)
                self.timer.start()

            return go

        def restart(lr=None, opt=None):
            def go(_e):
                self.stop_timer()
                if self.net.history:  # 지금 곡선은 회색으로 남긴다
                    self.past.append(
                        (
                            f"{TRAIN['opt']} η={TRAIN['lr']:g}",
                            list(self.net.history),
                        )
                    )
                    self.past = self.past[-4:]
                if lr is not None:
                    TRAIN["lr"] = lr
                if opt is not None:
                    TRAIN["opt"] = opt
                self.net.reset()
                self.show(self.cur)

            return go

        # 단추 줄 (위 → 아래):  학습률 η  ·  옵티마이저  ·  학습 횟수와 초기화
        h, y_lr, y_opt, y_run = 0.042, 0.165, 0.11, 0.055
        for y, title in ((y_lr, "학습률 η"), (y_opt, "옵티마이저")):
            self.ftext(
                LEFT[0] + 0.012,
                y + h / 2,
                title,
                va="center",
                fontsize=12.5,
                fontweight="bold",
                color=NAVY,
            )
        for i, v in enumerate(LRS):
            self.choice_button(
                0.09 + i * 0.059,
                0.055,
                y_lr,
                h,
                f"{v:g}",
                v == lr,
                restart(lr=v),
            )
        for i, (key, label) in enumerate(OPTIMIZERS):
            self.choice_button(
                0.09 + i * 0.079,
                0.075,
                y_opt,
                h,
                label,
                key == opt,
                restart(opt=key),
            )
        for i, (lab, n) in enumerate(
            (("1회", 1), ("10회 ▶", 10), ("100회 ▶", 100), ("500회 ▶", 500))
        ):
            self.teal_button(0.012 + i * 0.062, 0.056, lab, run(n), y=y_run)
        self.teal_button(
            0.012 + 4 * 0.062, 0.07, "↺ 초기화", restart(), y=y_run
        )
        self.train_info = self.ftext(
            LEFT[0] + 0.012,
            0.215,
            "",
            fontsize=12,
            color=NAVY,
            fontweight="bold",
            va="bottom",
        )

        self.loss_ax = self.fig.add_axes(
            [RIGHT[0] + 0.05, RIGHT[1] + 0.6, 0.57, 0.17]
        )
        self.dyn.append(self.loss_ax)
        a = self.loss_ax
        a.axhline(MIN_LOSS, color=TEAL, ls="--", lw=1.2)
        a.text(
            0.99,
            MIN_LOSS + 0.08,
            f"최소 손실 {MIN_LOSS:.3f}",
            color=TEAL,
            fontsize=10,
            va="bottom",
            ha="right",
            transform=a.get_yaxis_transform(),
        )
        greys = ("#d0d5db", "#bcc3cb", "#a6afb9", "#8f9aa6")
        for k, (lab, hist) in enumerate(self.past):  # 이전 설정의 곡선
            col = greys[k - len(self.past) + len(greys)]
            a.plot(
                range(1, len(hist) + 1),
                hist,
                color=col,
                lw=1.4,
                label=f"{lab} → {hist[-1]:.3f}",
            )
        (self.loss_line,) = a.plot(
            [], [], color=ORANGE, lw=2.2, label=f"{opt} η={lr:g}"
        )
        a.legend(loc="upper right", fontsize=10, frameon=False, ncol=2)
        self.loss_top = max(
            [np.log(V) + 0.8]
            + [min(max(hh), 15) * 1.05 for _, hh in self.past]
        )
        a.set_ylim(0, self.loss_top)
        a.set_ylabel("손실 L", fontsize=11)
        a.set_xlabel("업데이트 횟수", fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
        a.set_title(
            "손실 곡선",
            fontsize=12.5,
            fontweight="bold",
            color=NAVY,
            loc="left",
        )
        _, self.imQ = self.heat(
            [RIGHT[0] + 0.05, RIGHT[1] + 0.02, 0.27, 0.43],
            self.net.table(),
            "예측: softmax(W + b)",
        )
        m = max(abs(self.net.W).max(), 1e-9)
        _, self.imW = self.heat(
            [RIGHT[0] + 0.36, RIGHT[1] + 0.02, 0.27, 0.43],
            self.net.W,
            "가중치 W",
            cmap="RdBu_r",
            vmin=-m,
            vmax=m,
            ylabel=False,
        )
        self.update_train(draw=False)

    def choice_button(self, x, w, y, h, label, on, fn):
        """여러 개 중 하나를 고르는 단추 (고른 것은 주황)"""
        b = Button(
            self.widget_ax(x, w, y, h),
            label,
            color=ORANGE if on else "#e8f1f8",
            hovercolor="#d65a3a" if on else "#cfe0ee",
        )
        b.label.set_fontsize(12.5)
        b.label.set_fontweight("bold")
        b.label.set_color("white" if on else NAVY)
        b.on_clicked(fn)
        self.widgets.append(b)

    def update_train(self, draw=True):
        net = self.net
        _, _, loss = net.forward()
        h = net.history
        self.loss_line.set_data(range(1, len(h) + 1), h)
        longest = max([len(h)] + [len(hh) for _, hh in self.past])
        self.loss_ax.set_xlim(0, max(10, longest))
        if h and max(h) * 1.05 > self.loss_top:  # 발산하면 위쪽을 넓힌다
            self.loss_top = min(max(h), 15) * 1.05
            self.loss_ax.set_ylim(0, self.loss_top)
        T = net.table()
        self.imQ.set_data(T[np.ix_(SUB, SUB)])
        m = max(abs(net.W).max(), 1e-9)
        self.imW.set_data(net.W[np.ix_(SUB, SUB)])
        self.imW.set_clim(-m, m)
        r, e = EX, EX_NEXT
        self.update_flow()
        self.train_info.set_text(
            f"update {net.updates}회 · loss {loss:.3f} (최소 {MIN_LOSS:.3f})\n"
            f"예) p({id_to_tok[e]} | {id_to_tok[r]}) = {T[r, e]:.2f}   목표 {COUNT_P[r, e]:.2f}"
        )
        if draw:
            self.fig.canvas.draw_idle()

    def tick(self):
        k = 1 if self.left <= 10 else 5
        for _ in range(min(k, self.left)):
            self.net.step()
            self.left -= 1
        self.update_train()
        if self.left <= 0:
            self.stop_timer()

    # ════════════════════════════════════════════════════════════
    # 9. 생성
    # ════════════════════════════════════════════════════════════
    def s5(self):
        sw = split_words()
        piece = " | ".join(sw[0][1]) if sw else "·Ab | rax | as"
        net = self.net
        table = net.table()
        _, _, loss = net.forward()
        forks = int((table.max(axis=1) < 0.9).sum())
        if net.updates == 0:
            state = "아직 한 번도 Training하지 않은 무작위 Parameter. 그래서 엉터리 token이 뒤죽박죽 나옴"
        elif loss > MIN_LOSS + 0.5:
            state = "아직 덜 Training해서 자주 나온 token 위주로 어색한 문장이 나옴"
        else:
            state = "충분히 Training해서 4단계 Probability Table와 거의 같고, 원문다운 문장이 나옴"
        note = (
            f"""
※ 갈림길(가장 높은 Probability이 90% 미만인 token)이 {forks}개뿐 (전체 {V}개 중). 같은 단어가 거의 되풀이되지 않는 원문이라 Model이 문장을 통째로 외웠고(Overfitting), 세 번 시도가 비슷하게 나옴
"""
            if net.updates and forks < V * 0.25
            else ""
        )
        self.set_desc(
            f"""
■ Generation: Training한 Parameter로 다음 token을 하나씩 뽑기
p(token Probability) = softmax(W[현재 token] + b) 로 다음 token Probability 구하고, 뽑은 token에서 다시 반복

■ 지금 Parameter: update {net.updates}회 Training (Loss {loss:.3f}, 최소 {MIN_LOSS:.3f})
{state}
아래 [+10회] [+100회] [+500회] 로 더 Training하거나 [↺ 0회]로 되돌려 결과 비교

■ 오른쪽 그림
화살표 위 숫자 = 그 token이 뽑힌 Probability.
주황 = 확실하지 않았던 선택(갈림길)
색 반전 = 채택된 시도 (평균 token Probability이 가장 높은 문장)
{note}
시작 단어를 쓰고 Enter
""",
            bottom=0.165,
            size=15.5,
        )

        def train(n):
            def go(_e):
                if n == 0:
                    self.net.reset()
                else:
                    for _ in range(n):
                        self.net.step()
                self.show(GEN_STEP)

            return go

        for i, (lab, n) in enumerate(
            (
                ("+10회 학습", 10),
                ("+100회", 100),
                ("+500회", 500),
                ("↺ 0회", 0),
            )
        ):
            self.teal_button(
                0.012 + i * 0.07,
                0.065,
                lab,
                train(n),
                y=0.11,
                color=NAVY if n else "#8a96a3",
                hover="#2b4a75" if n else "#7a8693",
            )

        def regen(_=None):
            w = get().strip()
            tok = find_token(w)
            if tok is None:
                self.alert(
                    f"'{w}'은(는) 지금 원문에 없는 단어예요.  "
                    f"원문에 있는 단어로 해 보세요: {', '.join(word_examples())}"
                )
                return
            self.start_word = tok
            self.seed += 3
            self.show(GEN_STEP)

        get, _ = self.text_input(
            0.075, 0.1, join_text(self.start_word), regen, label="시작 단어"
        )
        self.teal_button(0.185, 0.1, "⟳ 다시 생성", regen)

        h = RIGHT[3] / 3
        runs = []
        for a in range(3):
            trace = []
            pre = self.start_word[
                :-1
            ]  # 시작 단어가 여러 조각이면 앞 조각은 그대로
            words = pre + generate(
                self.start_word[-1], table, seed=self.seed + a, trace=trace
            )
            trace = [None] * len(pre) + trace
            # 채택 기준: 뽑힌 토큰들의 평균 log 확률 (문장이 얼마나 그럴듯한가)
            score = float(
                np.mean([np.log(t[3]) for t in trace if t is not None])
            )
            runs.append((words, trace, score))
        best = max(range(3), key=lambda a: runs[a][2])
        for a, (words, trace, score) in enumerate(runs):
            self.draw_chain(
                [
                    RIGHT[0] + 0.006,
                    RIGHT[1] + RIGHT[3] - (a + 1) * h + 0.006,
                    RIGHT[2] - 0.012,
                    h - 0.016,
                ],
                f"시도 {a + 1}" + ("  ★ 채택" if a == best else ""),
                words,
                trace,
                picked=a == best,
                score=score,
            )

    def draw_chain(
        self, rect, label, words, trace, per_row=8, picked=False, score=None
    ):
        """picked=True 이면 채택된 시도: 머리 줄 색을 반전 (남색 바탕 + 흰 글자)"""
        ax, W, H = self.board(rect)
        ax.add_patch(
            FancyBboxPatch(
                (2, 2),
                W - 4,
                H - 4,
                boxstyle="round,pad=0,rounding_size=8",
                fc="#f8fafc",
                ec=ORANGE if picked else "#dbe3ea",
                lw=2.4 if picked else 1.2,
            )
        )
        if picked:  # 머리 줄 색 반전
            ax.add_patch(
                FancyBboxPatch(
                    (6, 6),
                    W - 12,
                    30,
                    boxstyle="round,pad=0,rounding_size=6",
                    fc=NAVY,
                    ec="none",
                )
            )
        fg = "white" if picked else NAVY
        ax.text(
            18,
            21,
            label,
            fontsize=13,
            fontweight="bold",
            color=SKY if picked else NAVY,
            va="center",
        )
        if score is not None:
            ax.text(
                150,
                21,
                f"평균 토큰 확률 {np.exp(score):.0%}",
                fontsize=10.5,
                color=SKY if picked else "#8a96a3",
                va="center",
            )
        sentence = "⇒ " + join_text(words)
        ax.text(
            W - 18,
            21,
            sentence,
            fontsize=fit_fs(sentence, W - 330, 13),
            style="italic",
            family="serif",
            color=fg,
            fontweight="bold" if picked else "normal",
            va="center",
            ha="right",
        )
        cw = (W - 40) / per_row
        chh = 40
        rows = (len(words) - 1) // per_row + 1
        gap = (H - 50 - rows * chh) / max(rows, 1)
        for k, w in enumerate(words):
            r, c = divmod(k, per_row)
            x = 20 + c * cw
            y = 44 + r * (chh + gap) + gap / 2
            self.tok_chip(ax, x, y, cw - 52, chh, w, fs=13)
            if (
                k < len(trace) and trace[k] is None
            ):  # 입력한 단어의 조각끼리는 회색 선만
                if c < per_row - 1:
                    self.arrow(
                        ax,
                        x + cw - 50,
                        y + chh / 2,
                        x + cw - 2,
                        y + chh / 2,
                        "#bbbbbb",
                        1.2,
                    )
            elif k < len(trace):
                _, cands, ch, p = trace[k]
                fork = p < 0.9  # 확실하지 않았던 선택 = 갈림길
                col = ORANGE if fork else "#999999"
                if c < per_row - 1:
                    self.arrow(
                        ax,
                        x + cw - 50,
                        y + chh / 2,
                        x + cw - 2,
                        y + chh / 2,
                        col,
                        2.4 if fork else 1.6,
                    )
                    ax.text(
                        x + cw - 26,
                        y + chh / 2 - 9,
                        f"{p:.0%}",
                        ha="center",
                        fontsize=9.5,
                        fontweight="bold",
                        color=col,
                    )
                else:
                    ax.text(
                        x + cw - 26,
                        y + chh / 2,
                        f"↵{p:.0%}",
                        ha="center",
                        va="center",
                        fontsize=9.5,
                        fontweight="bold",
                        color=col,
                    )

    # ════════════════════════════════════════════════════════════
    # 10. 대화
    # ════════════════════════════════════════════════════════════
    def s6(self):
        self.set_desc(f"""
■ Model과 대화하기
아래 입력칸을 클릭해 영어를 쓰고 Enter 또는 [보내기] 누르기
  · 문장 → GPT Tokenizer로 Tokenization·Encoding을 거친 뒤, 마지막 token부터 이어 씀
  · ? 단어 → 그 단어(마지막 조각) 다음에 올 Probability (예: ? the)
  · 단어목록 → Model이 아는 token {V}개

■ 알아 둘 점 ①
GPT Tokenizer는 어떤 영어든 자를 수 있지만, 우리 Model은 원문에 나온 token {V}개만 배움. 처음 보는 token은 '?'가 되어 무시됨
  예) the bird → ·the | ·bird (둘 다 앎)
      the cat  → ·the | ·cat (cat은 모름 → ?)

■ 알아 둘 점 ②
이 Model은 질문에 '답'하지 않고 '이어 쓰기'만 함

■ 지금 Model: update {self.net.updates}회 Training한 Parameter
Training이 부족하면 엉터리로 이어 씀. 6단계나 5-4단계에서 더 Training 필요
""")

        def send(_=None):
            q = get().strip()
            if q:
                put("")
                self.chat(q)

        get, put = self.text_input(0.012, 0.22, "", send, justify="left")
        self.teal_button(0.24, 0.08, "보내기", send)
        ax, W, H = self.board()
        ax.text(
            W / 2,
            H / 2,
            "왼쪽 아래 입력칸에 영어로 입력해 보세요\n\n"
            "예)   the bird      the cat      ? the      단어목록",
            ha="center",
            va="center",
            fontsize=17,
            color="#9aa5b1",
            fontweight="bold",
        )

    def chat(self, q):
        for a in self.dyn:
            a.remove()
        self.dyn = []
        if q == "단어목록":
            ax, W, H = self.board()
            ax.text(
                20,
                6,
                f"모델이 아는 토큰 {V}개 (왼쪽 위 숫자 = 어휘사전 ID)",
                fontsize=15,
                fontweight="bold",
                color=NAVY,
                va="top",
            )
            self.legend(ax, W - 640, 8, fs=11)
            cols = 12
            rows = (V - 1) // cols + 1
            rh = min(56, (H - 50) / rows)
            cw = (W - 40) / cols
            for k, t in enumerate(vocab):
                r, c = divmod(k, cols)
                self.tok_chip(
                    ax,
                    20 + c * cw,
                    44 + r * rh,
                    cw - 6,
                    rh - 5,
                    t,
                    sub=str(vocab[t]) if rh >= 40 else None,
                    fs=12 if rh >= 40 else 9,
                )
        elif q.startswith("?"):
            w = q[1:].strip()
            seq = find_token(w)
            word = seq[-1] if seq else w
            if word not in vocab:
                ax, W, H = self.board()
                ax.text(
                    W / 2,
                    H / 2,
                    f"'{q[1:].strip()}' — 배운 적 없는 토큰이에요",
                    fontsize=20,
                    ha="center",
                    color=ORANGE,
                    fontweight="bold",
                )
            else:
                ax = self.fig.add_axes(
                    [
                        RIGHT[0] + 0.08,
                        RIGHT[1] + 0.1,
                        RIGHT[2] - 0.12,
                        RIGHT[3] - 0.2,
                    ]
                )
                self.dyn.append(ax)
                row = self.net.table()[vocab[word]]  # 학습한 파라미터의 확률
                nxt = [(j, float(row[j])) for j in np.argsort(-row)[:15]]
                vals = [p for _, p in nxt]
                ax.barh(
                    range(len(vals)),
                    vals,
                    color=[ORANGE] + [TEAL] * (len(vals) - 1),
                    height=0.55,
                )
                ax.set_yticks(
                    range(len(vals)),
                    [id_to_tok[j] for j, _ in nxt],
                    fontsize=14,
                )
                ax.invert_yaxis()
                ax.set_xlim(0, 1.1)
                for k, v in enumerate(vals):
                    ax.text(
                        v + 0.01,
                        k,
                        f"{v:.0%}",
                        va="center",
                        fontsize=13,
                        fontweight="bold",
                    )
                ax.set_title(
                    f"'{word}' 다음에 올 확률 (상위 {len(vals)}개, update {self.net.updates}회 학습한 모델)",
                    fontsize=16,
                    fontweight="bold",
                    color=NAVY,
                    loc="left",
                )
                ax.spines[["top", "right"]].set_visible(False)
        else:
            self.chat_sentence(q)
        self.fig.canvas.draw_idle()

    def chat_sentence(self, q):
        pairs = tokenize(" " + q)  # 글 중간처럼 앞에 띄어쓰기를 붙여 자른다
        q_tokens = [t for t, _ in pairs]
        known = [t for t in q_tokens if t in vocab]
        top_h = 0.19
        ax, W, H = self.board(
            [
                RIGHT[0] + 0.006,
                RIGHT[1] + RIGHT[3] - top_h - 0.02,
                RIGHT[2] - 0.012,
                top_h,
            ]
        )
        ax.text(
            14,
            6,
            f"입력 '{q}'  →  GPT 토큰화 (위: GPT 번호)  →  우리 어휘 번호",
            fontsize=13,
            fontweight="bold",
            color=NAVY,
            va="top",
        )
        shown = q_tokens[:14]
        cw = min(140, (W - 30) / max(1, len(shown)))
        for k, t in enumerate(shown):
            x = 14 + k * cw
            ok = t in vocab
            gid = None if pairs[k][1] is None else str(pairs[k][1])
            if ok:
                self.tok_chip(ax, x, 38, cw - 10, 46, t, sub=gid, fs=14)
            else:
                self.chip(
                    ax,
                    x,
                    38,
                    cw - 10,
                    46,
                    t,
                    sub=gid,
                    fc="#ffffff",
                    ec=ORANGE,
                    fs=14,
                )
            self.arrow(ax, x + cw / 2 - 5, 86, x + cw / 2 - 5, 108)
            self.chip(
                ax,
                x,
                110,
                cw - 10,
                46,
                str(vocab[t]) if ok else "?",
                fc=NAVY if ok else ORANGE,
                ec=NAVY if ok else ORANGE,
                tc="white",
                fs=16,
            )
        if len(q_tokens) > len(shown):
            ax.text(
                W - 10,
                60,
                f"… +{len(q_tokens) - len(shown)}",
                ha="right",
                fontsize=12,
            )
        if not q_tokens:
            ax.text(
                14,
                60,
                "토큰이 없어요 (영어 알파벳으로 입력하세요)",
                fontsize=14,
                color=ORANGE,
            )
        if not known:
            ax2, W2, H2 = self.board(
                [RIGHT[0] + 0.006, RIGHT[1] + 0.02, RIGHT[2] - 0.012, 0.4]
            )
            ax2.text(
                W2 / 2,
                H2 / 2,
                "아는 토큰이 없어요 — '단어목록'을 입력해 보세요",
                fontsize=18,
                ha="center",
                color=ORANGE,
                fontweight="bold",
            )
            return
        trace = []
        words = generate(
            known[-1],
            self.net.table(),
            length=12,
            seed=random.randrange(10_000),
            trace=trace,
        )
        answer = join_text(
            q_tokens + words[1:]
        )  # 입력 그대로 + 모델이 이어 쓴 부분
        self.draw_chain(
            [
                RIGHT[0] + 0.006,
                RIGHT[1] + 0.13,
                RIGHT[2] - 0.012,
                RIGHT[3] - top_h - 0.17,
            ],
            f"'{known[-1]}'에서 이어 쓰기",
            words,
            trace,
        )
        ax3, W3, H3 = self.board(
            [RIGHT[0] + 0.006, RIGHT[1] + 0.012, RIGHT[2] - 0.012, 0.1]
        )
        ax3.add_patch(
            FancyBboxPatch(
                (2, 2),
                W3 - 4,
                H3 - 4,
                boxstyle="round,pad=0,rounding_size=8",
                fc=NAVY,
                ec="none",
            )
        )
        # '모델:'(한글)은 기본 한글 글꼴로, 영어 답은 이탤릭 세리프로 따로 쓴다.
        # 맑은 고딕에는 이탤릭이 없어서, 한글을 이탤릭으로 쓰면 글자가 □로 깨진다.
        ax3.text(
            24,
            H3 / 2,
            "모델:",
            fontsize=17,
            fontweight="bold",
            color=SKY,
            va="center",
        )
        ax3.text(
            110,
            H3 / 2,
            answer,
            fontsize=fit_fs(answer, W3 - 140, 17),
            style="italic",
            family="serif",
            color="white",
            va="center",
        )


def main():
    app = Dashboard()
    app.fig._app = app  # 단추가 사라지지 않도록 참조 유지
    app.setup_window()
    plt.show()


if __name__ == "__main__":
    main()
