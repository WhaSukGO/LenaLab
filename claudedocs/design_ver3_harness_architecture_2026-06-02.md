# ver3 Harness Architecture — Visual Odometry Research Lab

- 날짜: 2026-06-02
- 입력: `research_cv_research_lab_design_2026-06-02.md`(확정 결정) + `blueberry_ver2`(Touchstone) 실제 소스의 인터페이스 실측.
- 산출물: 아키텍처 문서(컴포넌트/시퀀스/상태 다이어그램 + 인터페이스 스펙). **구현 코드는 `/sc:implement`에서.**
- 확정 전제: ver2 = **library import**(검증기 재구현 금지), oracle = **reproduction-first**, 도메인 = **Learned VO (DPVO/DROID 계열)**, 데이터 = **실데이터 우선 + 합성 fallback**(held-out·시드는 검증기 소유), 단일 RTX 3080 16GB.

---

## 0. 가장 중요한 설계 발견 — "ver3는 새 harness가 아니라 plugin이다"

ver2의 척추(`Harness`)는 **도메인 무관(domain-agnostic)**이다. CIFAR도, 코드 implementer도 전부 세 개의 Protocol 뒤에 꽂힌 plugin일 뿐이다:

```
plugins/base.py:  Planner · Evaluator · DatasetProviderP · MetricExtractor · Oracle
```

`loop.py`의 `Harness`는 이 Protocol에만 의존한다. 따라서 **VO 랩 = 같은 척추에 꽂는 새 plugin 세트 + 전문가 프롬프트 + 이미지 1행 + 얇은 factory.** ver3가 *새로 쓰는 줄 수는 작고*, 그 작음이 "import, don't rebuild" 결정이 옳았다는 증거다. ver3는 더 똑똑한 **generator(솔버)** 를 만들 뿐, 검증 척추는 ver2 것을 그대로 쓴다.

ver3가 작성하는 것 (전부 솔버측):
1. **VO 도메인 plugin** — `VODatasetProvider`(실데이터/합성), `VOMetricExtractor`, reference 코드(`vo_ref/`), VO oracle, held-out 시퀀스 분할.
2. **VO 평가 스크립트** — held-out 시퀀스에서 `evo`로 ATE/RPE 측정 → `heldout.json`. **harness 소유**(솔버가 못 건드림).
3. **연구팀(committee) 전문가** — PI + Geometry/SLAM + Modeling + Data 전문가의 프롬프트. ver2 `Committee`/`Implementer` seam에 그대로 꽂음.
4. **CUDA 이미지 1행** — `images/registry.yaml`에 VO 스택 prebuilt 이미지.
5. **VO reproduction calibration** — 알려진 VO 결과를 positive/negative control로.
6. **얇은 factory** — `build_vo_harness(...)` (ver2 `build_implementer_harness` 패턴 미러).

ver3가 **절대 작성하지 않는 것:** `loop.py`, `evaluator.py`(ScriptEvaluator 척추), `registry`, `budget`, `gpu_lease`, `image_registry`, `dataset_cache`, `job_runner`, `exchange`(VerifiedResult). 전부 ver2 import.

---

## 1. 컴포넌트 아키텍처

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ver3 = vo_lab/  (SOLVER layer — ver3가 작성)                                  │
│                                                                              │
│  agents/ (the "meeting" = 파일/contract 협상, 채팅 아님)                      │
│   ├─ vo_committee.py   PI + Geometry/SLAM + Modeling + Data 전문가            │
│   │                    → Track A: 메뉴 recipe 선택·파라미터 (재현/탐색)        │
│   └─ vo_implementer.py → Track B: VO 알고리즘 코드 authoring (novel 구현)     │
│                          (ver2 Implementer + sdk_author + sandbox_tool 재사용) │
│                                                                              │
│  plugins/vo.py  (도메인 — 솔버측이지만 oracle/eval은 harness 소유)            │
│   ├─ VODatasetProvider   실데이터(KITTI/TartanAir) 우선 + 합성 fallback        │
│   ├─ VOMetricExtractor   generator 보고치 추출(기록만, 불신)                  │
│   ├─ vo_recipe()/vo_menu()   Track A 메뉴(clamped params, 고정 oracle bar)    │
│   └─ vo_calibration_records()  reproduction positive/negative control         │
│                                                                              │
│  plugins/vo_ref/   reference 코드 (harness 소유 영역)                         │
│   ├─ run.py    VO 추론/학습 진입점 → /artifacts 에 trajectory/checkpoint       │
│   └─ eval.py   ★ HELD-OUT 시퀀스에서 evo ATE/RPE → eval/heldout.json           │
│                (솔버는 절대 작성/수정 불가 — anti-tamper로 평가직전 재기록)     │
│                                                                              │
│  build_vo_harness(...)   얇은 factory (ver2 패턴 미러)                        │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                 │ imports (검증 척추 재사용, 0줄 재구현)
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ ver2 = lab/  (VERIFIER spine — 불변, import만)                                │
│  loop.Harness  state machine + autonomy/calibration gate + 자율 lineage       │
│  evaluator.ScriptEvaluator  독립·회의적, held-out 측정, upgrade 금지           │
│  models.*  ExperimentContract · VerifiedResult · Criterion · DatasetRef(held_out)│
│  image_registry · dataset_cache · job_runner · gpu_lease · budget · registry  │
│  exchange.*  서명된 VerifiedResult 신뢰 게이트 + peer_review (multi-lab 대비)  │
│  agents/  Committee · Implementer · sdk_author · sandbox_tool (seam 재사용)    │
└────────────────────────────────────────────────────────────────────────────┘
```

핵심 경계: **솔버는 train 시퀀스(이미지)만 본다. held-out 시퀀스의 ground-truth 포즈, 합성 시드, `eval.py`는 evaluator/harness만 소유.** 이게 당신의 이전 실패("실행=성공", 자기평가)와 VO 특유의 gaming(쉬운 데이터 합성, scale 정렬 조작)을 동시에 막는다.

---

## 2. VO를 ver2 계약에 매핑 (이게 설계의 심장)

ver2의 검증은 한 가지 규칙으로 돌아간다: **job이 `eval/heldout.json`에 metric dict를 쓰면, `ScriptEvaluator`가 `Criterion.metric`을 꺼내 `Criterion.satisfied()`로 채점한다.** VO를 여기에 정확히 맞춘다.

### 2.1 데이터/포즈 흐름과 "held-out"의 의미
- **train split (솔버 가시):** 시퀀스 이미지 + (학습형이면) 일부 GT. `DatasetRef(held_out=False)` → 컨테이너에 `/data`로 마운트.
- **held-out split (evaluator 전용):** *서로소 시퀀스*의 이미지 + **GT 궤적**. `DatasetRef(held_out=True)` → loop은 절대 마운트하지 않고(`loop.py`가 held_out=False만 `data_dir`로 세팅), `ScriptEvaluator._heldout_dir()`만 마운트.
  - 표준 분할 사용으로 gaming 차단: 예) KITTI odometry `00–08` train / `09–10` held-out (관례), 또는 TartanAir 장면 분리.
- **합성 fallback:** `VODatasetProvider`가 렌더 궤적 생성 → **정의상 GT 포즈를 안다**(5.3 synthetic oracle). 단 **생성 시드와 held-out 장면은 harness가 소유**, 솔버에는 train 장면만 노출.

### 2.2 metric / oracle — silent-wrong(정렬 함정) 차단
- metric = **ATE-RMSE**(주), 보조 **RPE**. `Criterion(metric="ate_rmse", op="<=", value=THRESH, tolerance=τ)`.
- **측정은 솔버가 아니라 harness 소유 `eval.py`가 표준 도구 `evo`로 수행.** monocular scale 모호성 → 정렬 정책을 **eval.py에 고정**(예: `evo_ape --align --correct_scale` = Sim(3) Umeyama). 솔버가 정렬을 고를 수 없게 한다. → 3.1 silent-wrong 방지.
- `eval.py`는 held-out 각 시퀀스 ATE를 구해 평균/최댓값을 `heldout.json`에 기록:
  ```json
  {"ate_rmse": 0.184, "rpe_trans": 0.021, "per_seq": {"09": 0.17, "10": 0.20}, "align": "sim3"}
  ```

### 2.3 두 실행 모델 — 어느 쪽이든 같은 척추
| | 산출물(artifact) | command (솔버 가시) | eval_command (harness 소유) |
|---|---|---|---|
| **Learned VO** (CIFAR 패턴) | `/artifacts/ckpt.pth` | train.py: train 시퀀스로 학습 → checkpoint | eval.py: held-out 시퀀스에 추론 → 궤적 → evo ATE |
| **Classical/authored VO** (Implementer 패턴) | `/artifacts/traj_*.txt` | main.py(솔버 authored): VO 알고리즘 | eval.py: held-out 시퀀스에 *authored 코드* 재실행 → evo ATE |

→ Learned VO는 `vo_recipe()`(메뉴, Track A)로, novel 구현은 `VOImplementationTask`(Track B)로. 둘 다 ver2의 기존 seam 그대로.

---

## 3. 인터페이스 스펙 (ver3가 구현할 것)

### 3.1 `vo_lab/plugins/vo.py`

```python
# ver2 import — 재구현 0
from lab.models import (DatasetRef, ExperimentContract, ExperimentRecord, FrameworkSpec,
                        Criterion, OracleRef, BudgetSpec, Usage, VerifiedResult, Status)
from lab.menu import Menu, Recipe, ParamSpec

VO_FW = FrameworkSpec(name="torch", version="2.4", cuda="12.1")   # §6 이미지 행과 일치
VO_CODE_DIR = ".../vo_lab/plugins/vo_ref"

class VODatasetProvider:                       # satisfies lab.plugins.base.DatasetProviderP
    """실데이터 우선(KITTI/TartanAir) + 합성 fallback. held_out 분리는 여기서 강제."""
    def __init__(self, raw_root, *, synthetic_seed: int, real_first: bool = True): ...
    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        # real_first: 표준 시퀀스 다운로드(1회, dataset_cache가 캐시).
        # 실패/부재 시: 합성 렌더 — held_out이면 synthetic_seed(=harness 소유)로 GT 생성,
        #              아니면 train 시드. 솔버는 train 장면만 받음.

class VOMetricExtractor:                        # satisfies MetricExtractor (보고치, 불신)
    def extract(self, artifacts_dir: str) -> dict: ...   # /artifacts/metrics.json 있으면 읽기

# Track A: 재현/파라미터 탐색용 vetted recipe (메뉴 가드레일 — 임의 명령 불가)
def vo_recipe() -> Recipe:
    return Recipe(
        id="dpvo-vo", description="Learned VO; held-out ATE-RMSE.",
        framework=VO_FW, code_dir=VO_CODE_DIR,
        datasets=[DatasetRef("vo-train", source=...),
                  DatasetRef("vo-heldout", source=..., held_out=True)],
        train_template="LAB_EPOCHS={epochs} LAB_LR={lr} LAB_SEED={seed} python /code/run.py",
        eval_command="python /code/eval.py",          # evo 기반, harness 소유
        metric="ate_rmse", threshold=THRESH,           # 고정 oracle bar(제안자가 못 낮춤)
        params=[ParamSpec("epochs","int",low=1,high=30,default=5),
                ParamSpec("lr","float",low=1e-4,high=1e-2,default=1e-3)],
        max_wall_s=2400.0)

def vo_menu() -> Menu: return Menu([vo_recipe()])

# reproduction-first 게이트: positive=known-good, negative=degenerate(예: 정지/항등 포즈)
def vo_calibration_records() -> tuple[ExperimentRecord, ExperimentRecord]: ...
```

`eval.py`(reference, harness 소유)의 **불변 계약**: `$LAB_DATA`(held-out 시퀀스 ro), `$LAB_ARTIFACTS`(솔버 산출 궤적/ckpt), `$LAB_EVAL_OUT`에 `heldout.json` 기록. 내부에서 `evo`를 정해진 정렬 정책으로 호출. **솔버는 이 파일을 작성·수정 불가**(ver2 `Implementer`가 `eval.py`를 직접 써넣고, `ScriptEvaluator`가 평가 직전 `contract.eval_code`로 재기록 = anti-tamper).

### 3.2 `vo_lab/agents/vo_committee.py` — "회의" (Track A)

ver2 `Committee`를 그대로 쓰되 **전문가 패널만 VO용으로 교체/추가**:

```python
from lab.agents.committee import Committee, Expert, PI   # PI 재사용 가능
GEOMETRY = Expert("Geometry/SLAM", "epipolar geometry·BA·pose-graph·scale drift·loop "
                  "closure 관점에서 제안을 검토. 메뉴 파라미터만 제안, 명령 발명 금지.")
MODELING = Expert("Modeling", "학습형 VO의 backbone/loss/스케줄 관점 파라미터 제안.")
DATA     = Expert("Data", "시퀀스 분할의 leakage(같은 장면이 train/held-out 양쪽?), "
                  "scale·정렬 정책의 타당성 점검. 하이퍼파라미터는 거의 안 건드림.")
def vo_committee(model) -> Committee:
    return Committee(vo_menu(), model=model, pi=PI, experts=[GEOMETRY, MODELING, DATA], ...)
```

회의 = ver2가 이미 구현한 **draft(PI) → review(각 전문가, param_overrides+concerns+approve) → deterministic synthesis(Menu가 clamp)** 흐름. 산출은 채팅이 아니라 `ExperimentContract`. 블로그의 "pre-run sprint contract"와 일치.

> 블로그 교훈(스캐폴딩은 모델이 못하는 것의 가정): **착수는 PI+Geometry+Data 3인으로 최소 구성**, Modeling 등은 3인 실패가 증명될 때만 추가.

### 3.3 `vo_lab/agents/vo_implementer.py` — novel 구현 (Track B)

ver2 `ImplementationTask` + `sdk_author` + `sandbox_tool` 재사용. ver3는 **task 정의만** 작성:

```python
from lab.agents.implementer import ImplementationTask, sdk_author, Implementer
def vo_impl_task(eval_code: str) -> ImplementationTask:
    return ImplementationTask(
        description="Implement a monocular visual-odometry algorithm. Read an image "
                    "sequence from $LAB_DATA, write per-frame poses (TUM format) to "
                    "$LAB_ARTIFACTS/traj.txt.",
        framework=VO_FW, entry_command="python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=eval_code,   # harness 소유 grader
        metric="ate_rmse", op="<=", threshold=THRESH,
        datasets=[DatasetRef("vo-train", source=...),
                  DatasetRef("vo-heldout", source=..., held_out=True)])
```

솔버는 sandbox(network none, host shell 없음, `/code`만 쓰기)에서 `main.py`를 쓰고 디버그. `eval.py`는 못 건드림(deny). held-out에서 ver2 `ScriptEvaluator`가 채점.

**(선택) differential oracle:** novel 구현을 reference VO와 동일 시퀀스에서 대조하려면 `eval.py`가 reference 궤적과의 일치도/우열을 추가 기록. C 컴파일러의 GCC 패턴.

### 3.4 `vo_lab/factory.py` — 얇은 factory

```python
from lab.factory import build_implementer_harness          # Track B
from lab.factory import build_cifar_committee_harness as _   # 패턴 참고
def build_vo_harness(root, *, track="committee", model=None, job_mode="docker", ...):
    # committee: build_*_harness 미러 + planner=vo_committee(model), provider=VODatasetProvider
    # implementer: build_implementer_harness(root, vo_impl_task(eval_code),
    #              author_fn=sdk_author(job_runner, image_registry, dataset_cache, model=model),
    #              provider=VODatasetProvider, ...)
```

---

## 4. 시퀀스 — 한 실험의 일생 (모델 턴은 ★ 4곳만)

```
solver(committee/implementer)        Harness(loop.py, ver2)         ScriptEvaluator(ver2)
        │                                   │                                │
  ★propose_contract ───────────────────────▶│ PROPOSED→CONTRACTED            │
   (회의 or 코드 authoring)                  │ config_hash 기록               │
        │                                   │ resolve image (no tokens)      │
        │                                   │ dataset_cache.ensure (1회, IO) │
        │                                   │  ─ held_out 은 마운트 안 함     │
        │                                   │ GPU lease → job_runner.run     │
        │                                   │  train/run (IO, 턴 0)          │
        │                                   │ metric_extractor (보고치, 불신) │
        │                                   │ EVALUATING ───────────────────▶│ held-out 마운트
        │                                   │                                │ eval.py 재기록(anti-tamper)
        │                                   │                                │ evo ATE 측정
        │                                   │◀── VerifiedResult(PASS/FAIL) ──│ heldout.json 채점
        │                                   │ VERIFIED/REJECTED 기록          │
  ★decide_next ◀───────────────────────────│ (lineage 다음 수)              │
```
다운로드·학습·추론 대기는 전부 harness 안(턴 0). 예산은 `budget.py`의 **token+experiment**. → 당신의 1번 실패(turn quota 소진) 구조적 제거.

---

## 5. 상태 기계 & 자율 게이트 (ver2에서 상속, 변경 0)

```
PROPOSED → CONTRACTED → ENV_READY → DATA_READY → RUNNING → ARTIFACTS_READY
         → EVALUATING → VERIFIED | REJECTED        (FAILED: 어느 단계든 예외 시)
```
- **calibration_gate(positive, negative):** positive(known-good VO)가 **VERIFIED**이고 negative(degenerate VO)가 **REJECTED**일 때만 `autonomy_enabled=True`. → reproduction-first 강제: evaluator가 "거의 완벽"함을 *알려진 정답*으로 증명한 뒤에만 자율 루프.
- **loop(require_gate=True):** 게이트 통과 후에만. crash 시 `queue.interrupted()` 재개(job idempotent). 예산·stall로 종료.
- **단일 GPU:** `gpu_lease` mutex가 train/eval을 직렬화 — 설계상 동시성 불필요.

VO negative control 후보: 정지(항등) 포즈 / 상수 속도 / GT를 무시한 랜덤 → 큰 ATE → 반드시 REJECT. positive: 알려진 ATE를 내는 reference VO/checkpoint.

---

## 6. CUDA 이미지 — 유일하게 무거운 인프라 작업 (리스크 격리)

`images/registry.yaml`에 **prebuilt 행 1개 추가**(런타임 빌드 금지 = 당신의 2번 실패 처방):

```yaml
  - key: torch-2.4-cu121-vo
    image: "<prebuilt VO image, pinned by digest>"   # torch+CUDA+evo+VO deps 사전 설치
    cuda: "12.1"
    healthcheck: "python -c 'import torch,evo; assert torch.cuda.is_available()'"
```

**리스크:** DPVO 등은 custom CUDA 확장(컴파일 필요)이 있어 이미지 빌드가 가장 어려운 부분(연구 노트 8.2). 격리 전략:
1. **De-risk 우선:** custom CUDA op이 *없는* 순수-torch VO(또는 OpenCV 기반 classical)로 척추·게이트·평가를 먼저 검증. → 빌드 지옥 없이 end-to-end 파이프라인 성립.
2. 그 다음 DPVO 확장을 prebuilt 이미지에 굽고(1회, 사람 단계) 행 추가.
3. RTX 3080 16GB 적합성은 healthcheck + smoke-subset(짧은 시퀀스 1개)로 *job_runner 단에서* 사전 검증 — 모델 턴 0.

---

## 7. Anti-gaming 설계 매트릭스 (VO 특화 위협 → 방어)

| 위협(솔버가 점수를 속이는 길) | 방어(설계가 강제) |
|---|---|
| 쉬운 데이터를 합성해 통과 | 합성 **시드·held-out 장면은 harness 소유**; 솔버는 train 장면만. 표준 실데이터 분할 우선 |
| scale/정렬을 유리하게 선택해 ATE 낮춤 | **정렬 정책을 `eval.py`에 고정**(Sim3 Umeyama 등); 솔버가 못 고름 |
| held-out에 오버핏(GT 엿봄) | loop이 held_out 마운트 차단; evaluator만 GT 접근 |
| grader(eval.py) 덮어쓰기 | sandbox deny(`eval.py`) + 평가 직전 `contract.eval_code` 재기록(anti-tamper) |
| 보고치 위조("ATE 0.05") | `reported_metrics`는 기록만, **불신**; evaluator가 held-out에서 *직접* 재측정 |
| "그냥 돌아감"을 성공이라 주장 | 성공 = `Criterion.satisfied(measured)` AND `job.ok` — exit 0만으로는 불충분 |
| 같은 실패 반복(턴 낭비) | `failed_approaches.md` + history가 committee/lineage에 주입 |

> 한계(연구 노트와 동일, 정직): held-out이 같은 분포면 *task family* 오버핏은 못 막는다. evaluator도 같은 모델 계열 → 상관된 맹점 가능. 진짜 보증은 **결정적 held-out 측정 + evo oracle**이지 LLM evaluator가 아니다. multi-lab `peer_review`(ver2 `exchange`)로 2차 의견 확보 가능.

---

## 8. 패키지 / import 레이아웃 (의존 전략 구체화)

```
blueberry_ver2/   (PYTHONPATH 또는 `pip install -e`)  ← 검증 척추, 불변
  lab/ ...

blueberry_ver3/
  vo_lab/
    __init__.py
    factory.py              build_vo_harness(...)
    plugins/
      vo.py                 VODatasetProvider · VOMetricExtractor · vo_recipe · vo_calibration_records
      vo_ref/
        run.py              학습/추론 진입점 (Track A)
        eval.py        ★    evo ATE/RPE → heldout.json (harness 소유)
    agents/
      vo_committee.py       전문가 패널(프롬프트만; Committee 재사용)
      vo_implementer.py     VOImplementationTask (Implementer/sdk_author 재사용)
    run_vo_calibration.py   reproduction 게이트 데모
    run_vo_committee.py     Track A 자율 lineage
    run_vo_implement.py     Track B 코드 authoring
  images/registry.yaml      (또는 ver2의 것을 재사용 + VO 행 추가)
  claudedocs/               (본 문서 등)
```

- 의존 방식: ver2를 **설치형 패키지로**(권장: `pip install -e ../blueberry_ver2`) 두고 ver3가 `from lab... import ...`. `lab`는 일반적 이름이라 충돌 위험 → 가능하면 ver2를 `touchstone`으로 패키징하거나, 최소한 ver3는 별도 패키지명(`vo_lab`)을 쓴다.
- ver3는 ver2 소스를 **수정하지 않는다.** 새 도메인이 ver2 척추의 가정을 깨면(예: multi-GPU 필요) 그건 ver2의 확장 과제로 분리.

---

## 9. 구현 순서 (→ `/sc:implement` 입력)

연구 노트 9절을 이 아키텍처로 구체화. **각 단계는 이전 단계가 결정적으로 검증된 뒤 진행**:

1. **import 경계 + 얇은 factory** — `vo_lab`가 ver2를 import, `build_vo_harness`가 dummy provider로 self-test 통과(모델·GPU·Docker 0). ver2 `build_dummy_harness` 패턴.
2. **VO plugin + `eval.py`(evo)** — `VODatasetProvider`(합성 fallback 먼저, 결정적·오프라인) + harness 소유 `eval.py`. de-risk용 순수-torch/classical VO reference.
3. **reproduction calibration 게이트** — positive(known-good)=VERIFIED, negative(degenerate)=REJECT 증명. *여기서 evaluator가 "거의 완벽"한지 확정*. 통과 전 자율 금지.
4. **CUDA 이미지 행 + 실데이터 provider** — KITTI/TartanAir 1회 캐시, prebuilt VO 이미지(필요 시 DPVO 확장 굽기). smoke-subset 사전검증.
5. **Track A 회의(committee)** — PI+Geometry+Data로 메뉴 제약 자율 lineage. 예산 token+experiment.
6. **Track B implementer** — novel VO 코드 authoring(sandbox), (선택) differential oracle.
7. **VerifiedResult 출력 고정** — 후속 multi-lab `peer_review` 대비(ver2 `exchange` 그대로).

---

## 10. 미해결 구현 결정 (`/sc:implement` 전 확인)

1. **De-risk VO 선택:** 2~3단계의 reference VO를 (a) 순수-torch 경량 VO vs (b) OpenCV classical VO 중 무엇으로? (둘 다 custom CUDA 회피; (b)가 빌드 가장 가벼움) — DPVO는 4단계에서.
2. **벤치마크/분할:** KITTI odometry(00–08/09–10) vs TartanAir(장면 분리) 중 첫 reproduction 대상. ATE threshold는 선택한 reference의 알려진 수치에서 역산.
3. **정렬 정책 확정:** monocular면 Sim3(scale 보정), stereo/RGB-D면 SE3. `eval.py`에 고정값으로 박을 정책.
4. **ver2 패키징:** 현 상태(`lab` import) 그대로 vs `touchstone`으로 재패키징(이름 충돌·배포 청결).
5. **첫 트랙:** Track A(committee, 재현·탐색)부터 vs Track B(implementer, novel 구현)부터. (reproduction-first 정신상 A 권장)

---

## Sources
- ver2 실측: `lab/models.py · loop.py · evaluator.py · plugins/base.py · plugins/cifar.py · factory.py · menu.py · image_registry.py · job_runner.py · paths.py · agents/{committee,implementer,sandbox_tool}.py · exchange.py · lab.py · images/registry.yaml`.
- 선행: `blueberry_ver3/claudedocs/research_cv_research_lab_design_2026-06-02.md`.
- Anthropic Engineering: harness-design-long-running-apps, building-c-compiler.
