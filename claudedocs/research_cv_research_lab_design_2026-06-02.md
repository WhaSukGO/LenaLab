# Computer Vision Research Lab (ver3) — Harness 설계 리서치 리포트

- 날짜: 2026-06-02
- 질문: Anthropic Agent SDK / Harness engineering으로 "CV 리서치 랩"(전문가 팀이 회의·조사·구현·실행해 결과를 도출)을 만들되, 이전 ralph-loop 시도의 실패(턴 낭비, 외부 리소스 관리, "그냥 돌아가기만 함")와 Anthropic 블로그가 경고한 실수를 반복하지 않으려면 어떻게 설계해야 하는가.
- 1차 출처: Anthropic Engineering — *Harness design for long-running agentic apps*, *Building a C compiler* (둘 다 직접 인용 기반, High).
- 2차 입력: **당신이 이미 만든 `blueberry_ver2` = Touchstone (검증기)** 의 실제 코드/README (실측, High).
- 신뢰도: 블로그 교훈 High. ver2 재사용 가능성 판단 High. CV 솔버측 설계 추론 Medium(미확정 결정 5절에 의존).

> 이 리포트는 `/sc:research` 산출물이며 **구현은 하지 않는다.** 끝에 설계를 가르는 결정 질문을 제시한다.

---

## 0. Executive Summary — 단 하나의 재구성(reframing)

당신은 이미 답의 절반을 만들었다. `ver2`(Touchstone)는 **검증기(verifier)**다 — generator가 자기 결과를 채점하지 못하게 하고, held-out·oracle·calibration gate로 "실행 성공 ≠ 연구 성공"을 강제하는 *불변 척추*. 두 블로그가 "가장 강력한 레버"라 부른 **generator ⟂ evaluator 분리**가 이미 구현돼 있다.

**ver3는 ver2가 의도적으로 *하지 않는다*고 명시한 바로 그 절반이다.** ver2 README의 "What it does *not* do":

> - "It does **not invent novel methods or chase SOTA**. Autonomy = 벗어난 추론이 아니라 *vetted recipe의 파라미터 공간 탐색*, 또는 *명시된 task*의 구현."
> - "wiring a genuinely new *domain* (its dataset provider + CUDA image) is still **human setup**."

즉 ver3 = **솔버(solver) / 리서치팀**: 문제·요구·환경이 주어지면 (1) 데이터를 구하거나 없으면 합성하고, (2) 알고리즘을 구현하고, (3) 실행해 결과를 낸다. 이건 ver2가 비워둔 "novel implementation + 새 도메인 wiring + 데이터 확보/합성" 슬롯이다.

따라서 **가장 중요한 설계 결정은 "ver3에서 검증기를 다시 만들지 말 것"**이다. ver3는 ver2를 *라이브러리로 의존*해서 — `image_registry`, `dataset_cache`, `job_runner`, `gpu_lease`, `evaluator`, `VerifiedResult` 계약을 import — 그 위에 **리서치팀 레이어**만 얹는다. ver3의 유일한 합법적 출력은 ver2가 서명한 `VerifiedResult`다. ver3가 자기 결과를 "좋다"고 말하는 순간 그게 바로 당신의 이전 실패다.

이 한 줄이 나머지를 정한다: **ver3는 "더 똑똑한 generator"를 만드는 프로젝트이지, "더 너그러운 verifier"를 만드는 프로젝트가 아니다.**

---

## 1. 당신이 이미 가진 것 (ver2 / Touchstone) — 그리고 비워둔 슬롯

`ver2`에서 **그대로 재사용**할 결정적 인프라 (모델 호출 0, 턴 소비 0):

| ver2 모듈 | 역할 | ver3에서 |
|---|---|---|
| `image_registry.py` | 사전 빌드된 (framework×CUDA) 이미지 매트릭스, 런타임 빌드 금지 | **그대로 import.** "torch 언급 → cuda-compiled image" 가 이미 표로 고정됨 |
| `dataset_cache.py` | download-once 캐시 | **그대로 import.** 다운로드 turn 낭비의 직접 해법 |
| `job_runner.py` | Docker/local 잡 실행, read-only mount, host user | **그대로 import.** 학습/다운로드를 harness 잡으로 |
| `gpu_lease.py` | 단일 GPU mutex | **그대로 import.** 직렬 실행 보장 |
| `budget.py` | **token + experiment** 예산 (turn 아님) | **그대로 import.** "20-turn quota 소진"의 직접 해법 |
| `registry.py` | crash-resumable SQLite 실험 store | **그대로 import.** context reset 생존 |
| `evaluator.py` / `agents/evaluator.py` | 독립·회의적 evaluator, held-out 측정, 절대 upgrade 금지 | **그대로 import = 신뢰 게이트** |
| `exchange.py` / `lab.py` | 서명된 `VerifiedResult` + provenance (config hash, image digest, dataset hash, seed) | **그대로 import = ver3의 출력 계약** |
| `agents/committee.py` | PI + Modeling + Data 전문가가 menu-constrained 제안 협상 | **확장의 출발점** (아래 3·4절) |
| `agents/implementer.py` + `sandbox_tool.py` | 코드를 쓰고 컨테이너에서만 실행(host shell 없음) | **확장의 출발점** |

ver2가 **비워둔 슬롯 = ver3의 본체**:
1. **데이터 확보/합성** — ver2는 "새 도메인의 dataset provider는 human setup". ver3는 이걸 *에이전트가* 한다 (구하거나, 없으면 합성).
2. **알고리즘 구현(novel)** — ver2는 "vetted recipe의 파라미터 탐색 또는 *명시된* task 구현"만. ver3는 명세가 느슨한 상태에서 *방법을 구현*한다.
3. **새 CV 도메인 wiring** — recipe = dataset provider + reference code + metric + oracle + CUDA image. ver3는 이 플러그인을 *에이전트가* 채운다.

→ 이 세 슬롯은 전부 **검증이 더 어려워지는 방향**이다. 그래서 1절의 결론이 강제된다: 검증기는 ver2에 두고 절대 약화시키지 말 것.

---

## 2. 블로그 교훈 — 솔버(ver3) 렌즈로만 다시 읽기 (델타만)

> 일반 교훈은 ver2의 기존 노트(`research_cv_lab_harness_design_2026-06-01.md`)에 정리돼 있다. 여기선 **솔버측에서 새로 의미가 바뀌는 것만**.

- **"Verifier가 거의 완벽해야 한다"** (C 컴파일러) → ver3에선 *완벽한 검증기를 빌리되, 그 검증기가 ver3의 새 도메인을 채점할 oracle을 갖는지*가 새 병목. ver2의 검증력은 "known oracle이 있을 때" 강하다(README 명시). ver3가 oracle 없는 일을 시도하면 검증기는 무력해진다 → **5절(oracle 부재 문제)이 ver3의 핵심 난제.**

- **Oracle 기반 분해 (GCC를 reference로)** → ver3 CV 매핑: novel 구현을 검증할 땐 *reference implementation*(공식 repo / 논문 코드 / 표준 라이브러리 함수)을 differential oracle로. "내 구현 출력 == reference 출력"은 "그냥 돌아감"보다 훨씬 강한 기준.

- **Over-specification 회피** → ver3는 PI/Planner가 "deliverable(성공=무엇)"만 못박고 구현 경로는 열어둔다. 단 *deliverable은 채점가능*해야 함(metric + oracle + held-out). 이게 sprint contract = experiment contract.

- **Context reset > compaction, handoff artifact** → ver3는 다운로드·학습이 길어 세션이 자주 리셋된다. 모든 상태는 `registry`(SQLite) + lab notebook 파일로. 에이전트 컨텍스트엔 *집계+표준 에러만*, 학습 stdout 금지(C 컴파일러: "thousands of useless bytes 금지").

- **harness 컴포넌트 = 모델이 못하는 것의 가정** → Opus 4.8 기준 ver2의 committee/menu 같은 스캐폴딩 중 일부는 과할 수 있다. **가장 단순한 솔버부터 시작**(단일 implementer + 독립 evaluator), 실패가 증명될 때만 committee/role 분화 추가.

---

## 3. ver3가 *새로* 들여오는 실패 모드 (이전 노트엔 없던 것)

솔버를 진짜로 일하게 만들면 새 reward-hacking·오류 표면이 생긴다. 각각에 *검증기측 대응*을 붙인다.

### 3.1 "구현이 돌아감" ≠ "구현이 맞음" (silent-wrong)
CV 코드는 *돌아가면서 미묘하게 틀리기* 쉽다: 잘못된 정규화, train/test 누수, 채널 순서(BGR/RGB), metric 정의 오류(mIoU vs pixel-acc), eval mode 미설정(dropout/BN), test-time augmentation 누락. 전부 "실행 성공"을 통과한다.
- **대응:** deliverable에 *reference oracle* 또는 *property/invariant 체크*를 강제(5절). 단순 exit-code 0은 성공 아님.

### 3.2 데이터 확보/합성이 새로운 gaming 벡터
ver3는 "없으면 synthetic"을 한다. 그런데 **솔버가 데이터와 held-out을 동시에 만들면**, 솔버는 *쉬운 데이터를 합성*해 자기 알고리즘이 통과하게 만들 수 있다(= 당신의 이전 실패의 CV판). 또한 다운로드 데이터는 *라이선스/출처/leakage* 위험.
- **대응 (중요):** **held-out split과 합성 데이터의 생성 시드는 *검증기(ver2)*가 소유**한다. 솔버는 train split / 합성 *생성기 코드*만 본다. → 역설적으로 **합성은 검증의 *자산*이 된다**: 합성 과정이 곧 ground-truth oracle(라벨을 정의상 안다). ver2엔 이미 `plugins/vision_blobs.py`가 있어 이 패턴의 씨앗이 있다.

### 3.3 다운로드·CUDA 빌드의 turn 낭비 (당신의 1번 실패)
- **대응:** 이미 ver2가 풀었다 — `dataset_cache`(1회 다운로드) + `image_registry`(런타임 빌드 금지, 사전 빌드 매트릭스). ver3는 *재발명하지 말고 import*. 예산은 `budget.py`의 token+experiment, IO 대기는 예산·턴에서 0.

### 3.4 단일 GPU에서 자율 루프 + 약한 검증 = 무한 낭비
완전 자율 + 단일 GPU + 약한 verifier는 *존재론적 결합*이다: verifier가 약하면 모델이 엉뚱한 문제를 무한히 풀며 유일한 GPU를 통째로 태운다.
- **대응:** **reproduction-first**로 착수해 evaluator가 *알려진 정답*을 재현하는지부터 보정(calibration gate). 게이트가 열리기 전엔 자율 루프 금지. 모든 실험은 idempotent·resumable(`registry` 재개).

### 3.5 "회의(meeting)"를 채팅으로 구현하는 함정
전문가 회의를 다중 에이전트 *대화*로 만들면 토큰만 태우고 context degradation을 부른다.
- **대응:** 회의 = **파일 기반 아티팩트 협상** (가설 → experiment contract → 결과 → verdict). 합의는 채팅 종료가 아니라 *contract 파일에 서명*. ver2 committee가 이미 이 방향.

---

## 4. ver3 권장 아키텍처 — "ver2를 의존하는 솔버 레이어"

### 4.1 두 개의 루프 (결정적 harness ⟂ 모델 추론) — ver2에서 상속
```
[Outer harness loop — 결정적 코드, 턴 0]  (대부분 ver2 재사용)
  while budget.remaining() and not stalled:
    contract = team.propose_experiment(history)   # ← 모델 추론(가설·설계)
    code     = team.implement(contract)            # ← 모델 추론(구현, sandbox)
    env      = image_registry.resolve(framework)   # 결정적 (ver2)
    data     = dataset_cache.ensure(datasets)      # 결정적, 1회 (ver2)
    job      = job_runner.run(env, data, code)     # harness가 대기, 턴 0 (ver2)
    verdict  = touchstone.evaluate(contract, job)  # ← 독립 evaluator (ver2)
    registry.record(...) ; notebook.append(...)    # 영속화 (ver2)
    history  = team.reflect(verdict)               # ← 모델 추론(다음 수)
```
모델 턴은 **추론 지점 4곳에서만** 소비: 실험설계 / 구현 / 해석 / (필요시) 회의. 다운로드·빌드·학습 대기는 전부 harness.

### 4.2 전문가 팀 (서브에이전트) — ver2 committee 확장
- **PI / Planner** — 문제→가설→experiment contract(성공=채점가능 deliverable). 고수준에 머묾(over-spec 회피).
- **Data expert** — 데이터 *확보*(cache 경유 다운로드) 또는 *합성 생성기 코드* 작성. **단 held-out·시드는 검증기 소유**(3.2).
- **Modeling/Implementer expert** — 알고리즘 구현. sandbox(host shell 없음)에서만 실행 (ver2 `implementer`/`sandbox_tool`).
- **(독립) Evaluator** — *ver3가 아니라 ver2의 것*. 별도 컨텍스트/프로세스. held-out 재현, oracle 대조, rubric 채점, upgrade 금지.
- 역할 분화는 **필요가 증명될 때만** 추가(2절 마지막 교훈). 1차 버전은 PI+Implementer+ver2 evaluator로 시작.

### 4.3 영속 상태 (context reset 생존) — ver2 상속
`registry`(실험 DAG: hypothesis·config·env·datasets·cmd·status·metrics·artifacts·verdict·parent) + lab notebook + **failed-approaches log**(순환 방지) + dataset/weights 캐시 + CUDA image matrix. 로그는 파일로, 에이전트엔 집계+표준 에러만.

### 4.4 출력 계약 — ver3의 유일한 산출은 서명된 `VerifiedResult`
provenance(config hash·image digest·dataset hash·seed) 포함. 이게 있어야 (a) 재현 가능, (b) 나중에 다른 랩이 peer-review/협업, (c) "솔버 주장"이 결과로 둔갑하는 일 차단.

---

## 5. ver3의 핵심 난제 — **oracle이 없을 때 무엇이 "성공"인가**

ver2는 "known oracle이 있을 때" 강하다. ver3의 야심(데이터 합성·novel 구현)은 oracle이 *주어지지 않는* 영역으로 간다. 이게 진짜 어려운 부분이고, "그냥 돌아가면 만족"으로 후퇴하는 정확한 지점이다. **oracle을 약한 것→강한 것 순서의 사다리로 확보**하라:

1. **Reproduction oracle (가장 강함, 착수점)** — 알려진 논문/벤치마크 수치. evaluator가 held-out에서 tolerance 내 재현하면 성공. *여기서 검증기를 calibrate*한 뒤에만 위로 올라간다.
2. **Differential / reference oracle** — novel 구현을 *표준 reference 구현*(공식 repo, torchvision, scipy 등)과 동일 입력에서 출력 대조 (C 컴파일러의 GCC 패턴). "reference와 일치"가 기준.
3. **Synthetic ground-truth oracle** — 데이터가 없어 합성할 때, **생성 과정이 곧 정답**(라벨을 정의상 안다). 검증기가 시드·held-out 소유(3.2). CV에 특히 강력: 도형/blob/렌더링으로 검출·분할·기하 task의 정답을 무한 생성.
4. **Baseline-beating oracle** — oracle 수치가 없으면 *동일 조건의 정당한 baseline*을 검증기가 직접 돌려 바를 만들고, 솔버가 held-out에서 그걸 *유의하게* 넘는지(seed 분산·신뢰구간).
5. **Property / invariant oracle (정답 없이도 가능)** — ground truth가 전혀 없어도 검증 가능한 불변식: segmentation mask validity, 기하 equivariance, ablation monotonicity, train-loss↓ 시 held-out 일반화, 수치 안정성. 약하지만 "그냥 돌아감"보다 훨씬 강하다.

**규칙:** ver3의 모든 experiment contract는 위 사다리에서 *최소 하나*의 oracle을 명시해야 실행 자격이 생긴다. 1~3이 가능하면 4~5로 후퇴 금지. oracle을 못 붙이는 실험은 *설계 미완*이지 실행 대상이 아니다.

---

## 6. 안티패턴 체크리스트 (ver3 추가분)

ver2 노트의 8개에 더해:
- [ ] **검증기를 ver3에서 재구현하지 않는다** (ver2를 import). ver3는 솔버만.
- [ ] 솔버가 held-out split / 합성 시드를 *소유하지 않는다* (검증기 소유).
- [ ] exit-code 0 / "돌아감"을 성공으로 기록하지 않는다 — contract의 oracle 통과만 성공.
- [ ] oracle 없는 실험을 실행 큐에 넣지 않는다 (설계 미완으로 반려).
- [ ] novel 구현은 가능하면 reference 구현과 differential 대조한다.
- [ ] 전문가 "회의"를 토큰 태우는 채팅이 아니라 contract 파일 협상으로 한다.
- [ ] 역할 분화·committee를 *기본값으로* 켜지 않는다 — 단일 솔버 실패가 증명될 때만.
- [ ] reproduction calibration gate가 열리기 전 자율 루프를 켜지 않는다.

---

## 7. 미해결 결정 → 당신에게 물을 질문

이전 노트에서 확정된 것: ① 한 랩 먼저 → 후에 multi-lab 협업, ② 단일 로컬 GPU, ③ 완전 자율(단 독립 evaluator), ④ Python Agent SDK. **아래는 ver3 특화로 새로 갈리는 결정들**(별도 질문 도구로 확인):

- **Q1. ver2 의존 방식** — ver3가 Touchstone을 *라이브러리로 import*(권장)할지, 일부만 베껴 standalone으로 갈지. (검증기·인프라 재사용 vs 결합도)
- **Q2. "성공"의 oracle 모드** — 5절 사다리 중 어디서 시작하고 어디까지 허용할지(reproduction-first vs synthetic-first vs baseline-beating). 검증기 난이도를 근본적으로 가른다.
- **Q3. CV 하위도메인** — 분류/검출/분할/생성(diffusion)/비디오/3D. 데이터·oracle·CUDA 이미지·infra가 전부 달라진다. (여전히 미확정)
- **Q4. 데이터 정책** — 실데이터 다운로드가 범위인가(네트워크/라이선스 허용?), 아니면 합성 우선인가. 그리고 합성 시드/held-out을 검증기가 소유하는 설계를 받아들일지.

---

## 8. 확정 결정 (2026-06-02 사용자 답변) + SLAM 도메인 분석

확정: ① ver2를 **라이브러리로 import**(검증기 재구현 금지), ② **reproduction-first** oracle, ③ 도메인 = **SLAM/Visual Odometry**(RTX 3080 16GB에서 전부 구동), ④ 데이터 = **실데이터 우선 + fallback 합성**(시드·held-out은 검증기 소유).

### 8.1 왜 SLAM이 DL 4개 옵션보다 *이 harness에* 더 맞는가 (당신의 직관이 정확하다)

당신이 친 핵심 질문 — "1~4는 다 deep learning인데 알고리즘이 중요한가?" — 은 이 프로젝트의 목표("오픈소스가 돌아가는 것에 만족하면 안 된다")와 **정확히 정렬**된다:

1. **당신의 1번 실패(턴 낭비)를 구조적으로 제거.** DL 랩의 wall-clock은 대부분 *대용량 다운로드 + 수 시간 학습*에 잡힌다 — 바로 당신이 turn quota를 태운 지점. SLAM/VO는 데이터셋이 GB 단위(KITTI odometry, TUM RGB-D, EuRoC)로 1회 캐시면 끝, 한 실험 iteration이 *분 단위*다. GPU babysitting 문제가 거의 증발한다.
2. **"그냥 돌아감"으로 후퇴할 여지가 작다.** DL 분류 랩은 결국 "lr/epoch 튜닝" = ver2가 이미 하는 *menu 파라미터 탐색*으로 퇴화한다(당신이 넘어서려는 바로 그것). SLAM은 *알고리즘 구현 자체가 일*이라 "남의 학습 스크립트 돌렸다" 뒤에 숨을 수 없다. 솔버의 가치가 코드 구현에 있다는 당신의 정의와 맞는다.
3. **CV 전체에서 oracle 상황이 가장 좋다 = reproduction-first의 이상적 짝.** SLAM은 *ground-truth 궤적*을 가진 골드 벤치마크(KITTI odometry, TUM RGB-D, EuRoC MAV, TartanAir)와 *표준 결정적 metric*(ATE/RPE, `evo`·TUM 스크립트)이 존재. C 컴파일러 블로그의 "verifier가 거의 완벽해야 한다"를 *실제로 달성 가능*하게 만든다 — gaming 불가한 정답 궤적 + 표준 측정 도구.
4. **단일 GPU에 이상적.** classical SLAM은 대부분 CPU+geometry, 학습형(DPVO/DROID)도 16GB에 들어간다. 직렬 실행(gpu_lease)로 충분, 동시성 불필요.
5. **합성 fallback이 강한 oracle이 된다.** 합성 궤적/렌더 장면은 *정의상 정답 포즈*를 준다(TartanAir가 바로 이 방식). Q4의 "fallback 합성" + Q2의 "reproduction-first"가 5.3 synthetic-ground-truth oracle로 자연 결합.

### 8.2 SLAM의 진짜 위험 (반대편도 정직하게)

- **환경 구축이 가장 어려운 부분.** classical ORB-SLAM 계열은 C++ 빌드 지옥(Pangolin, g2o, Eigen, OpenCV, CUDA dep). 이게 당신이 말한 "CUDA 컨테이너 최소 리소스" 고통 그 자체. → ver2 `image_registry`(사전 빌드 매트릭스)로 *1회* 흡수해야지, 매 실험 빌드는 금물.
- **평가의 silent-wrong(3.1).** ATE는 비교 전 Sim(3)/SE(3) 정렬(Umeyama)이 필요(monocular scale 모호성). 잘못 구현하면 *그럴듯하게 틀린 ATE*가 나온다. → 검증기는 metric을 자작하지 말고 **표준 도구(`evo`)를 oracle로 호출**.
- **비결정성.** RANSAC·멀티스레드·feature 매칭 순서로 같은 입력도 궤적이 흔들린다. → tolerance band + multi-seed(설계가 이미 요구). 관리 가능하나 실재함.

### 8.3 권장 첫 랩 형태 (= 다음 결정 fork)

SLAM "맛"이 인프라를 가른다 — Python/torch(ver2 image_registry 재사용 쉬움) vs C++ 빌드(셋업 무겁지만 oracle 성숙). MVP 권장: **monocular/stereo Visual Odometry(루프클로저 없는 부분집합)부터, KITTI/TartanAir에서 oracle = `evo`의 ATE/RPE, 합성 fallback = 렌더 궤적.** 스코프가 작아 iteration이 빠르고 oracle은 동일하게 강하다.

## 9. 확정 구성(LOCKED) + 권장 착수 순서

**최종 확정 구성:**

| 축 | 결정 | 함의 |
|---|---|---|
| 검증기 | ver2(Touchstone)를 **library import** | 솔버만 신규 구현. `image_registry·dataset_cache·job_runner·gpu_lease·budget·registry·evaluator·exchange(VerifiedResult)` 재사용 |
| oracle | **Reproduction-first** | 첫 실험은 알려진 VO 수치 재현. evaluator가 oracle 재현 보정(calibration gate) 후에만 자율 |
| 도메인 | **SLAM/VO**, 시작 = **Learned VO (DPVO/DROID 계열)** | Python/PyTorch → ver2 torch image 재사용. 16GB 적합. 알고리즘 구현 중심 |
| 데이터 | **실데이터 우선 + 합성 fallback** | KITTI/TartanAir 우선(cache 1회). 없으면 합성 — 단 **held-out·시드는 검증기 소유** |
| 컴퓨트 | 단일 RTX 3080 16GB, 직렬 | gpu_lease mutex, smoke-subset 선검증 |
| 기반 | Python Agent SDK | 턴 아닌 token+experiment 예산 |

**Learned VO 선택의 추가 함의:** DPVO는 16GB에 들어가고 학습/추론이 짧아 turn-waste가 최소. 솔버의 "구현" = VO 알고리즘(patch/correlation, bundle adjustment, pose graph) 코드. oracle = TartanAir(합성 ground-truth 내장, Q4 fallback과 5.3 결합) + KITTI odometry, metric = `evo` ATE/RPE를 검증기가 호출(self-impl 금지, 3.1 silent-wrong 방지).

**권장 착수 순서 (구현 단계에서):**
1. **ver2 import 경계 확정** — ver3가 ver2의 어떤 심볼을 쓰는지 thin adapter. 검증기 0줄 재구현.
2. **VO recipe 플러그인 1개** — dataset provider(TartanAir/KITTI via dataset_cache) + reference VO + metric(`evo` ATE/RPE) + oracle + torch CUDA image. ver2 `plugins/` 패턴 그대로.
3. **독립 evaluator 보정** — 알려진 VO 결과 1건 reproduction으로 evaluator가 "거의 완벽"한지 증명(calibration gate). 통과 전 자율 금지.
4. **솔버(리서치팀) 레이어** — PI/Planner + Implementer(sandbox) + (ver2)evaluator. experiment contract = 채점가능 deliverable + oracle 명시. 회의는 파일 협상.
5. **자율 루프 개방** — calibration gate 통과 후에만. 예산 token+experiment. 합성 시드/held-out은 검증기 소유.
6. **VerifiedResult 출력 고정** → 후속 multi-lab 협업/peer-review 대비.

> 다음 단계는 사용자 결정: 아키텍처 상세는 `/sc:design`, harness/플러그인 골격 구현은 `/sc:implement`. 본 `/sc:research`는 리포트까지만 산출한다.

## Sources
- Anthropic Engineering — Harness design for long-running agentic apps: https://www.anthropic.com/engineering/harness-design-long-running-apps
- Anthropic Engineering — Building a C compiler: https://www.anthropic.com/engineering/building-c-compiler
- 실측: `blueberry_ver2` (Touchstone) README + `lab/` 소스 (image_registry, dataset_cache, job_runner, gpu_lease, budget, registry, evaluator, exchange, committee, implementer, plugins/vision_blobs).
- 선행 노트: `blueberry_ver2/claudedocs/research_cv_lab_harness_design_2026-06-01.md`.
