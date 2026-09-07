# MatFormBench range-adapted 결과 — 첫 전체 sweep

**요약.** 50회 예산 안에서 **서로 다른 규격 만족 배합을 몇 개 찾는가**로 보면 TRMC-BO가
1위다 — full 변형이 **6.41개**, Point-target BO는 2.86개, Range-Aware TB는 1.37~1.73개.
세 구성요소 ablation은 모든 window width에서 단조 증가한다(2.31 → 4.65 → 6.41).

**적중 횟수만 보면 순위가 다르다.** Point-target BO는 50회 중 11.01회를 창 안에
넣지만(TRMC-BO full 6.88회), 그 적중이 **서로 다른 배합 2.86개로 붕괴**한다(-74%).
TRMC-BO full은 6.88 → 6.41로 **7%만** 줄어든다. 두 지표는 서로 다른 질문에 답하며
둘 다 보고한다.

**Extremum-seeking BO는 random search보다 8배 나쁘다.** 논문의 전제를 강하게
뒷받침한다.

**정보 누출은 배제했다.** 재현 가능한 감사(`eval/scripts/audit_leakage.py`)가 3개 task
전부에서 통과한다. Point-target이 읽는 필드 집합은 TRMC-BO와 **정확히 동일**하다.

**대신 우리 구현에서 명세 불일치를 찾았다 — 그런데 명세가 더 나쁘다.** 논문 §4.2는
확률의 **곱**을 명세하는데 구현은 출력별 확률 벡터를 qEHVI에 넘긴다. 1-step 순위
진단(AUC)에서는 곱이 0.641로 현 구현 0.542를 앞선다. 그런데 캠페인에서 곱은 hit rate를
올리는 대신 다양성을 무너뜨린다 — L4-1에서 **적중 27개가 distinct 1개로** 붕괴한다
(현 구현은 8 → 8). 곱은 단일 내부점에서 최대가 되는 unimodal attractor이고 hypervolume은
확산을 보상하며, **C2/C3는 이를 상쇄하지 못한다.** 즉 **결함으로 보였던 것이 우리
헤드라인 결과인 다양성을 만들던 장치다.** 논문 §4.2를 다시 봐야 한다.

1200 cells, error 0건, 22 workers로 약 9.5시간. **native protocol도 완료**했고 벤치마크
자체 채점에서도 C2+C3가 모든 하위 점수를 올린다(57.19 → 64.90). **COMBOO는 보류**했다.

---

## 무엇을 돌렸는가

과제는 **specification-driven design**이다. 물성이 extremum이 아니라 window
`L ≤ y ≤ U` **안에** 들어오는 formulation을 찾는다. 모든 task가 물성 3개를 동시에
만족시켜야 한다.

- **Benchmark**: [MatFormBench](https://github.com/DeepVerse/MatFormBench).
  smooth response → coupled/noisy → local invalid region → multimodal →
  sparse feasibility로 이어지는 5개 task. 10D/15D task는 진짜 mixture 문제다
  (성분 6개 합이 1, 여기에 process variable 추가).
- **Window**: MatFormBench는 one-sided target(`y1 > 61`)을 제공한다. native
  threshold를 한쪽 bound로 유지하고 반대쪽을 quantile로 배치하되,
  *input-feasible* design 중 합격 비율이 **10% / 3% / 1%**(wide/medium/narrow)가
  되도록 calibration했다. 실행 전에 동결하고 commit했으므로 모든 method와 seed가
  바이트 단위로 동일한 target을 본다.
- **Protocol**: initial design 30점(MatFormBench 자체 설정), 이후 **sequential
  evaluation 50회**, seed 10개. 같은 seed에서는 모든 method가 동일한 initial
  design에서 출발한다. Method는 noisy observation만 보고, 채점은 noiseless
  oracle로 한다.
- **Matrix**: 5 tasks × 3 widths × 8 methods × 10 seeds = 1200 runs.

### 비교 대상 method

| | 무엇을 하는가 |
| --- | --- |
| **Random search** | 바닥선. hit rate가 calibration된 window 비율 근처에 와야 정상 |
| **Standard BO (qEHVI)** | window를 무시하고 native 방향으로 extremum 탐색 |
| **Point-target BO** | window 중앙을 단일 target으로 겨냥 |
| **TRMC-BO (range probability only)** | C1: `P(L ≤ f(x) ≤ U)` 최대화 |
| **TRMC-BO (+ two-sided constraints)** | C1 + C2: window 경계를 posterior mean에 대한 hard constraint로 부여 |
| **TRMC-BO (full)** | C1 + C2 + C3: optimizer restart를 in-band 점으로 filtering까지 |
| **Range-Aware TB** ×2 | 가장 가까운 published competitor([arXiv 2606.11574](https://www.alphaxiv.org/abs/2606.11574)). 우리 box target을 그들 ball target으로 옮기는 두 가지 mapping |

---

## 주 결과 — 서로 다른 배합을 몇 개 찾았는가

`hit_rate`는 **50번의 평가 중 창 안에 든 비율**이다. 서로 다른 배합의 수가 아니다.
한 지점에 수렴해 그 근처를 계속 제안하는 method는 이 지표에서 유리하다. 중복 판정
허용오차가 `1e-8`이라 1e-6 떨어진 두 배합은 중복으로 걸리지 않고 둘 다 적중으로
계산된다.

`distinct`는 정규화된 설계공간에서 **서로 δ=0.1 이상 떨어진** 유효 배합의 수다
(Range-Aware BO 논문의 δ-uniqueness 개념). `conc.` = distinct/valid 는 그 기전을
드러낸다.

| Method | **distinct** | valid | conc. | 1st hit | viol | CV | s/iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random search | 1.65 | 1.65 | 100% | 29.2 | 0.460 | 0.58 | 0.0 |
| Standard BO (qEHVI) | 0.19 | 0.19 | 100% | 44.8 | 0.866 | 1.64 | 63.7 |
| Point-target BO | 2.86 | **11.01** | **26%** | 16.1 | **0.143** | 0.76 | 12.0 |
| TRMC-BO (range prob. only) | 2.31 | 2.47 | 94% | 21.6 | 0.378 | 0.61 | 4.4 |
| TRMC-BO (+ constraints) | 4.65 | 4.81 | 97% | 18.9 | 0.267 | **0.51** | 4.8 |
| **TRMC-BO (full)** | **6.41** | 6.88 | 93% | **14.8** | 0.188 | 0.54 | 6.8 |
| Range-Aware TB (equal-vol.) | 1.73 | 5.27 | 33% | 27.2 | 0.173 | 0.99 | 26.9 |
| Range-Aware TB (inscribed) | 1.37 | 6.70 | **20%** | 24.7 | 0.184 | 0.96 | 18.3 |

TRMC-BO 계열은 **찾은 것이 거의 다 서로 다른 배합**이다(93~97%). Point-target은 26%,
Range-Aware TB는 20~33%로 **한 분지를 반복 제안한다.** `1st hit`(미발견을 검열값으로
포함한 평균)도 TRMC-BO full이 14.8로 가장 빠르다.

### 누적 곡선 — 뭉치는 method는 평평해진다

서로 δ=0.1 이상 떨어진 유효 배합의 누적 개수:

| Method | it 1 | it 10 | it 20 | it 30 | it 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Point-target BO | **0.17** | 0.85 | 1.39 | 1.83 | 2.86 |
| TRMC-BO (+ constraints) | 0.09 | 0.74 | 1.65 | 2.51 | 4.65 |
| **TRMC-BO (full)** | 0.12 | **1.01** | **1.95** | **3.27** | **6.41** |
| Range-Aware TB (equal-vol.) | 0.13 | 0.64 | 1.12 | 1.39 | 1.73 |
| Range-Aware TB (inscribed) | 0.14 | 0.53 | 0.89 | 1.02 | 1.37 |

TRMC-BO가 **느리게 시작하지 않는다.** iteration 1에서만 뒤지고 10회차부터 앞서며
격차가 계속 벌어진다. Point-target과 TB는 20~30회차 이후 곡선이 평평해진다 — 찾을
새로운 분지가 없다는 뜻이 아니라, 이미 찾은 분지를 다시 두드리고 있다는 뜻이다.

### δ 선택이 결론을 바꾸지 않는다

δ는 벤치마크가 제공하는 물리적 허용오차가 아니라 우리가 정한 값이므로, 순위가 δ에
의존하지 않음을 보여야 한다.

| Method | d>0.05 | d>0.1 | d>0.2 | d>0.4 |
| --- | ---: | ---: | ---: | ---: |
| Random search | 1.65 | 1.65 | 1.63 | 1.53 |
| Point-target BO | 3.69 | 2.86 | 2.39 | 1.95 |
| TRMC-BO (range prob. only) | 2.39 | 2.31 | 2.19 | 1.88 |
| TRMC-BO (+ constraints) | 4.71 | 4.65 | 4.36 | 3.28 |
| **TRMC-BO (full)** | **6.72** | **6.41** | **5.59** | **3.79** |
| Range-Aware TB (equal-vol.) | 2.07 | 1.73 | 1.32 | 1.01 |
| Range-Aware TB (inscribed) | 1.67 | 1.37 | 1.25 | 1.16 |

TRMC-BO full이 네 δ 전부에서 1위이고 ablation 사다리도 전부에서 단조다.

### 그림

5개 task 전부에 대해 생성되어 있다 (`eval/results/figures/distinct_{task}_medium.png`).
이 저장소는 `eval/results/`를 추적하지 않으므로 이슈에는 파일을 직접 첨부한다. 재생성:

```bash
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_distinct_designs \
    --task L3-1 --width medium
```

각 패널은 한 method가 찾은 유효 배합의 위치다. **작은 하늘색 점**은 제조 가능한 설계,
**회색 원**은 근접 중복 적중, **검정 사각형**은 서로 δ 이상 떨어진 배합이다. 중복점을
사각형보다 크게 그려서, 뭉친 곳이 검정 사각형 주변의 회색 후광으로 보이게 했다.

축은 **유효 영역에 맞춘 공통 PCA 평면**이다. 각 method의 제안이 아니라 유효 영역에
적합했으므로 패널 간 비교가 성립한다. 제목에 그 평면이 설명하는 분산 비율을 적었다 —
L1~L4는 56~70%지만 **L5-4는 30%**이므로 그 그림의 2D 배치는 신뢰도가 낮다.

L3-1이 가장 선명하다. Point-target은 17.8 적중이 2.8개(16%), Range-Aware TB는 12.3이
1.3개(11%)로 줄고 회색 후광이 몇 군데 뭉쳐 있다. TRMC-BO full은 8.5 적중이 7.4개(87%)로
유효 영역 전반에 흩어져 있다.

### 지표 선택에 대한 고백

**주 지표를 결과를 본 뒤에 바꿨다.** 처음 보고는 hit rate를 헤드라인으로 썼고 그때는
Point-target이 1위였다. 정보 누출을 의심해 감사하는 과정에서 hit rate가 평가 횟수를
센다는 점이 드러나 distinct를 도입했다.

방어 논거는 두 가지다. 논문 §8.6이 이미 diversity를 지표로 명세했고, δ-uniqueness는
Range-Aware BO 논문에서 온 개념이라 사후에 발명한 지표가 아니다. 그래도 **주 지표로
승격한 시점은 데이터를 본 뒤**이므로, 두 지표를 모두 보고하고 어느 것이 주 지표인지는
논문이 명시적으로 주장해야 한다.

**hit rate가 무의미하지도 않다.** 실험 1회가 극히 비쌀 때 "제안이 얼마나 자주 규격을
맞추는가"는 그 자체로 중요하다. 두 지표는 다른 목적에 답하며, 이 결과는 **두 방법이
서로 다른 것을 최적화한다**는 것을 보여준다.

## 핵심 결과

5 tasks × 3 widths × 10 seeds 평균. Violation은 window width로 정규화했다.
Rank는 15개 (task, width) cell에 대한 average rank이며 낮을수록 좋다.

| Method | Hit rate | Found rate | First hit | Lower viol. | Upper viol. | Invalid | Avg rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random search | 0.033 | 0.66 | 14.8 | 0.453 | 0.536 | 0.267 | 6.93 |
| Standard BO (qEHVI) | **0.004** | 0.15 | 9.2 | 0.572 | 0.515 | 0.423 | **7.93** |
| Point-target BO | **0.220** | 0.85 | 6.0 | 0.179 | 0.247 | 0.196 | **1.47** |
| TRMC-BO (range prob. only) | 0.049 | 0.79 | 13.0 | 0.414 | 0.436 | 0.244 | 5.70 |
| TRMC-BO (+ constraints) | 0.096 | 0.85 | 8.5 | 0.320 | 0.391 | 0.249 | 4.00 |
| TRMC-BO (full) | 0.138 | **0.89** | 7.0 | 0.247 | 0.206 | 0.237 | 2.67 |
| Range-Aware TB (equal-vol.) | 0.105 | 0.57 | 3.5 | 0.197 | 0.151 | 0.140 | 4.20 |
| Range-Aware TB (inscribed) | 0.134 | 0.63 | 6.5 | 0.234 | 0.182 | 0.158 | 3.10 |

위 표의 hit rate와 rank는 **평가 횟수 기준**이므로 위의 "무엇을 세는가" 절과 함께
읽어야 한다.

### 누적 valid design 발견 수

| Method | it 1 | it 10 | it 20 | it 30 | it 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random search | 0.01 | 0.32 | 0.64 | 1.03 | 1.67 |
| Standard BO (qEHVI) | 0.02 | 0.09 | 0.17 | 0.18 | **0.19** |
| Point-target BO | 0.17 | 1.83 | 4.11 | 6.30 | **11.01** |
| TRMC-BO (range prob. only) | 0.05 | 0.57 | 1.12 | 1.61 | 2.47 |
| TRMC-BO (+ constraints) | 0.09 | 0.76 | 1.71 | 2.61 | 4.81 |
| TRMC-BO (full) | 0.12 | 1.09 | 2.10 | 3.50 | **6.88** |
| Range-Aware TB (equal-vol.) | 0.13 | 1.21 | 2.40 | 3.38 | 5.27 |
| Range-Aware TB (inscribed) | 0.14 | 1.31 | 2.55 | 3.86 | 6.70 |

Standard BO는 iteration 20 부근에서 곡선이 평평해진다. extremum 쪽으로 걸어나간
뒤 아무것도 찾지 못한다.

---

## 분해

### Window width별 (hit rate)

| Method | wide (10%) | medium (3%) | narrow (1%) |
| --- | ---: | ---: | ---: |
| Random search | 0.068 | 0.022 | 0.009 |
| Standard BO (qEHVI) | 0.009 | 0.002 | 0.001 |
| Point-target BO | 0.302 | 0.212 | 0.147 |
| TRMC-BO (range prob. only) | 0.087 | 0.037 | 0.024 |
| TRMC-BO (+ constraints) | 0.168 | 0.078 | 0.042 |
| TRMC-BO (full) | 0.197 | 0.122 | 0.093 |
| Range-Aware TB (equal-vol.) | 0.228 | 0.055 | 0.033 |
| Range-Aware TB (inscribed) | 0.258 | 0.088 | 0.057 |

**Width별 average rank** — 여기서 실제 추세가 보인다.

| Method | wide | medium | narrow |
| --- | ---: | ---: | ---: |
| Point-target BO | 2.00 | 1.20 | 1.20 |
| **TRMC-BO (full)** | **3.60** | **2.30** | **2.10** |
| Range-Aware TB (equal-vol.) | **2.40** | **4.40** | **4.80** |
| Range-Aware TB (inscribed) | 3.40 | 3.40 | 3.50 |

Window가 좁아질수록 TRMC-BO의 순위는 **올라가고**(3.60 → 2.30 → 2.10),
Range-Aware TB는 **내려간다**(2.40 → 4.40 → 4.80). wide에서 2위였던 TB가 narrow에서
5위다. target이 좁을수록 방법이 중요해진다는 논문의 예상과 일치한다. 다만
point-target BO도 함께 좋아지며 끝까지 앞선다.

### Window width별 (distinct, δ=0.1)

| Method | wide (10%) | medium (3%) | narrow (1%) |
| --- | ---: | ---: | ---: |
| Random search | 3.42 | 1.08 | 0.44 |
| Standard BO (qEHVI) | 0.44 | 0.08 | 0.06 |
| Point-target BO | 3.66 | 3.32 | 1.60 |
| TRMC-BO (range prob. only) | 4.24 | 1.70 | 0.98 |
| TRMC-BO (+ constraints) | 8.34 | 3.82 | 1.80 |
| **TRMC-BO (full)** | **9.54** | **5.74** | **3.94** |
| Range-Aware TB (equal-vol.) | 3.58 | 1.10 | 0.50 |
| Range-Aware TB (inscribed) | 2.36 | 1.04 | 0.72 |

TRMC-BO full이 **세 조건 전부**에서 1위이며, ablation 사다리도 세 조건 전부에서
단조다. hit rate 표보다 훨씬 깨끗하다.

### Task별 (hit rate)

| Method | L1-1 smooth | L2-1 noisy | L3-1 local invalid | L4-1 multimodal | L5-4 sparse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random search | 0.046 | 0.032 | 0.043 | 0.034 | 0.009 |
| Standard BO (qEHVI) | 0.000 | 0.003 | 0.009 | 0.005 | 0.003 |
| Point-target BO | 0.207 | 0.129 | 0.370 | 0.345 | 0.051 |
| TRMC-BO (range prob. only) | 0.057 | 0.057 | 0.067 | 0.047 | 0.020 |
| TRMC-BO (+ constraints) | 0.138 | 0.122 | 0.101 | 0.100 | 0.020 |
| TRMC-BO (full) | 0.144 | 0.173 | 0.188 | 0.161 | 0.021 |
| Range-Aware TB (equal-vol.) | 0.069 | 0.103 | 0.177 | 0.133 | 0.045 |
| Range-Aware TB (inscribed) | 0.102 | 0.109 | 0.217 | 0.200 | 0.042 |

Input-infeasible rate가 난이도 사다리를 따라 가파르게 오른다 — 0% / 0% / 20% /
35% / **65%**. L5-4에서는 제안된 formulation 셋 중 둘이 **아예 만들 수 없다.**
모든 method가 여기서 무너진다.

### Stability

Seed 표준편차 원값은 hit rate 크기와 교락된다(아무것도 못 찾는 method는 분산도
작다). 따라서 coefficient of variation(CV)이 공정한 비교다.

| Method | mean hit | seed std | **CV** |
| --- | ---: | ---: | ---: |
| TRMC-BO (+ constraints) | 0.096 | 0.051 | **0.52** |
| TRMC-BO (full) | 0.138 | 0.077 | **0.56** |
| Random search | 0.033 | 0.019 | 0.58 |
| TRMC-BO (range prob. only) | 0.049 | 0.031 | 0.62 |
| Point-target BO | 0.220 | 0.198 | 0.90 |
| Range-Aware TB (inscribed) | 0.134 | 0.148 | 1.10 |
| Range-Aware TB (equal-vol.) | 0.105 | 0.137 | 1.30 |
| Standard BO (qEHVI) | 0.004 | 0.006 | 1.64 |

TRMC-BO 계열이 가장 일관적이다. Point-target BO는 평균은 더 높지만 자기 평균
대비 변동이 약 1.6배 크다.

CV는 **평균이 0에 가까우면 의미를 잃는다**(분모가 작아져 값이 폭발). 표에서 Standard
BO의 1.64와 Random search의 0.58은 그 경우이므로 "불안정하다"로 읽으면 안 된다.
실제로 무언가를 찾는 method들끼리(TRMC-BO 계열, Point-target, Range-Aware TB)만
비교가 성립한다.


---

## Native protocol — MatFormBench 자체 채점 (완료)

**다른 질문이다.** 위의 sweep은 순차적이다 — 한 번 제안하고 결과를 보고 다시 제안한다.
Native protocol은 초기설계에서 **한 번에 100개 제안을 배치로 뽑고** 피드백 없이
MatFormBench 자체 복합 점수로 채점한다(recommend / top-k 5라운드 / design-set-size
민감도 / 내부 seed 10개 stability). 우리는 제안만 제공하고 **모든 지표는 MatFormBench의
컴파일된 scorer가 계산한다.** 그래서 이 숫자는 그 벤치마크가 함께 배포하는 baseline과
직접 비교 가능하다.

5 tasks × 3 seeds. **TRMC-BO의 ablation 두 단계만 돌렸다** — 이 프로토콜에서는 타
baseline을 돌리지 않았으므로 아래는 우리 구성요소 간 비교이고 경쟁 method와의 비교가
아니다.

| Method | Total | Success | Efficiency | Explore | Robust | Stability | HV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRMC-BO (range probability only) | 57.19 | 0.520 | 0.741 | 0.791 | 0.489 | 0.395 | 0.868 |
| **TRMC-BO (full)** | **64.90** | **0.601** | **0.812** | **0.805** | **0.599** | **0.455** | **0.876** |

C2+C3를 켜면 **모든 하위 점수가 올라간다.** 순차 sweep의 ablation 순서와 같은 방향이며,
채점 주체가 우리가 아니라 벤치마크라는 점에서 독립적인 확인이다.

task별로 보면 5개 중 4개에서 full이 앞선다.

| Task | C1 only | full | 차이 |
| --- | ---: | ---: | ---: |
| L1-1 | 59.77 | **77.58** | +17.81 |
| L2-1 | 73.00 | **79.74** | +6.74 |
| L3-1 | 42.81 | **55.68** | +12.87 |
| L4-1 | 74.33 | **77.81** | +3.48 |
| L5-4 | **36.05** | 33.72 | −2.33 |

**L5-4는 여기서도 벽이다.** 두 변형 모두 36점 아래이고 full이 오히려 조금 낮다. seed 3개
차이로 결론 내릴 수는 없지만, 순차 sweep에서 L5-4가 제안의 65%를 제조 불가로 만든 것과
같은 방향이다.

## COMBOO (constrained BO baseline) — 보류

**이번 보고에서 제외한다.** 5D task 3개(90 cells)는 완주했고 수치도 있지만, 그
수치를 신뢰할 근거가 아직 부족하다.

관측된 것은 이렇다. COMBOO가 50 iteration 중 평균 31회 infeasibility를 선언하고
model-guided 제안을 포기하며, hit rate 0.007로 random search(0.040)보다 5배 나쁘다.
Auxiliary probe의 최적값이 매번 `-ref[y2]`와 같아 `max_x UCB(y2) ≈ 0`이다.

**검증된 것과 검증되지 않은 것을 구분해야 한다.**

- 검증됨: 알고리즘 이식. 저자들 자신의 branin-currin 설정에서 3 seeds × 30
  iterations를 선언 0회로 완주하고 hypervolume이 0 → 약 62로 상승한다
  (`eval/scripts/verify_comboo_port.py`).
- 검증됨: acquisition 최적화. 최적값이 20,000점 brute-force 탐색과 소수점 4자리까지
  일치한다.
- **검증되지 않음: box → COMBOO 매핑.** 저자들의 `ref`는 느슨한 성능 하한인데, 우리
  구현은 좁은 window의 아래끝을 hypervolume 기준점과 constraint 임계값에 **동시에**
  쓴다. 이 선택 때문에 "기준점 위로의 개선" 신호가 +0.03 수준으로 얇아졌을 수 있고,
  그렇다면 부진의 원인은 COMBOO가 아니라 우리 매핑이다.

세 번째가 결론을 좌우한다. 검증 전에는 "COMBOO가 이 과제에서 약하다"와 "우리가
COMBOO에게 잘못된 목표를 줬다"를 구분할 수 없다.

**재개할 때 할 일**: hypervolume 기준점을 constraint 임계값과 분리해(예: 관측
분포의 하위 quantile) 다시 측정하고, 그때도 같은 거동이면 그것이 결과다. 고차원
simplex task는 cell당 시간이 아니라 일 단위이므로 별도 계획이 필요하다.

코드(`eval/methods/comboo.py`), 검증 스크립트, 완료된 90 cells는 모두 보존되어 있다.

## COMBOO 재개 -- 매핑을 분리했더니 두 번째 버그가 드러났다

hypervolume 기준점(`ref`)을 관측 분포의 하위 10% quantile로 분리했다(`HV_REF_QUANTILE`,
`comboo.py`의 "Mapping fix" 절). 그런데 그 직후 **완전히 다른 실패 모드**가 나타났다:
stage 2의 `optimize_acqf`가 거의 매 iteration `batch_initial_conditions`가 제약을
만족하지 않는다며 예외를 던지고(L1-1: 50회 중 ~49회, L2-1: ~48회, L3-1: ~34회),
`propose()`의 `except`가 이를 조용히 삼켜 probe의 점을 그대로 반환하고 있었다 --
즉 COMBOO의 "본 획득함수"는 거의 실행되지 않았고, 결과는 probe의 argmax를 반복
제안한 것에 가까웠다.

원인은 probe(`AuxiliaryUCB`, 저자들의 Algo. 2)가 **한쪽만** 확인한다는 데 있다.
`ref == lower`였던 예전 코드에서는 이게 어쩌다 stage 2의 편측 제약과 우연히
일치해 seed로 그럭저럭 통했지만, `ref`를 분리하자 probe가 확인하는 조건(느슨한
`ref` 위)과 stage 2가 요구하는 조건(실제 window의 상·하한 양쪽)이 완전히
갈라졌다. `TwoSidedAuxiliaryUCB`를 새로 추가해 probe가 두 물성 경계를 모두
확인하도록 고쳤다 -- `optimistic_window_constraints`가 만드는 항 전부를 하나의
`min`으로 묶은 것과 같다. 이제 stage 2로 넘기는 seed는 항상 그 제약을 만족하고,
90 cells를 재실행한 결과 `cnt_comboo_constrained_solve_failed`는 한 번도
기록되지 않았다 (이전에는 사실상 매 iteration).

**결과는 바뀌지 않았다 -- 이유가 바뀌었을 뿐이다.**

| task | hit rate (M8_comboo) | hit rate (M0_random) | declared infeasible (/50) | n_unique |
| --- | ---: | ---: | ---: | ---: |
| L1-1 | 0.003 | 0.061 | 48.0 | 0.13 |
| L2-1 | 0.011 | 0.032 | 42.4 | 0.53 |
| L3-1 | 0.010 | 0.043 | 16.9 | 0.50 |

probe를 두 방향 모두 확인하게 고치자 **declared-infeasible 비율이 오히려
급등했다** -- 세 물성 전부에 대해 양쪽 optimistic bound를 동시에 클리어하는
지점을 찾는 게, 한쪽만 보던 예전 probe보다 훨씬 어려운 요구이기 때문이다.
L1-1/L2-1는 여전히 y2의 배치효과 노이즈(위 "관측된 것은 이렇다" 참고)가
벽으로 작용하고, L3-1은 그보다는 낫지만 여전히 random보다 크게 처진다.

**이제는 신뢰할 수 있는 결과다.** 매핑 버그와 seed-검증 버그를 모두 고친
뒤에도 COMBOO는 random search보다 4~20배 나쁘고 고유 후보 수도 1개 미만으로
붕괴한다 -- probe가 거의 항상 infeasible을 선언해 본 획득함수 자체가 거의
실행되지 않기 때문이다. §10.2의 논지("constrained BO는 range satisfaction을
별도 objective에 종속시킨다")를 지지하는 방향의 결과지만, 근거가 "약한 성능"이
아니라 "좁은 두-방향 window에서는 optimistic feasibility probe 자체가 거의
항상 실패한다"는 훨씬 구체적인 메커니즘이라는 점을 명시해야 한다. `L4-1`/`L5-4`는
여전히 계산 비용 때문에 범위 밖이다 (위 config 주석 참고).

## 이 결과가 뒷받침하는 것과 뒷받침하지 않는 것

**뒷받침되는 것.**

- **각 구성요소가 기여한다.** hit rate 0.049 → 0.096 → 0.138이고, found rate,
  first hit, 양쪽 violation, diversity, 그리고 개별 width 전부에서 같은 순서가
  나온다. 이 사다리에는 모호한 구석이 없다.
- **Extremum-seeking BO는 잘못된 도구다.** 0.004 대 random search 0.033 — 찍는
  것보다 8배 나쁘고, iteration 20 이후로는 개선이 멈춘다. 논문의 전제가 성립한다.
- **TRMC-BO가 CV 기준으로 가장 일관적이다.**
- **Window가 좁아질수록 TRMC-BO의 상대 위치가 개선된다.** 이 방법이 설계된
  regime이 바로 그쪽이다.

**뒷받침되지 않는 것.**

- **여기서 TRMC-BO는 최고 성능이 아니다.** Point-target BO가 모든 width, 모든
  task에서 hit rate로 앞서고 종합 1위다(1.47 대 2.67).
- **왜 그런지는 아직 설명하지 못한다.** 아래 내용은 전부 가설이며 finding이 아니다.

### 정보 누출은 배제했다 — 재현 가능한 감사로

`eval/scripts/audit_leakage.py`가 네 가지 검사를 8개 method 전부에 대해 실행한다.
L1-1(5D box), L4-1(10D simplex), L5-4(15D sparse) 세 task에서 **전부 통과**했다.

| 검사 | 내용 | 결과 |
| --- | --- | --- |
| A1 | `oracle.truth`를 쓰레기값으로 바꿔도 제안이 비트 단위 동일한가 | pass (deviation 0) |
| A2 | 같은 seed에서 초기설계가 method 간 바이트 단위 동일한가 | pass |
| A3 | oracle 평가 횟수가 `n_init + budget`으로 동일한가 | pass |
| A4 | 각 method가 실제로 읽은 `target_info` 필드 | 아래 표 |

A1이 성립하려면 캠페인이 재현 가능해야 하는데 그렇지 않았다 — acquisition 최적화가
torch 전역 generator를 쓰는데 아무도 seed하지 않았다. `loop.py`가 이제 campaign마다
seed한다. 완료된 1200 cell이 무효가 되는 건 아니고(유효한 draw다), 앞으로 같은 cell을
다시 돌리면 재현된다는 뜻이다.

A4가 원래 의심을 논증이 아니라 측정으로 정리한다.

| Method | 읽은 필드 |
| --- | --- |
| Random search | (없음) |
| Standard BO | `obj`, `weight` |
| **Point-target BO** | **`lb`, `obj`, `ub`, `weight`** |
| **TRMC-BO (전 변형)** | **`lb`, `obj`, `ub`, `weight`** |
| Range-Aware TB | `lb`, `obj`, `ub`, `weight` + `ball_center`, `ref_center`, `ref_scale` |

**Point-target이 읽는 집합은 TRMC-BO와 정확히 동일하다.** 더 읽는 것은 Range-Aware
TB뿐이며, 이는 ball을 정의하려면 중심과 축척이 필요해서다(`y_valid_centroid` 등 동결된
calibration 스칼라). 설계별 정보가 아니라 method-무관 목표 정의이지만 box 정의 `[L,U]`
보다 많은 정보이므로, **논문에 명시해야 한다.** 이제 이것이 유일한 비대칭임을
`eval/tests/test_fairness.py`가 assert한다.

### 대신 우리 구현에서 명세 불일치를 찾았다

논문 §4.2는 확률의 **곱**을 명세한다.

$$A_{\mathrm{range}}(x)=\prod_k P\big(L_k\le f_k(x)\le U_k\big)$$

구현은 그렇게 하지 않는다. `CDFRangeMultiOutputObjective`가 **출력별 확률의 벡터**를
반환하고 `get_acqf`가 그 벡터를 **qEHVI**에 넘긴다. hypervolume은 Pareto 확산을
보상하므로 $P=(0.9,\,0.01,\,0.5)$ 같은 후보도 hypervolume을 개선한다. 세 물성을
동시에 창 안에 넣어야 하는 우리 과제에서 그 후보는 결합확률 0.45%로 쓸모가 없다.
곱은 그런 식으로 속지 않는다 — 한 인자가 작으면 전체가 가라앉는다.

`eval/scripts/acquisition_auc.py`가 결과를 측정한다. surrogate 하나를 동결하고 후보
600개를 뽑아 각 acquisition으로 순위를 매긴 뒤, 참값 합격 여부에 대한 AUC를 계산한다
(0.5 = 무작위 섞기와 동등, 1.0 = 합격 후보가 전부 불합격 후보보다 위).

15개 설정 × 5 seed 평균:

| Acquisition | 평균 AUC | 15행 중 최고인 횟수 |
| --- | ---: | ---: |
| **product, 직접 (§4.2 명세)** | **0.641** | **11** |
| point-target distance qEI | 0.608 | 3 |
| qEHVI over P vector (현 구현) | 0.542 | 1 |
| product, qEI로 감싼 것 | 0.514 | 0 |

**출하 중인 형태는 후보를 섞는 것보다 겨우 나은 수준으로 순위를 매긴다.** 그리고 이것이
baseline을 설명한다 — point-target의 목적함수도 결합적이므로 곱과 비슷하게 순위를
매긴다. 우리가 잃고 있던 것은 range라는 아이디어가 아니라 그것을 결합하는 방식이다.

두 가지 단서를 붙인다. 첫째, 이것은 **1-step 순위 진단**이며 50 iteration 탐색에
대해서는 아무 말도 하지 않는다. 순위를 잘 매기는 acquisition도 갇힐 수 있다. 둘째,
`product, qEI`가 정확히 0.514(≈0.5)인 것은 우연이 아니다. 곱은 $x$의 결정론적 함수라
$E[(A-\text{best})^+]$가 $(A-\text{best})^+$로 붕괴하고, incumbent 미만의 모든 후보가
정확히 0으로 동점이 된다. 출하 중인 단일출력 경로(`CDFRangeObjective` + qEI)도 구조가
같다.

#### 캠페인 결과 — 순위가 좋아지고 다양성이 무너진다

**결론이 났으므로 스윕은 44 cell에서 의도적으로 중단했다.** 곱 직접 형태의 거동이 세
task에서 같은 방향으로 확인됐고, 남은 비용의 약 69/75 CPU-h가 그 변형이었다. 설정
파일과 코드는 남겨 두었으니 재개는 한 명령이다.

**(1) L1-1 / wide+medium, 5 seed, 전 method 짝 맞춤**

| Method | 적중 | **distinct** | 비율 |
| --- | ---: | ---: | ---: |
| TRMC-BO (full, product in qEI) | 10.00 | **10.00** | 1.00 |
| TRMC-BO (full) — 현 구현 | 9.10 | 8.80 | 0.97 |
| TRMC-BO (C1만) — 현 구현 | 4.80 | 4.50 | 0.94 |
| Point-target BO | 14.40 | 3.90 | 0.27 |
| TRMC-BO (product in qEI, C1만) | 3.80 | 3.70 | 0.97 |
| **TRMC-BO (product 직접, C1만)** | **10.10** | **1.80** | **0.18** |

**(2) 단일 cell 정밀 대조 (task/medium/seed0).** 핵심 질문은 곱 + C2 + C3가 attractor를
상쇄하는지였고, 답은 아니다.

| Task | Method | 분 | 적중 | **distinct** | 제안간 거리 |
| --- | --- | ---: | ---: | ---: | ---: |
| L1-1 | TRMC-BO (full) — 현 구현 | 13.2 | 14 | **13** | 0.738 |
| L1-1 | **TRMC-BO (full, product 직접)** | **56.5** | 11 | **2** | **0.105** |
| L1-1 | TRMC-BO (product 직접, C1만) | 29.3 | 8 | **1** | 0.063 |
| L4-1 | TRMC-BO (full) — 현 구현 | 15.0 | 8 | **8** | 0.829 |
| L4-1 | Point-target BO | 10.0 | 28 | 6 | 0.610 |
| L4-1 | **TRMC-BO (full, product 직접)** | **71.3** | 27 | **1** | **0.122** |
| L4-1 | TRMC-BO (product 직접, C1만) | 11.5 | 1 | 1 | 0.471 |

L4-1에서 곱은 **적중 27개를 distinct 1개로** 붕괴시킨다. 같은 cell에서 현 구현은 8개
적중이 8개 distinct이고, point-target조차 28 → 6이다.

**기전.** 곱은 세 posterior mean이 모두 창 중앙에 오고 분산이 낮은 **단일 내부점에서
최대**가 되는 unimodal attractor다 — point-target의 quadratic distance와 구조가 같고 더
날카롭다. 그래서 후보 순위는 잘 매기고(AUC 0.641 대 현 구현 0.542) 캠페인에서는 한
basin으로 수렴한다. qEHVI는 Pareto 확산을 보상하므로 그 반대다. **C2/C3는 이것을
상쇄하지 못한다** — 제약은 어디가 허용되는지만 정하고, 그 안에서 어디로 갈지는
acquisition이 정한다.

**그러므로 §4.2와 구현의 불일치는 "구현이 틀렸다"가 아니다.** 두 형태가 서로 다른 것을
최적화하며 **우리 주 지표에 맞는 쪽은 구현이다.** 논문이 §4.2에서 곱을 명세한 것을 다시
봐야 한다 — 다양성을 주장하면서 명세는 그 주장과 어긋난다.

앞선 AUC 절에 붙인 단서가 실제로 물렸다. 1-step 순위는 탐색을 대변하지 못한다. 곱은
순위를 더 잘 매기고 탐색은 더 못한다. 그 진단이 쓸모없진 않았다 — point-target이 왜
강한지 설명했고(그 목적함수도 결합적이라 곱과 비슷하게 순위를 매긴다) hit rate 방향도
맞게 예측했다. 다만 우리 주장에 맞는 종점을 재지 않았다.

**(3) 곱을 qEI로 감싸면 오히려 현 구현을 조금 앞선다.** `M5q`가 distinct 10.00 대
8.80(L1-1 wide+medium 5 seed), 단일 cell에서 15 대 13이다. plateau 때문이다 —
incumbent 미만이 전부 0으로 동점이어서 acquisition이 한 점으로 끌지 못하고 C2/C3와
restart가 일을 한다. **비용도 같다**(7.4 대 7.5 s/iter). 후속으로 볼 값이 있는 유일한 곱
변형이며, 전체 매트릭스로는 확인하지 않았다.

**(4) 비용.** 곱을 직접 최적화하면 4~5배 비싸다.

| 변형 | L1-1 | L4-1 | L5-4 |
| --- | ---: | ---: | ---: |
| product 직접 (C1+C2+C3) | 56.5분 | 71.3분 | 4.3분 |
| product 직접 (C1만) | 29.3분 | 11.5분 | 4.3분 |
| product in qEI (C1+C2+C3) | 12.2분 | — | — |
| 현 구현 (C1+C2+C3) | 13.2분 | 15.0분 | 7.1분 |

이는 plateau의 이면이다 — qEI로 감싼 변형은 동점 때문에 L-BFGS가 즉시 끝나고, 직접
형태는 어디에나 기울기가 있어 128 restart가 실제로 수렴까지 간다. 고차원 task가 싼 것은
쉬워서가 아니라 **퇴화**해서다(L5-4는 초기 30점 중 라벨이 2개뿐이라 posterior가 거의
평평하다).

#### L5-4에서 곱은 제조 가능한 배합을 하나도 못 낸다

L5-4/medium seed0에서 같은 초기설계, 같은 라벨 2개를 받는데 결과가 갈린다.

| Method | feasible | 제안간 평균거리 | floor에 붙은 성분수 |
| --- | ---: | ---: | ---: |
| **product 직접 (C1만)** | **0.00** | **0.234** | **0.7/6** |
| **product 직접 (C1+C2+C3)** | **0.00** | **0.231** | **0.6/6** |
| TRMC-BO (C1만) | 0.48 | 1.157 | 0.2/6 |
| TRMC-BO (full) | 0.16 | 1.119 | 0.1/6 |
| Point-target BO | 0.46 | 1.187 | 0.0/6 |
| Random search | 0.30 | 1.207 | 0.0/6 |

**거의 평평한 posterior에서도 곱은 단일 영역으로 붕괴한다**(0.23 대 1.12~1.21). 같은
2점을 받은 다른 method들이 흩어지므로 데이터가 아니라 목적함수의 성질이다.

그 영역이 MatFormBench의 `SOFT_CONSTRAINT`를 위반해 **50/50 전부 제조 불가**가 된다
(다른 method는 50개 중 11~18개). 곱 변형은 simplex 성분을 floor(0.01)로 밀어붙인다.

**우리 제약 처리의 결함은 아니다.** 곱 변형의 제안도 simplex 등식을 2e-16까지 만족하고
min_component floor를 하나도 위반하지 않는다. `SOFT_CONSTRAINT`의 정확한 정의는
MatFormBench의 컴파일된 `.so` 안에 있고 문서화되어 있지 않다.

**단서.** L5-4는 모든 method에게 벽이다(전 method valid 0~1). 이 cell만으로 일반화할 수
없다. 다만 mixture task에서 §4.2 형태가 퇴화 조성으로 몰릴 수 있다는 것은 실제 재료
문제에서 중요한 실패 양상이다.

#### 이 절의 통계적 무게

| 주장 | 근거 |
| --- | --- |
| 곱은 hit rate를 올린다 | L1-1 wide+medium 10 cell (0.077 → 0.202) |
| 곱은 다양성을 무너뜨린다 | L1-1 10 cell (4.50 → 1.80) + 단일 cell 2개 (13→2, 8→1) |
| C2/C3가 상쇄하지 못한다 | 단일 cell 2개 (L1-1, L4-1) |
| 곱을 qEI로 감싸면 조금 낫다 | L1-1 wide+medium 10 cell + narrow 4 cell |
| 곱은 4~5배 비싸다 | 단일 cell 5개 |
| L5-4에서 곱은 제조 불가만 낸다 | 단일 cell 2개 |

**단일 cell 근거는 오차범위가 없다.** 방향이 세 task에서 일치하고 효과 크기가 크지만
(27 → 1), seed 1개에서 나온 수치를 논문에 그대로 쓸 수는 없다. 논문에 넣으려면 최소한
L1-1·L3-1에 5 seed씩은 필요하고, 그것은 `matformbench_product.yaml`의 tasks를 줄여
30분 안에 얻을 수 있다.

## 열린 질문 / 다음 단계

1. **[해결] 왜 중앙을 겨냥하는 것이 hit rate에서 나은가?** **결합적 목적함수는 hit
   rate를 올리고 다양성을 죽인다.** 곱(§4.2 명세)과 point-target distance는 둘 다
   unimodal attractor이고, 둘 다 적중은 많고 distinct는 적다(비율 0.12~0.18, 0.27).
   출력별 확률을 hypervolume으로 결합하는 현 구현은 순위를 잘 못 매기지만(AUC 0.542)
   흩어진 해를 만든다(비율 0.93~0.97). C2/C3는 이를 상쇄하지 못한다.
   **→ 논문 §4.2의 곱 명세를 hypervolume 결합으로 고쳐야 한다.** 지금 논문은 우리가
   실제로 돌린 것도 아니고 우리 주장을 뒷받침하는 것도 아닌 식을 적고 있다.
2. **`M5q`(곱을 qEI로 감싼 것 + C2/C3)를 전체 매트릭스로 확인할 값이 있다.** L1-1
   14 cell에서 현 구현을 distinct 10.00 대 8.80으로 앞서고 비용은 같다. 곱 변형 중
   유일하게 유망하다. 300 cell 중 이 변형만 75 cell이면 약 10 CPU-h다.
2. **순위가 뒤집히는 지점이 있는가?** Window가 좁아질수록 TRMC-BO가
   point-target을 따라잡는다(rank 격차 1.60 → 1.10 → 0.90). 더 좁은 조건(0.3%?)이
   이 추세가 계속되는지 정체되는지 시험해 줄 것이다.
3. **비대칭 window.** 지금 window는 전부 quantile 규칙에서 나와 자기 질량 기준으로
   대체로 대칭이다. midpoint target이 무너져야 할 곳은 의도적으로 비대칭인
   window이며, 그쪽이 실제 재료 specification에 더 가깝다.
4. **L5-4는 모두에게 벽이다**(제안의 65%가 제조 불가). Method 비교가 아니라
   benchmark 자체의 성질로 보고할 사안이다.
5. **COMBOO 재개** — hypervolume 기준점을 constraint 임계값과 분리해 재측정하는 것이
   먼저다. 고차원 simplex task는 cell당 일 단위이며, `num_restarts`를 줄여 맞추는 것은
   안 된다(probe 최적값이 +0.03로 얇아 거짓 infeasibility 선언이 생긴다).
6. **native protocol에서 baseline을 돌리지 않았다.** 지금은 우리 구성요소 간 비교뿐이다.
   Point-target을 이 프로토콜에 얹는 것은 acquisition factory 하나만 바꾸면 되므로,
   벤치마크 자체 채점에서도 순위가 같은지 확인할 가치가 있다.
7. **비트 재현성**은 이제 확보됐지만(campaign마다 torch seed) 1200 cell은 그 이전에
   돌았다. 재현 스크립트로 다시 돌리면 통계는 같고 개별 cell의 좌표는 다를 수 있다.

## 재현

```bash
cd TRMC-BO

# 주 스윕 (1200 cells)
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/matformbench_range_adapted.yaml --workers 22

# 공정성 감사 — task 하나에 몇 분
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.audit_leakage --task L4-1

# acquisition 순위 진단 — 캠페인 없이 15 설정 × 5 seed
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.acquisition_auc --seeds 0 1 2 3 4

# 공정성 불변식 테스트 (전체 152개 중 11개)
PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests -q
```

환경 구성, protocol 결정 사항, 그리고 우회해야 했던 MatFormBench upstream defect
2건은 `eval/README.md`에 문서화되어 있다. 동결된 target window는
`eval/specs/matformbench/`에 commit되어 있다. 모든 실행은 resumable이며,
재실행하면 완료된 cell은 건너뛴다.

## 부록 -- δ-capacity를 처음 쟀다 (2026-08-27)

주 지표가 "서로 δ 이상 떨어진 유효 배합의 수"인데 **이 suite의 분모를 한 번도 재지
않았다.** Olympus(21~77)와 HOIP(7)에는 있고 주 suite에만 없었다.
`valid_region_geometry.py --suite matformbench` (400,000점 균일 표본, δ=0.1):

| task | wide | medium | narrow |
| --- | ---: | ---: | ---: |
| L1-1 (5D) | 2529 | 1759 | 1117 |
| L2-1 (5D) | 2527 | 1790 | 1247 |
| L3-1 (5D) | 2270 | 1560 | 896 |
| L4-1 (10D simplex) | 3761 | 3412 | 1769 |
| L5-4 (15D) | 4000* | 3745 | 1274 |

`*` 표본 기준 하한.

**세 suite가 질적으로 다른 영역에 있다.** branin(2D)은 천장 21이 실제로 가깝고
(우리가 58% 회수), HOIP은 열거 가능해서 7이 진짜 목표다. 여기서는 896~4000이라
**어떤 method도 천장 근처에 못 간다** — 50회 예산에 6.41개는 천장의 0.4%다.

그래서 이 suite에서 분모는 "얼마나 덮었나"를 재는 게 아니라 **천장 효과가 없다는
것을 보증한다.** method 간 차이가 포화가 아니라 탐색의 차이라는 뜻이다.

**다만 주의할 점이 있다.** 고차원에서 δ=0.1은 잘 안 걸린다 — L5-4 medium/narrow는
capacity가 n_valid와 **정확히 같다**(3745/3745, 1274/1274). 즉 표본된 유효 설계가
서로 전부 δ 이상 떨어져 있다. 스스로 반복하지 않는 method에게는 distinct ≈ valid가
된다.

그런데도 지표가 일을 한다: 같은 suite에서 point-target의 concentration은 0.26,
TRMC-BO는 0.93이다. 잡히는 중복은 실제로 거의 동일한 제안이다. 그리고 δ를 0.4까지
올려도(15차원에서도 상당한 거리) 순위가 그대로다 (full 3.79 대 point-target 1.95).
논문은 이 두 가지를 함께 적어야 한다.
