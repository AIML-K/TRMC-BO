# Olympus 결과 — benchmark 2 of 3

**상태**: 완료 — branin 270 cell + emulator 810 cell(3 mixture task), **오류 0**.
**모든 예측은 실행 전에 커밋**되어 있다([predictions_olympus.md](eval/predictions_olympus.md),
커밋 `91746f0`; task 교체 경위는 같은 파일의 부록).

**한 줄 요약: 주 지표(서로 다른 규격 만족 설계 수)에서 TRMC-BO가 4개 task 전부,
3개 window width 전부, 4개 δ 전부에서 1위다.** Range-Aware TB는 50회 중 41회를 창에
넣고 설계 1개를 돌려준다. 다만 §2에서 단일 출력의 C1 acquisition이 포화해 소멸하는 것을
측정했고, §8.4에서 TRMC-BO가 input feasibility를 학습하지 않는 것을 확인했다 — 후자는
Olympus로는 시험할 수 없고 MatFormBench에만 근거가 있다.

**요약.** 서로 다른 규격 만족 설계 수(δ=0.1)로 TRMC-BO full이 **12.17개**로 1위다.
Point-target BO는 9.23개, Range-Aware TB는 **1.7~1.8개**다. 세 window width 전부와
네 개의 δ 전부에서 TRMC-BO full이 1위이고, ablation 사다리는 단조다
(2.70 → 11.67 → 12.17).

**적중 횟수로 보면 순위가 거꾸로 선다.** Range-Aware TB는 50회 중 **49.0회**를 창 안에
넣는다 — 거의 완벽하다 — 그런데 그것이 **서로 다른 설계 1.7개로 붕괴**한다(concentration
0.035). 같은 표에서 TRMC-BO full은 35.3 적중이 12.17개로 남는다(0.344). 평균순위가
지표에 따라 완전히 뒤집힌다: hit rate로는 TB가 1.5~2.7위이고 TRMC-BO가 5.3위,
diversity로는 TRMC-BO가 1.3위이고 TB가 5~6위다.

**그리고 예상 밖의 기전을 찾았다 — 단일 출력에서 C1의 acquisition은 스스로 소멸한다.**
아래 §2가 본론이다. 요지는 M3/M4/M5의 성능 차이가 **전부 C2와 C3에서 나온다**는 것이다.
C1은 10~25 iteration 안에 완전히 평평해져서 아무 신호도 주지 않는다.

---

## 1. 무엇을 돌렸는가

- **Task**: Olympus `branin` — 2차원 analytical surface, box 제약, 출력 1개.
  전역 최소가 3개라 창을 두면 valid 영역이 **연결성분 4/3/3개**(wide/medium/narrow)로
  갈라진다. 측정값이며 가정이 아니다(grid ≥ 1400; grid 700에서 narrow가 75개로 나오는
  것은 격자가 얇은 shell을 자르는 인공물이다).
- **의도적으로 무잡음**이다(`observe` == `truth`). MatFormBench의 모든 task가
  noisy이므로, acquisition 비교를 잡음 처리와 분리한다. Range-Aware BO 논문도 Branin을
  이렇게 쓴다.
- **사양은 구성한 것**이다. Olympus는 `minimize`만 준다. 균일 측도에서
  `tau = q25(response)`를 상한으로 동결하고, 기존 보정기가 하한을 놓아
  input-feasible 설계 중 합격 비율이 10%/3%/1%가 되게 했다(달성 9.99/3.04/0.98%).
  q25는 선택이 아니다 — m=1에서 보정은 해석적(`valid = P(y≤tau)·alpha`)이므로 tau를
  목표 분위수에 두면 alpha=1이 되어 가장 넓은 창이 one-sided native target으로
  붕괴한다.
- **Protocol**: initial design 30점, 순차 평가 50회, seed 10개, `num_restarts` 128,
  9 methods × 3 widths × 10 seeds = 270 cell. 오류 0.

**δ=0.1 capacity ≈ 21개.** 균일 표본에서 서로 δ 이상 떨어진 valid 설계가 최대 몇 개
존재하는지를 먼저 측정했다. 이 분모 없이는 "12개를 찾았다"를 읽을 수 없다. TRMC-BO
full은 천장의 **58%**, point-target은 44%, TB는 8%를 회수한다.

## 2. 본론 — 단일 출력에서 C1 acquisition은 포화되어 소멸한다

m=1에서 C1은 출하 중인 `CDFRangeObjective` + qEI 경로를 탄다. **이 경로는 지금까지 한
번도 실행된 적이 없다** — MatFormBench는 모든 task가 m=3이라 qEHVI 분기를 탄다.

그 목적함수는 x의 결정론적 함수라서 posterior 표본에 대한 기대값이 평균낼 것이 없고,
qEI는 자기 피적분함수로 붕괴한다:

    A_C1(x) = max(0, P(L ≤ f(x) ≤ U) − best_f)

수치로 확인했다(6e-9 이내). 그리고 `best_f`는 **관측점에서의 P의 running max**이고 P는
[0,1]에 갇혀 있다. 따라서 창 안의 점을 한 번 관측해 그 지점의 P가 1에 가까워지면,
`best_f ≈ 1`이 되어 **설계공간 전체에서 acquisition이 정확히 0**이 된다. 되돌릴 수 없다.

`eval/scripts/plateau_fraction.py`로 완료된 cell의 궤적에서 surrogate를 재적합해
측정했다(20,000점 균일 probe, branin/medium/seed0):

| iteration | best_f | P ≤ best_f 인 probe 비율 |
| ---: | ---: | ---: |
| 0 (초기설계 후) | 0.5126 | 99.88% |
| 10 | 0.9242 | **100.00%** |
| 20 | 0.9995 | **100.00%** |
| 50 | 0.9999 | 99.98% |

M4와 M5도 같다(25 iteration에서 `best_f = 1.0000`, plateau 100.00%). 세 변형이 **동일한
평평한 지형**을 본다.

그런데 성능은 2.70 / 11.67 / 12.17개로 갈린다. 즉 **이 task에서 C1은 사실상 불활성이고,
TRMC-BO의 성능은 전부 C2(양측 제약)와 C3(posterior-mean 필터 restart)에서 나온다.**
직관과 반대다 — C1이 엔진이고 C2/C3가 보정이라고 읽기 쉬운데, 단일 출력에서는 그렇지
않다.

**등록한 예측 P1의 검증**: `box` mapping의 TB는 같은 창을 target으로 하되 hinge가 없다
(직접 계산과 6.7e-16 일치, 두 독립 구현의 교차검증도 된다). hinge가 hit rate를 diversity와
교환한다는 예측이었고, 결과는 예측보다 강하다.

| | 적중 | **distinct** | concentration |
| --- | ---: | ---: | ---: |
| M3 TRMC-BO (C1만, hinge 있음) | 3.67 | **2.70** | 0.736 |
| M6c Range-Aware TB (같은 창, hinge 없음) | 48.93 | **1.70** | 0.035 |

hinge를 떼면 적중이 **13배**로 늘고 distinct는 **줄어든다**. hinge 없는 형태는 P를 직접
최대화하므로 포화하지 않고, 그래서 매 iteration 같은 최적점으로 돌아간다.

## 3. 주 결과 — 서로 다른 설계를 몇 개 찾았는가

3 widths × 10 seeds 평균. `conc.` = distinct/valid.

| Method | **distinct** | valid | conc. | 1st hit | viol | s/iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random search | 2.00 | 2.33 | 0.86 | 27.1 | 14.94 | 0.0 |
| Standard BO (extremum-seeking) | **0.73** | 0.73 | 1.00 | 36.5 | 11.91 | 1.3 |
| Point-target BO | 9.23 | 39.97 | 0.23 | 2.0 | 0.007 | 1.4 |
| TRMC-BO (range prob. only) | 2.70 | 3.67 | 0.74 | 9.3 | 14.25 | 1.3 |
| TRMC-BO (+ constraints) | 11.67 | 32.00 | 0.36 | 4.1 | 0.300 | 0.5 |
| **TRMC-BO (full)** | **12.17** | 35.33 | 0.34 | 2.8 | 0.010 | 1.9 |
| Range-Aware TB (equal-vol.) | 1.80 | **49.03** | **0.037** | 1.8 | 0 | 6.0 |
| Range-Aware TB (inscribed) | 1.70 | 49.10 | 0.035 | 1.8 | 0 | 5.8 |
| Range-Aware TB (box, hinge 없음) | 1.70 | 48.93 | 0.035 | 1.8 | 0 | 5.4 |

**Extremum-seeking BO는 random search보다 2.7배 나쁘다**(0.73 대 2.00). MatFormBench에서
8배 나빴던 것과 같은 방향이다. 논문의 전제가 두 벤치마크에서 성립한다.

### width별 distinct

| Method | wide (10%) | medium (3%) | narrow (1%) |
| --- | ---: | ---: | ---: |
| Random search | 4.3 | 1.3 | 0.4 |
| Standard BO | 2.2 | 0.0 | 0.0 |
| Point-target BO | 11.0 | 9.6 | 7.1 |
| TRMC-BO (C1만) | 4.2 | 2.6 | 1.3 |
| TRMC-BO (+C2) | **14.3** | 11.8 | 8.9 |
| **TRMC-BO (full)** | 13.4 | **12.2** | **10.9** |
| Range-Aware TB ×3 | 2.3~2.5 | 1.4~1.5 | 1.4 |

**창이 좁아질수록 C3가 중요해진다.** M4 대 M5 격차가 wide에서 −0.9(M4가 낫다),
medium +0.4, narrow **+2.0**으로 뒤집힌다. 좁은 창에서 in-band restart를 고르는 것이
실제로 값을 한다.

TRMC-BO full과 point-target의 격차도 창이 좁아질수록 벌어진다(+2.4 → +2.6 → **+3.8**).

### δ 선택이 결론을 바꾸지 않는다

| Method | d>0.05 | d>0.1 | d>0.2 | d>0.4 |
| --- | ---: | ---: | ---: | ---: |
| Point-target BO | 13.93 | 9.23 | 5.10 | 2.60 |
| TRMC-BO (+C2) | 17.03 | 11.67 | 6.53 | 3.00 |
| **TRMC-BO (full)** | **18.37** | **12.17** | **6.60** | **3.07** |
| Range-Aware TB (equal-vol.) | 2.17 | 1.80 | 1.50 | 1.37 |

TRMC-BO full이 네 δ 전부에서 1위다.

### 평균순위가 지표에 따라 완전히 뒤집힌다

| Method | rank (hit rate) | rank (diversity) |
| --- | ---: | ---: |
| Range-Aware TB (inscribed) | **1.5** | 6.0 |
| Range-Aware TB (equal-vol.) | **1.8** | 5.0 |
| Range-Aware TB (box) | 2.7 | 6.0 |
| Point-target BO | 4.0 | 3.0 |
| **TRMC-BO (full)** | 5.3 | **1.3** |
| TRMC-BO (+C2) | 5.7 | **1.7** |
| TRMC-BO (C1만) | 7.0 | 6.3 |

이 표가 논문 §10.2의 논증 그 자체다. 두 지표는 다른 질문에 답하며, 어느 것이 주
지표인지는 논문이 명시적으로 주장해야 한다.

### 분리된 valid 성분을 몇 개 짚었는가 (§5.2의 주장을 그 자체 용어로)

Branin의 valid 영역은 연결성분 4/3/3개로 갈라진다. seed당 도달한 성분 수의 평균:

| Method | wide (4개 중) | medium (3개 중) | narrow (3개 중) |
| --- | ---: | ---: | ---: |
| Random search | 2.70 | 1.57 | 1.00 |
| Standard BO | 2.44 | — (적중 0) | — (적중 0) |
| Point-target BO | 2.80 | 2.90 | 2.70 |
| TRMC-BO (C1만) | 2.60 | 1.70 | 1.11 |
| TRMC-BO (+C2) | 3.00 | 3.00 | 2.80 |
| **TRMC-BO (full)** | **3.00** | **3.00** | **3.00** |
| Range-Aware TB ×3 | 1.6~1.8 | 1.20 | 1.20~1.30 |

**TRMC-BO full은 모든 width에서 모든 seed가 주요 성분 3개 전부에 도달한다.**
Range-Aware TB는 1.2~1.8개 — 한 성분에 머문다. wide의 4번째 성분은 valid 면적의 0.2%
수준이라 3.00은 주요 3개를 전부 짚었다는 뜻이다.

**C1만 쓰면 medium/narrow에서 random search와 구분되지 않는다**(1.70/1.11 대 1.57/1.00).
§2의 포화 기전에 대한 독립적인 확인이다 — acquisition이 평평해진 뒤 M3의 탐색은 사실상
무작위다.

### 누적 적중 곡선

| Method | it 1 | it 10 | it 20 | it 30 | it 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Range-Aware TB (equal-vol.) | 0.70 | 9.10 | 19.07 | 29.03 | **49.03** |
| Point-target BO | 0.67 | 8.20 | 16.13 | 23.77 | 39.97 |
| TRMC-BO (full) | 0.63 | 4.13 | 10.17 | 17.87 | 35.33 |
| TRMC-BO (C1만) | 0.33 | 1.33 | 2.37 | 2.67 | **3.67** |
| Standard BO | 0.07 | 0.63 | 0.67 | 0.67 | 0.73 |

M3의 곡선이 iteration 20 부근에서 **완전히 평평해진다**(2.37 → 2.67 → 3.67). §2의 포화
시점과 일치한다.

## 4. Optimizer 건강 상태

전 method에서 acquisition 전면 실패율 0, random fallback률 0, feasible률 1.0,
중복 제안률 ≤ 0.010. 위 숫자를 optimizer 고장으로 설명할 수 없다.

## 5. 사전 등록한 예측의 결과

| 예측 | 결과 |
| --- | --- |
| **P1** hinge가 hit rate를 diversity와 교환 | **확인**, 예측보다 강하게 (13배 적중, distinct는 감소) |
| **P2** distinct에서 M3 < M4 < M5 단조 | **확인** (2.70 → 11.67 → 12.17) |
| **P3** point-target은 hit rate 승/distinct 패 | **확인** (39.97 적중 → 9.23, conc. 0.23) |
| **P4** extremum-seeking ≤ random | **확인** (0.73 대 2.00) |
| **P5** hinge 있는 쪽이 싸다 | **확인** (0.5~1.9 대 5.4~6.0 s/iter) |
| **P6** Branin 연결성분이 method를 가른다 | **확인** — TRMC-BO full은 3/3 성분에 전 width·전 seed 도달, TB는 1.2~1.8 |
| **caveat** M6/M6b/M6c는 이 suite에서 사실상 동일 | **확인** (distinct 1.70~1.80, 적중 48.9~49.1) |

예측을 먼저 커밋한 것이 실제로 값을 했다. §2의 포화 기전은 예측 목록에 없었고,
**P1의 방향은 맞았지만 그 크기와 원인은 예측보다 강하고 더 구조적이다.**

## 6. 뒷받침되는 것과 뒷받침되지 않는 것

**뒷받침되는 것**

- **서로 다른 설계 회수에서 TRMC-BO가 1위다.** 3 width × 4 δ = 12개 조건 전부에서.
  MatFormBench에서는 point-target에 hit rate로 밀렸는데(avg rank 1.47 대 2.67), 여기서는
  주 지표로 앞선다.
- **Extremum-seeking BO는 잘못된 도구다.** 두 벤치마크에서 random search보다 나쁘다.
- **좁은 창에서 C3가 값을 한다.** M4 대 M5 격차가 창 폭에 따라 뒤집힌다.
- **Range-Aware TB의 실패 양상이 선명하다.** 50회 중 49회를 창에 넣고 설계 1.7개를
  돌려준다. 규격 만족과 설계 다양성이 다른 목표라는 것의 가장 깨끗한 예다.
- **분리된 valid 영역에서 다양한 설계를 회수한다는 주장이 직접 확인된다.** 거리 대리
  지표가 아니라 성분 수로 측정했다: TRMC-BO full 3/3, TB 1.2~1.8.

**뒷받침되지 않는 것**

- **C1이 이 task에서 기여한다는 주장은 못 한다.** §2에서 acquisition이 완전히 평평해진다.
  단일 출력에서 TRMC-BO의 성능은 C2/C3의 것이다. **논문이 C1을 주 탐색 신호로
  서술한다면 m=1에서는 그 서술이 성립하지 않는다고 밝혀야 한다.**
- **Olympus는 input feasibility 난이도를 제공하지 않는다.** branin은 전부 feasible이고,
  pce10 hull은 simplex의 99.89%, oer_plate hull은 simplex 자체다. L5-4의 65% 제조불가와
  비교할 수 없다. sparse feasibility 주장은 MatFormBench 몫이다.
- **box→ball mapping 민감도는 이 suite에서 공허하다.** 세 mapping의 구간이 window 폭의
  99% 이상 겹친다(얇은 quantile 조각이라 valid centroid가 box 중점에 붙는다).
  MatFormBench에서는 겹치지 않았고 결과도 달랐다(1.73 대 1.37).
- **2차원 한 task의 결과다.** emulator 프로토콜(4-D, 6-D simplex, 실제 측정 기반) 540
  cell이 실행 중이며, 그것이 §5.3의 실질 증거다.

## 8. emulator 프로토콜 (§5.3) — 실제 mixture 3개

810 cell, 오류 0. `pce10`(4성분 고분자 블렌드 1,040점), `wf3`(같은 1,040 배합, 다른
고분자), `thin_film`(3성분 페로브스카이트 94점).

**task 선정에 실패가 두 번 있었고 둘 다 실행 전/후에 잡아 폐기했다.** Olympus가 simplex로
선언한 8개 중 쓸 수 있는 것은 3개뿐이다 — 자세한 경위는 §8.1, 전수 검사는
`eval/scripts/audit_olympus_datasets.py`.

### 8.1 Olympus의 mixture 데이터셋 전수 검사

선언은 **양방향으로** 믿을 수 없으므로, 43개 데이터셋에서 아핀 종속성(`Σx_S = const`,
즉 중심화 설계행렬 영공간의 지시벡터)을 찾았다. 8개가 constant-sum 구조를 갖고 3개가
쓸 수 있다. **다섯 개의 기각이 각각 다른 검사에만 걸린다:**

| 데이터셋 | 실패 유형 | 잡는 검사 |
| --- | --- | --- |
| `colors_bob` | simplex가 아님 (241행 전부 합≠1, 합 0.64~4.40) | constant-sum |
| `oer_plate` ×4 | simplex인데 6종 중 최대 4종만 혼합 (내부 표본 0%) | `interior_support` |
| `p3ht` | emulator가 자기 학습점 41.6%를 0으로 예측 | `activation_floor_rate` |

`colors_bob`은 270 cell을 돌린 뒤 폐기했다. 선언을 믿어 `sum(x)=1`을 부과했으므로 5-D
박스의 4-D 단면만 탐색했고, emulator는 합 0.64~4.40인 박스 전역으로 학습돼 있었다.
`hull_membership`이 simplex 가정으로 좌표를 떨어뜨리므로 그 "22.6% infeasible"도
인공물이었다. `p3ht`는 실행 전에 잡혔다 — window가 빈 구간 `[0,0]`으로 보정됐다.

선언 오류는 양방향 각 1건이다: `colors_bob`은 simplex라 선언하고 아니며, **`p3ht`는
`"none"`이라 선언하는데 5성분이 100±0.1로 합해지는 조성 백분율이다**(Olympus가 놓쳤다).
`tasks._measured_simplex_total`이 이제 세 검사를 모두 돌리고, 기각 사례가 테스트에
반례로 고정돼 있다.

### 8.2 주 결과 — 서로 다른 설계를 몇 개 찾았는가

3 task × 3 width × 10 seed 평균.

| Method | **distinct** | 적중 | conc. | 1st hit | viol | s/iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random search | 2.41 | 2.64 | 0.91 | 23.7 | 16.50 | 0.0 |
| Standard BO (extremum-seeking) | **1.24** | 1.26 | 0.99 | 32.6 | 15.85 | 2.6 |
| Point-target BO | 4.08 | 23.10 | 0.18 | 6.1 | 0.917 | 2.6 |
| TRMC-BO (C1만) | 2.60 | 3.20 | 0.81 | 19.7 | 15.79 | 1.3 |
| TRMC-BO (+C2) | 5.97 | 11.66 | 0.51 | 13.7 | 6.112 | **0.7** |
| **TRMC-BO (full)** | **6.46** | 13.89 | 0.46 | 10.6 | 2.357 | 1.7 |
| Range-Aware TB ×3 | **0.99~1.01** | **41.1~41.4** | **0.024** | 6.0~6.2 | 0.070 | 8.5~8.8 |

**Range-Aware TB는 50회 중 41회를 창 안에 넣고 서로 다른 설계 1개를 돌려준다**
(concentration 0.024). 비용은 TRMC-BO의 5~13배다. 이것이 §10.2 논증의 가장 깨끗한
형태다 — 규격 만족과 설계 다양성은 다른 목표다.

**Extremum-seeking BO는 random search보다 2배 나쁘다**(1.24 대 2.41). MatFormBench에서
8배 나빴던 것과 같은 방향이다.

### 8.3 task별·width별·δ별 — 전부 1위

| Method | pce10 | wf3 | thin_film |
| --- | ---: | ---: | ---: |
| Point-target BO | 4.20 | 4.43 | 3.60 |
| TRMC-BO (+C2) | 7.73 | 5.03 | 5.13 |
| **TRMC-BO (full)** | **8.07** | **5.83** | **5.47** |
| Range-Aware TB ×3 | 0.93~0.97 | 1.00~1.07 | 1.00~1.03 |

| Method | wide | medium | narrow | d>0.05 | d>0.1 | d>0.2 | d>0.4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Point-target BO | 6.80 | 3.70 | 1.73 | 6.19 | 4.08 | 2.80 | 1.92 |
| TRMC-BO (+C2) | 11.20 | 5.07 | 1.63 | 7.88 | 5.97 | 3.87 | 2.32 |
| **TRMC-BO (full)** | **11.43** | **5.40** | **2.53** | **8.50** | **6.46** | **4.18** | **2.34** |

**3 task × 3 width × 4 δ 전부에서 1위**이고 ablation 사다리도 전부에서 단조다
(2.60 → 5.97 → 6.46). 창이 좁아질수록 C3의 값이 커진다 — M4 대 M5 격차가
wide +0.23 → medium +0.33 → narrow **+0.90**이다.

### 8.4 평균순위가 지표에 따라 뒤집힌다

| Method | rank (hit rate) | rank (diversity) |
| --- | ---: | ---: |
| Range-Aware TB (inscribed) | **1.7** | 4.9 |
| Range-Aware TB (box) | 1.9 | 5.3 |
| Range-Aware TB (equal-vol.) | 2.4 | 6.1 |
| Point-target BO | 4.0 | 2.7 |
| **TRMC-BO (full)** | 5.0 | **1.8** |
| TRMC-BO (+C2) | 6.0 | 2.9 |

hit rate로는 TB가 1~2위이고 TRMC-BO가 5위, diversity로는 TRMC-BO가 1위이고 TB가
5~6위다. 두 지표는 다른 질문에 답하며, 어느 것이 주 지표인지는 논문이 명시적으로
주장해야 한다.

### 8.5 optimizer 건강 상태

`Standard BO`를 제외한 8개 method 전부 acquisition 전면 실패 0, random fallback 0,
중복 제안 0, feasible ≥ 0.998이다. 위 숫자를 optimizer 고장으로 설명할 수 없다.

`Standard BO`만 fallback 0.08, 중복 0.415, feasible 0.829다(pce10 0.792, wf3 0.698).
극값 추구가 simplex 꼭짓점으로 걸어나가고 그곳이 측정 hull이 끝나는 지점이다.

### 8.6 이 suite가 시험할 수 없는 것

**Olympus의 쓸 수 있는 mixture task 3개는 전부 input feasibility가 사실상 없다**
(feasible 0.998~1.000). 따라서 여기서 §5.3은 outcome-range 축만 시험한다.

앞선 보고에서 `colors_bob`을 근거로 "input feasibility가 병목이면 TRMC-BO가 진다"고
썼는데, 그 task가 무효였으므로 **철회한다.** 다만 그 관찰 자체는 MatFormBench 1,200
cell에서 독립적으로 성립한다:

| task | infeasible | Random | TRMC-BO full | Point-target | distinct (TRMC vs PT) |
| --- | ---: | ---: | ---: | ---: | --- |
| L3-1 | 24% | 0.764 | 0.867 | 0.889 | **8.33** vs 3.40 |
| L4-1 | 41% | 0.588 | **0.634** | 0.804 | **7.70** vs 3.63 |
| L5-4 | 69% | 0.314 | **0.313** | 0.329 | 1.07 vs **1.23** |

L4-1·L5-4에서 TRMC-BO의 feasible_rate가 random과 사실상 같다 — C1/C2/C3는 outcome
window는 겨냥하지만 **입력 실현가능성 축에 신호를 주지 않는다.** 24~41%에서는 다양성
우위가 압도하고, 69%에서 point-target에 근소하게 밀린다.

**근거는 MatFormBench L5-4 한 task뿐이며 Olympus로는 보강할 수 없다.** 논문에 넣으려면
그 한계를 밝히거나 HOIP에서 확인해야 한다. 이 관찰을 §2의 m=1 포화로 설명했던 것도
철회한다 — MatFormBench는 m=3 qEHVI 경로인데 같은 현상이 나타나므로 별개 사실이다.

## 부록 -- SCBO 추가 (2026-08-26)

Olympus의 모든 task는 m=1이라 COMBOO(다목적 전용)는 구조적으로 실행될 수 없다.
그 자리를 SCBO(Eriksson & Poloczek 2021, trust-region 기반 constrained BO,
`M7_scbo`)로 메웠다 -- COMBOO가 다목적 스위트(MatFormBench, HOIP)만 다루는 것과
정확히 상보적이다. 두 방향 window `[L,U]`는 COMBOO와 같은 방식으로 두 개의
블랙박스 부등식 제약으로 바꿨고, trust-region의 성공/실패 판정에 쓰는 목적함수는
`M2_target_distance`와 같은 "window 중점까지의 음의 거리"를 재사용했다.

| task | method | hit rate | n_unique |
| --- | --- | ---: | ---: |
| branin | M0_random | 0.047 | 2.3 |
| branin | M2_target_distance | 0.799 | 40.0 |
| branin | M5_full (TRMC-BO) | 0.707 | 35.3 |
| branin | **M7_scbo** | **0.797** | 39.9 |
| pce10 | M0_random | 0.063 | 3.2 |
| pce10 | M2_target_distance | 0.469 | 23.4 |
| pce10 | M5_full | 0.251 | 12.6 |
| pce10 | **M7_scbo** | **0.402** | 20.1 |
| thin_film | M0_random | 0.051 | 2.6 |
| thin_film | M2_target_distance | 0.569 | 28.5 |
| thin_film | M5_full | 0.401 | 20.0 |
| thin_film | **M7_scbo** | **0.487** | 24.3 |
| wf3 | M0_random | 0.044 | 2.2 |
| wf3 | M2_target_distance | 0.348 | 17.4 |
| wf3 | M5_full | 0.181 | 9.1 |
| wf3 | **M7_scbo** | **0.383** | 19.2 |

**SCBO는 네 task 전부에서 TRMC-BO의 M5_full을 앞선다**, 격차가 작지 않다
(branin +0.09, pce10 +0.15, thin_film +0.09, wf3 +0.20). `wf3`에서는
`M2_target_distance`까지 앞선다 -- 이 스위트 전체에서 TRMC-BO가 아닌 다른
method가 point-target baseline을 이기는 유일한 지점이다.

**해석.** m=1(단일 목적)에서는 TRMC-BO의 C1(range-probability qEI)이 이미
"클수록 좋다" 류의 단순한 global 획득함수로 축약된다는 점(§ 위 "At m=1 the
shipped C1 path is qEI(CDFRangeObjective)")과 맞물려, trust-region 기반
local search(SCBO)가 그보다 더 잘 맞는 문제라는 뜻으로 읽힌다. COMBOO가
다목적(m>=2)에서 겪는 "두 방향 제약을 동시에 만족하는 optimistic 점을 찾기
어렵다"는 어려움이, 단일 출력에서는 상대적으로 덜하다는 점도 일치한다 --
여기서는 output이 하나뿐이라 두 방향 제약을 "동시에" 만족해야 하는 다른
output이 없다.

**한계.** SCBO의 trust-region 후보 생성은 `pce10`/`wf3`/`thin_film`의 심플렉스
등식 제약을 이 하네스 자체의 Dirichlet 샘플러(`sample_raw_candidates`)로
우회했다 -- 논문이 공식적으로 제안한 방식이 아니라 이 프로젝트의 각색이다
(`scbo.py`의 "Adaptation -- simplex tasks" 절 참고). branin(순수 박스)에서는
튜토리얼의 원 방식을 그대로 썼다.


**버그 재검토 후 갱신 (2026-08-26).** 위 표는 최초 구현 기준이었다. 재검토에서
두 가지를 더 찾아 고쳤다: (1) n_init 전체를 trust-region의 성공/실패 판정
1회로 뭉뚱그려 셌던 것 -- 튜토리얼 원본은 초기 배치에 `update_state`를
아예 안 부른다. (2) restart 시 trust-region 길이/카운터만 리셋하고 중심점은
그대로였던 것 -- 실측 결과 캠페인당 restart가 평균 1회씩 발생해 드문 경우가
아니었고, 이 상태로는 "재시작"이 같은 자리서 다시 수렴하는 것과 다를 게
없었다. 두 버그 다 고친 뒤 재실행한 최종 수치:

| task | method | hit rate | n_unique |
| --- | --- | ---: | ---: |
| branin | M7_scbo | 0.787 | 39.4 |
| pce10 | M7_scbo | 0.383 | 19.2 |
| thin_film | M7_scbo | 0.485 | 24.3 |
| wf3 | M7_scbo | 0.371 | 18.6 |

수정 전 수치(branin 0.797, pce10 0.402, thin_film 0.487, wf3 0.383)와 거의
같다 -- 결론은 바뀌지 않는다: SCBO는 네 task 전부에서 여전히 TRMC-BO의
M5_full을 앞선다. 두 버그 모두 SCBO에게 불리한 방향이었으므로(초기 배치를
잘못 세면 성공/실패 판정이 왜곡되고, restart가 제 역할을 못하면 지역 최적에
갇히기 쉽다), 고친 뒤 수치가 거의 그대로라는 건 이 승리가 그 버그들에서
나온 인공물이 아니었다는 뜻이다.

## 7. 재현

```bash
# emulator 가중치 추출과 검증 (일회용 TF venv)
.venv-olympus/bin/python eval/scripts/extract_olympus_emulator.py
.venv-olympus/bin/python eval/scripts/verify_olympus_emulator.py

# 사양 상한과 window 동결 (둘 다 커밋되어 있음)
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.synthesize_olympus_targets
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.calibrate_ranges --suite olympus

# 실행 전에 측정한 valid 영역의 성질 (성분 수, δ-capacity)
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.valid_region_geometry --suite olympus

# 스윕과 표
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/olympus_branin.yaml --workers 22
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.report --suite olympus --protocol branin

# §2의 포화 기전
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plateau_fraction \
    --suite olympus --protocol branin --task branin --width medium --method M3_range_prob
```
