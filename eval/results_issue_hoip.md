# HOIP 결과 — benchmark 3 of 3

**요약.** HOIP은 실제 무기물 조성탐색에서 **입력측 feasibility가 알려지지 않을 때**
TRMC-BO가 어떻게 되는지 보려고 설계했다. 3개 rung(dense/restricted/full,
input-infeasible 60.0%/73.5%/91.3%) × 12 methods × 10 seeds = 360 cell, 오류 0으로
완주했다.

**답을 얻지 못했다 — 답할 우위 자체가 없었다.** MatFormBench L5-4에서 본 "69% 근방에서
TRMC-BO의 distinct-design 우위가 역전된다"는 관측을 이 사다리로 검증하려 했는데,
가장 쉬운 rung(dense, 60% infeasible)에서부터 TRMC-BO full이 random search를
distinct designs에서 이기지 못한다(1.00 대 1.60). 역전을 측정하려면 먼저 우위가
있어야 하는데, 그게 없다.

**원인은 사전에 측정해뒀다 — 대리모델이 `m_star`를 배우지 못한다.** 어떤 커널·표현으로도
held-out R²가 −0.85~+0.10(평균 예측 수준). `bandgap`은 학습된다(R² 0.50~0.78). 그런데
목표창은 둘 다 만족해야 하고, bandgap 창을 통과하는 14개 중 7개가 `m*≥4`로 탈락한다.
그 결과 관측점이 사후확률 (0.99, 0.95)로 나머지 전부를 지배(dominate)해버리고,
qEHVI가 미관측 물질 100%에서 정확히 0이 된다.

**그리고 그게 C3를 자멸시킨다.** C3가 "posterior mean이 창 안"인 restart를 찾으려다
찾는 게 전부 이미 평가한 물질이라, 요청한 restart의 72~90%가 채워지지 않는다
(dense 27.7% → restricted 19.1% → full 10.4%만 in-band). 그러면 루프는 그 iteration을
random 설계로 때운다 — M5_full의 iteration당 random-fallback 비율이 dense 0.92,
restricted 0.72, full 0.77.

**더 나쁜 소식: 지금까지 "이겼다"고 보였던 것도 대부분 그 fallback 덕분이었다.**
random-fallback으로 채워진 evaluation을 빼고 **순수 acquisition 제안만**으로
distinct designs를 다시 세면, `M5_full`이 dense에서 찾은 valid design 1.00개는
**전부** fallback에서 나온 것이었다(non-fallback 기준 0.00). `M6`/`M6b`도 마찬가지다.
사전 등록한 P6가 우려했던 바로 그 상황이 실제로 일어났다 — 이 suite에서 "distinct
designs"를 method별로 비교하는 것 상당 부분이 acquisition이 아니라 **fallback 정책을
비교하는 것**이었다.

---

## 1. 무엇을 돌렸는가

- **Task**: Anubis(Aspuru-Guzik 그룹의 "unknown feasibility constraint BO" 논문)의
  페로브스카이트 응용. 명세는 논문 텍스트가 아니라 실제 실행 코드
  (`application_hoip/ei/*/run.py`)에서 읽어냈다:

  ```python
  match = df_results.loc[(molcat==...)&(metal==...)&(halogen==...)]
  if len(match) == 0:
      return np.nan, np.nan            # <- input feasibility가 알려지지 않음
  ...
  converged = measurement['bandgap'] < 0.5 and measurement['m_star'] < 4.
  ```

  설계공간은 유기 양이온 11종 × 금속 29종 × 할로겐 4종 = 1,276개 조합, 전부
  categorical. 오라클은 DFT 계산 테이블(Körbel/Marques/Botti) lookup이고
  **무잡음**이다. 테이블에 없는 조합이 input-infeasible(1,165개, **91.3%**)이다.
  목표창 `0.75≤Eg≤1.75` and `m*≤4`는 벤치마크가 직접 준 것이다 — MatFormBench는
  단측 threshold, Olympus는 우리가 구성한 것이었던 것과 다르다. 노이즈 없는
  비해석적(lookup) 오라클도 이 벤치마크가 처음이다.

- **입력 표현**: categorical 3개를 Anubis 자체가 쓴 물리화학적 descriptor로 인코딩
  (양이온 6 + 금속 4 + 할로겐 4 = 14차원 연속). 제안된 점은 block별로 가장 가까운
  실제 조성에 snap해서 평가하고, `Observation.X`는 snap된 좌표를 담는다 — GP가
  실재하는 물질로만 학습되고, 중복 제안이 중복으로 잡힌다.

- **사전 검사** (`scripts/audit_hoip_catalogue.py`, 설계공간이 유한해 전부 정확한 수):
  파싱된 옵션이 전부 선언된 목록 안에 있는가, 조성 하나당 행 하나인가, `molcat+metal+halogen`
  concatenation이 단사(injective)인가, si_table의 조성이 파싱을 통해 되돌아오는가,
  descriptor가 완비되고 유한한가 — **전부 통과**. 파싱 검사가 특히 중요한 이유는
  업스트림이 문자열 부분매칭으로 조성을 복원하는데, `S`가 금속이면서 동시에 `H3S`/`MS`의
  꼬리이기도 해서 잘못 라벨링될 수 있는 구조이기 때문이다(Olympus에서 convex hull
  검사가 `colors_bob`/`oer_plate` 두 실패를 못 잡았던 교훈).

  두 가지는 논문에 반드시 적어야 한다. (1) infeasible 라벨이 두 가지를 섞는다 —
  "안정한 페로브스카이트가 아님"과 "안정하지만 목표 물성(m\*)이 정의되지 않는 금속성
  물질"이 같은 라벨이다(8개, 전부 PBE gap 0.00). hit에는 영향 없다. (2) `m_star`의
  `>1000`은 censoring sentinel이지 측정값이 아니다(feasible 물질 2개, NaN으로 처리).

- **feasibility 사다리** (`scripts/freeze_hoip_catalogues.py`, 스윕 전 동결):

  | task | 물질 수 | infeasible | valid | δ-capacity | 캠페인 커버리지 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `dense` | 240 | 60.0% | 7 | 7 | 33.3% |
  | `restricted` | 408 | 73.5% | 7 | 7 | 19.6% |
  | `full` | 1,276 | 91.3% | 7 | 7 | 6.3% |

  세 rung은 다른 문제가 아니라 **같은 설계공간의 중첩 제한**이다. 후보 사각형은
  전부 valid 물질 7개를 포함해야 한다는 조건을 걸어서, valid 집합이 세 rung에서
  완전히 동일하고 δ=0.1 capacity도 정확히 7로 고정된다. 오직 "만들 수 있는 조성의
  비율"만 60.0%→73.5%→91.3%로 움직여서, L5-4의 69% 근방을 한 화학계 안에서
  양쪽으로 감싼다. 14차원에서 δ=0.1은 7개 valid 물질을 전부 분리하므로, 이 suite에서
  "distinct qualifying designs"는 문자 그대로 "distinct usable materials"다.

- **Protocol**: `n_init` 30, `budget` 50, `q` 1, `num_restarts` 128(다른 두 suite와
  동일 — 32로 줄이면 point-target baseline이 붕괴한다는 측정된 이유로 유지), 12
  methods × 3 tasks × 10 seeds = 360 cell, 오류 0.

---

## 2. 본론 — 대리모델이 `m_star`를 배우지 못하고, 그게 C3를 자멸시킨다

스윕 **전에** 측정해서 `predictions_hoip.md`에 동결해 둔 진단이고, 스윕 결과가 정확히
이 기전대로 나왔다.

**절반만 학습 가능한 문제.** 109개 완전 labeled 물질에서 held-out R²
(`n_train`=80, 3회 반복):

| representation | kernel | bandgap | `m_star` | log `m_star` |
| --- | --- | ---: | ---: | ---: |
| descriptor(14) | RBF *(production 그대로)* | 0.502 | −0.016 | −0.081 |
| descriptor(14) | RBF-ARD | 0.776 | −0.847 | −0.136 |
| descriptor(14) | Matérn-ARD | 0.680 | −1.330 | −0.068 |
| one-hot(44) | RBF | 0.822 | −0.025 | 0.106 |
| one-hot(44) | RBF-ARD | 0.753 | 0.098 | −0.021 |
| one-hot(44) | Matérn-ARD | 0.814 | −0.193 | 0.121 |

학습 평균을 예측하면 R²=0이다. bandgap은 두 표현 모두에서 학습되지만, **effective
mass는 어떤 커널·표현·로그변환으로도 학습되지 않는다.** 그리고 이게 실제로 걸린다 —
bandgap 창을 통과하는 14개 feasible 물질 중 **7개가 `m*≥4`로 탈락**한다. 즉 HOIP은
절반만 학습 가능한 문제이고, 학습 불가능한 절반이 학습 가능한 절반을 통과한 후보의
절반을 다시 걸러낸다.

**acquisition에 미치는 영향.** `dense`에서 50개 관측 시점(`scripts/acquisition_plateau.py`):

- 대리모델이 학습점 밖에서 사전분포로 되돌아간다 — posterior sd가 관측점 0.045,
  미관측점 1.023(사전분포 수준);
- 관측된 in-window 물질은 range probability **(0.99, 0.95)**를 받는 반면 미관측
  물질은 **(0.19, 0.09)**를 못 넘는다;
- **미관측 물질 193개 전부가 관측된 하나에 지배(dominate)**되고, qEHVI가 미관측
  물질 100%에서 **정확히 0**이다;
- objective가 x에 대해 결정론적이라(사후표본 8개 간 spread 2×10⁻¹⁶), qEHVI가
  평균낼 것이 없다;
- C3의 in-band 필터가 240개 중 **2개만** 통과시키는데, **둘 다 이미 관측된
  물질**이다.

이건 Olympus의 saturation과 **다른 기전**이고 구분해야 한다. Olympus는 m=1에서
hinge에 의한 소멸(`qEI`가 incumbent 아래를 전부 정확히 0에 묶음)이다. HOIP은 m=2라
그 경로를 안 타는데도, **대리모델이 약해서** 관측점이 확률공간에서 (1,1) 근방까지
치솟아 나머지 전부를 지배해버리는 별개의 경로로 같은 결과(평평한 acquisition)에
도달한다. "saturation은 단일 출력 현상"이라는 기존 서술은 그대로 두고, 이건 두 번째,
별개의 평평한-acquisition 경로로 기록한다.

**계획서의 대응책(PCA로 차원 축소)은 적용되지 않는다.** 차원이 문제가 아니라
`m_star` 자체가 신호를 안 담고 있어서다. ARD나 one-hot으로 바꾸면 bandgap 적합만
올라가고 `m_star`는 그대로다. 그리고 non-ARD RBF는 production이 실제로 배포하는
surrogate라, 이 ablation이 진술해야 하는 대상 그 자체다.

**스윕 전체에서 확인.** `dense`에서 M5_full이 요청한 restart 243,200개 중 in-band는
67,483개(27.7%), `restricted`는 19.1%, `full`은 10.4%로 — 단일 시드 probe가 아니라
전체 3,600개 loop iteration 걸쳐 확인된다.

---

## 3. 주 결과 — 우위가 있어야 역전을 잴 수 있는데, 우위가 없다

전체 360 cell 집계(`eval.scripts.report --suite hoip --protocol catalogue`).

### task별 distinct designs

| method | dense (60%) | restricted (73.5%) | full (91.3%) |
| --- | ---: | ---: | ---: |
| Random search | 1.60 | 0.50 | 0.80 |
| Standard BO (extremum) | 1.20 | 1.40 | 0.20 |
| Point-target BO | 1.10 | 1.20 | 0.00 |
| TRMC-BO (C1) | 1.70 | 0.60 | 0.40 |
| TRMC-BO (C1+C2) | 1.60 | 0.80 | 0.40 |
| **TRMC-BO (full)** | **1.00** | 1.30 | 0.60 |
| TB (equal-volume) | 1.00 | 1.40 | 0.20 |
| TB (inscribed) | 1.00 | 1.40 | 0.00 |

**세 rung을 관통하는 단조 패턴이 없다.** M5_full은 dense에서 진다(1.00 대 1.60),
restricted에서 이긴다(1.30 대 0.50), full에서 다시 진다(0.60 대 0.80). random을
이긴 건 세 rung 중 restricted 하나뿐이고, 그마저 아래 §3.2에서 non-fallback
기준으로 다시 지는 것으로 드러난다. "infeasibility가 커질수록 우위가 단조롭게
준다"는 이야기가 아니다.

### random-fallback을 제외하면: "이겼다"는 결과가 대부분 사라진다

`predictions_hoip.md`의 P6가 미리 지적한 위험 — fallback 비율이 method마다 다르므로,
distinct 비교가 acquisition이 아니라 fallback 정책을 재는 것일 수 있다 — 을 그대로
검증했다. `trace.parquet`의 `cnt_random_fallback`으로 순수 acquisition 제안(non-fallback)만
남기고 distinct를 다시 셌다.

| method | dense: all → non-fb | restricted: all → non-fb | full: all → non-fb |
| --- | ---: | ---: | ---: |
| Random search | 1.60 → 1.60 | 0.50 → 0.50 | 0.80 → 0.80 |
| Point-target BO | 1.10 → 0.40 | 1.20 → 0.30 | 0.00 → 0.00 |
| TRMC-BO (C1) | 1.70 → 1.20 | 0.60 → 0.30 | 0.40 → 0.40 |
| **TRMC-BO (full)** | **1.00 → 0.00** | 1.30 → 0.30 | 0.60 → 0.60 |
| TB (equal-volume) | 1.00 → 0.00 | 1.40 → 0.00 | 0.20 → 0.20 |
| TB (inscribed) | 1.00 → 0.00 | 1.40 → 0.00 | 0.00 → 0.00 |

Random search는 정의상 all == non-fb(자체 fallback률 0.2% 미만이라 거의 개입이
없다). 나머지 전부 큰 폭으로 줄어들고, **`dense`에서 M5_full·M6·M6b는 non-fallback
기준 정확히 0.00** — 이 세 method가 dense에서 "찾았다"고 표에 나온 valid design은
**전부 loop의 random-fallback 안전장치에서 나온 것이지, acquisition이 제안한 게
아니다.** `restricted`의 M6/M6b도 마찬가지(1.40 → 0.00).

**이게 §3.1의 표를 다시 읽게 만든다.** M5_full이 restricted에서 random을 앞선
것처럼 보인 1.30 대 0.50도, non-fallback 기준으로는 0.30 대 0.50으로 **다시 진다.**
Random search가 fallback 없이 스스로 낸 결과가 model-based method들의 진짜
acquisition 기여보다 나은 경우가 대부분이다.

### 평균순위 — hit rate 기준 point-target이 12개 중 12위

| method | hit_rate 평균순위 | diversity_norm 평균순위 |
| --- | ---: | ---: |
| TRMC-BO (full, product in qEI) | 4.67 | 6.17 |
| TRMC-BO (+ C2) | 4.83 | 5.83 |
| Random search | 5.50 | 5.33 |
| TRMC-BO (full) | 6.83 | 6.83 |
| **Point-target BO** | **9.50** | 8.83 |

MatFormBench·Olympus에서 point-target은 hit rate로 1~2위를 지키던 baseline이었다.
HOIP에서는 **최하위**다(9.5/12). feasible_rate도 다른 method들과 사실상 구분되지
않는다(랜덤 대비 −0.024~+0.058, method 전체가 이 좁은 띠 안에 있다) — MatFormBench
L5-4에서 point-target이 feasibility에서 뚜렷이 앞섰던 패턴이 여기서는 재현되지 않는다.

### optimizer 건강 상태

| method | random_fallback | duplicate_proposal(/4) | acqf_total_failure |
| --- | ---: | ---: | ---: |
| Random search | 0.001 | 0.15 | 0 |
| TRMC-BO (full) | 0.80 | 3.31 | 0 |
| TB (equal-volume) | 0.91 | 3.73 | 0 |
| Point-target BO | 0.77 | 3.26 | 0 |

`acqf_total_failure`는 전 method 0 — acquisition 자체가 예외로 죽은 건 아니다.
문제는 실패가 아니라 **매 iteration 재시도 4번이 전부 이미 평가한 물질로 수렴**하는
것이다(duplicate_proposal이 최대치 4에 근접). Random search만 0.15로 낮다.

---

## 4. 사전 등록한 예측(`predictions_hoip.md`)과의 대조

| # | 예측 | 판정 |
| --- | --- | --- |
| P1 | TRMC-BO의 feasible_rate가 세 rung 모두 random 이하 | **반증** — M5_full의 delta는 세 rung 모두 +0.012~+0.030으로 random보다 항상 살짝 위다. 다만 재해석하면: method 전체(M1~M6b)의 delta가 −0.024~+0.058 좁은 띠 안에 몰려 있어, TRMC-BO가 다른 method 대비 유의하게 낮지도 않다. blindness는 TRMC-BO만의 것이 아니라 **전-method 현상**이다 |
| P2 | distinct 우위가 단조 감소하다 역전 | **불능 판정** — §3.2가 보여주듯 애초에 잴 우위가 없다(대부분 fallback 기여). 사다리로 크로싱을 짚을 수 없었다 |
| P3(수정판) | M3가 세 rung 모두 random 수준 | **확인** — dense 1.70/1.60, restricted 0.60/0.50, full 0.40/0.80로 표준편차 안에서 구분 불가 |
| P4 | M4가 M3보다 낫지 않음, `M0≈M3≈M4` 이고 M5/M6가 그 아래 | **부분 확인** — dense에서 그 순서대로다(M3 1.70 ≈ M0 1.60 ≈ M4 1.60, 셋 다 M5 1.00보다 위). restricted/full은 fallback 잡음 안에서 뒤섞임 |
| P5 | product 변형이 hit rate는 올리고 distinct는 낮춤(MatFormBench 재현) | **불명확** — `M5q_full_product_ei`가 오히려 전체 평균 distinct 1위(1.067). MatFormBench의 §4.2 패턴이 이 suite에서 재현되지 않음 |
| P6 | fallback이 캠페인 커버리지를 따라가고 method 간 비교를 오염시킬 수 있다 | **확인, 최악의 형태로** — dense/restricted에서 M5·M6·M6b의 "승리"가 non-fallback 기준 0으로 사라진다. 사전 등록에서 대비해 둔 바로 그 실패 |

---

## 5. 결론 — 이 suite가 답할 수 있는 것과 없는 것

**답하지 못한 것.** MatFormBench L5-4(69% infeasible)에서 본 distinct-design 역전을
다른 화학계에서 재현/반증하는 것이 이 suite의 원래 목적이었다. 가장 쉬운 rung(60%)
에서부터 TRMC-BO에 우위가 없어서, 잴 대상 자체가 없었다. 그러니 "69%가 진짜 경계인가"
라는 질문은 HOIP으로는 답할 수 없고, L5-4 하나의 관측으로 남는다.

**대신 측정된 것 — 더 근본적인 실패 모드.** 목표 물성 중 하나가 조성만으로 예측
불가능할 때(`m_star`, R² ≈ 0), C1은 대리모델이 관측점만 지배하는 acquisition을
만들어 랜덤과 구분 안 되고, C3는 "범위 내 restart"를 찾다가 이미 평가한 점만 찾아서
오히려 랜덤보다 나빠진다. 이건 Olympus의 m=1 saturation과 별개의, m≥2에서 나타나는
새로운 실패 경로다.

**그리고 하나 더, 방법론적으로 중요한 발견.** random-fallback을 제외하고 다시 세면
TRMC-BO(및 TB)가 "이겼다"고 보였던 결과 다수가 사라진다. 이건 HOIP만의 문제가
아니라, **fallback이 비영(non-zero)인 어떤 결과든 재검토가 필요하다**는 일반적
경고다. `random_fallback_rate`가 무시할 수 없는 수준일 때 distinct-designs
비교표는 acquisition 품질이 아니라 fallback 정책을 재고 있을 수 있다.

**feasible_rate 결과는 유효하게 남는다.** 대리모델이 작동하든 안 하든 이 지표는
오라클이 실제로 관측한 조성의 비율이라, §4의 P1 판정처럼 methods 전체가 좁은 띠
안에 있다는 결론 자체는 흔들리지 않는다 — 다만 TRMC-BO만 따로 낮은 게 아니라
**모든 method가** manufacturability 축에 대해 랜덤 수준이라는 더 강한 결론이다.

**논문에 적어야 할 문장.** "TRMC-BO의 다양성 우위는 목표 물성이 조성으로부터
예측 가능할 때 성립하며, 예측 불가능한 물성이 목표창에 포함되면 C1/C2/C3 전체가
random search 수준으로 붕괴한다." — 이건 적용 범위의 한계이지 방법의 결함이
아니고, HOIP이 이 경계를 실측으로 제공한다.

---

## 부록 -- COMBOO / Anubis 추가 (2026-08-26)

MatFormBench에서 매핑 버그를 고친 COMBOO(`M8_comboo`)와, 이 스위트의 출처 논문
자체를 baseline으로 이식한 Anubis(`M9_anubis`, feasibility classifier로
획득함수를 가중)를 12개 기존 method에 추가했다. HOIP은 등식/심플렉스 제약이
없는 14차원 박스라 COMBOO의 feasibility probe가 MatFormBench의 심플렉스
task에서처럼 비싸지지 않는다.

| task | method | hit rate | n_unique | feasible_rate |
| --- | --- | ---: | ---: | ---: |
| dense (60% infeasible) | M0_random | 0.032 | 1.6 | 0.388 |
| dense | M5_full (TRMC-BO) | 0.022 | 1.0 | 0.418 |
| dense | **M8_comboo** | **0.040** | 1.9 | 0.466 |
| dense | M9_anubis | 0.034 | 1.7 | 0.320 |
| restricted (73.5%) | M0_random | 0.010 | 0.5 | 0.248 |
| restricted | M5_full | 0.026 | 1.3 | 0.272 |
| restricted | **M8_comboo** | **0.036** | 1.8 | 0.344 |
| restricted | M9_anubis | 0.024 | 1.2 | 0.174 |
| full (91.3%) | **M0_random** | **0.016** | 0.8 | 0.106 |
| full | M5_full | 0.012 | 0.6 | 0.118 |
| full | M8_comboo | 0.006 | 0.3 | 0.124 |
| full | M9_anubis | 0.006 | 0.3 | 0.080 |

**COMBOO는 여기서 이긴다 -- MatFormBench와 정반대다.** `dense`/`restricted`
양쪽에서 M8_comboo가 M0_random과 TRMC-BO의 M5_full을 모두 앞선다. 이건
`results_issue_matformbench.md`가 기록한 "COMBOO는 좁은 두-방향 window에서
거의 항상 infeasibility를 선언한다"는 메커니즘이 여기서는 작동하지 않는다는
뜻이다 -- 실제로 HOIP의 window(`0.75<=Eg<=1.75, m*<=4`)는 benchmark 자체가
준 것이라 MatFormBench의 quantile-보정 window들보다 훨씬 넓고, m*의 하한 0은
물리적 하한이라 사실상 한쪽만 타이트한 제약에 가깝다 -- COMBOO의 probe가 걸려
넘어질 좁은 두-방향 조건 자체가 약하다. `full`(가장 infeasible한 조건)에서만
COMBOO가 random에 진다.

**이 스위트 자체의 negative finding(§ 위 "HOIP's headline result is negative")과
합쳐 읽으면**: TRMC-BO가 HOIP에서 random을 못 이기는 건 "이 문제가 모든
constrained BO에게 어렵기 때문"이 아니다 -- COMBOO는 두 rung에서 TRMC-BO보다
낫다. 즉 원인은 이 스위트가 아니라 TRMC-BO의 특정 메커니즘(m*가 학습 불가능해
C3의 in-band 필터가 굶는 것, § 위 참고)에 있다는 진단이 한 번 더 확인된다.

**Anubis는 두드러진 이점을 보여주지 않는다.** `dense`에서는 TRMC-BO와
random을 근소하게 앞서지만(0.034), `restricted`/`full`에서는 TRMC-BO보다
낮다. Naive product 결합(`P(feasible|x) * base_acquisition(x)`)만 구현했다는
점을 고려하면, 이건 "unknown-feasibility 인식이 이 문제에서 별 도움이 안
된다"보다는 "naive 결합 방식으로는 부족하다"는 쪽에 더 가까운 해석이다 --
원 논문이 벤치마크한 FIA/FCA 같은 정교한 결합 규칙은 구현하지 않았다는 한계를
명시해야 한다.

**버그 재검토 후 갱신 (2026-08-26).** 위 수치는 재검토에서 찾은 버그를
고친 뒤의 최종값이다: feasible 관측이 0개인 이른 구간에서 모든 catalogue
후보의 점수가 상수(bootstrap P(feasible)=1.0)가 되어 `argsort`가 매번 같은
후보를 반환하는 문제가 있었다 -- HOIP은 결정론적 lookup이라 한번
infeasible로 확인된 배합을 재제안해도 아무것도 배우지 못하고, 예산이 남아
있는데도 같은 배합에 갇혀버렸다. 이미 관측한 catalogue 행을 후보에서
제외하도록 고친 뒤(`_unexplored_mask`) 30 cells를 재실행했다. `feasible_rate`가
전반적으로 낮아졌는데(dense 0.408→0.320, restricted 0.266→0.174,
full 0.120→0.080) -- 갇혀서 몇 개 후보 주변만 맴돌던 것보다, 실제로 새
catalogue 행을 계속 탐색하는 게 HOIP의 기본 infeasibility 비율(60~91%)에
더 가깝게 나오는 것은 이치에 맞는다.

## 6. 재현

```bash
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.fetch_hoip_data
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.audit_hoip_catalogue
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.freeze_hoip_catalogues
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.freeze_hoip_specs
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.valid_region_geometry --suite hoip
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.verify_variants --suite hoip --task dense
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.acquisition_plateau --suite hoip --task dense

PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/hoip_catalogue.yaml --workers 22

PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.report --suite hoip --protocol catalogue
PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests/test_hoip.py -q
```

예측은 `predictions_hoip.md`, 하니스 설계는 `eval/README.md`의 HOIP 절 참고.

## 부록 -- C4 (feasibility weighting) 결과 (2026-08-27)

90 cell, 오류 0. 예측은 스윕 전에 `eval/predictions_c4.md`로 동결했다(커밋 `cc1a3f0`).
**세 예측 중 둘이 반증됐고, 결론은 바뀌지 않았다.**

| task | method | feasible_rate | fallback | dup /4 | distinct | non-fb distinct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense | M0_random | 0.388 | 0.002 | 0.25 | 1.60 | **1.60** |
| dense | M5_full | 0.418 | 0.918 | 3.72 | 1.00 | 0.00 |
| dense | M10 (C4a) | 0.422 | **0.988** | **3.97** | 1.30 | 0.10 |
| dense | M11 (C4b) | 0.420 | 0.906 | 3.68 | 1.10 | 0.10 |
| dense | M12 (C4a+b) | 0.422 | 0.988 | 3.96 | 1.30 | 0.10 |
| restricted | M0_random | 0.248 | 0.000 | 0.13 | 0.50 | 0.50 |
| restricted | M5_full | 0.272 | 0.722 | 3.04 | 1.30 | 0.30 |
| restricted | M10 (C4a) | 0.276 | **0.888** | **3.63** | 1.50 | 0.50 |
| restricted | M11 (C4b) | 0.264 | **0.680** | 2.91 | 1.40 | 0.40 |
| full | M0_random | 0.106 | 0.000 | 0.06 | 0.80 | **0.80** |
| full | M5_full | 0.118 | 0.770 | 3.17 | 0.60 | 0.60 |
| full | M10 (C4a) | 0.138 | 0.784 | 3.24 | 0.70 | 0.70 |
| full | M11 (C4b) | 0.128 | **0.716** | 2.99 | 0.60 | 0.60 |

### 예측 대조

**C1p(C4a가 세 rung 전부에서 feasible_rate를 올린다) -- 반증.** 부호는 세 rung 모두
맞지만(+0.004 / +0.004 / +0.020) 등록한 기각 기준이 "세 rung 전부 ±0.02 이내면 기각"
이었고, 세 값 모두 그 안에 있다. **그리고 등록해 둔 교란 규칙이 그대로 물렸다**:
C4a의 duplicate가 같이 올라갔고(3.72→3.97, 3.04→3.63) fallback도 올라갔다
(0.918→0.988, 0.722→0.888). 규칙대로 이것은 개선이 아니라 **효과 없음**으로 보고한다.

기전은 예측 파일이 미리 짚어 둔 대로다. HOIP의 평평한 acquisition에서
`feasibility_weighted_scores`의 flat guard가 발동해 순위가 **`P(feasible|x)` 단독**이
되고, 그 값이 가장 높은 곳은 **이미 관측한 feasible 물질 근처**다. 그래서 C4a는
manufacturability를 찾는 게 아니라 아는 곳을 다시 두드린다.

**C2p(C4b가 restart 고갈을 악화시킨다) -- 반증, 방향이 반대다.** M11의 fallback이
세 rung **전부에서 M5_full보다 낮다**(0.906 대 0.918, 0.680 대 0.722, 0.716 대 0.770).
in-band 생존율도 0.187 대 0.194로 예측대로 조금 낮은데, fallback은 오히려 줄었다.
이유는 duplicate다 -- C4b가 restart 분포를 바꾸면서 중복 제안이 줄었고(2.91~3.68 대
3.04~3.72), 이 루프에서 fallback을 부르는 것은 고갈이 아니라 **중복**이기 때문이다.
"필터를 하나 더 AND로 걸면 굶는다"는 추론이 틀린 게 아니라, 굶는 것과 fallback을
부르는 것이 같은 사건이 아니었다.

**C3p(HOIP의 negative 결론이 유지된다) -- 확인.** all 기준으로 세 변형 모두 3 rung 중
`restricted` 하나에서만 random을 넘고, **non-fallback 기준으로는 세 rung 어디서도
random을 넘지 못한다**(0.10/0.50/0.70 대 1.60/0.50/0.80). M5_full의 0.30에서
0.43으로 오르긴 하지만 random의 0.97에는 한참 못 미친다.

**C6p(고치는 것은 classifier가 아니라 restart pool이다) -- 지지된다.** 같은 classifier를
쓰는 `M9_anubis`는 여전히 fallback 0.000, duplicate 0.00이고 non-fallback distinct
1.07로 전 method 1위다. 차이는 분류기가 아니라 **열거된 catalogue에서 restart를
뽑는다**는 것뿐이다. C4는 분류기만 가져왔고 pool은 가져오지 않았다.

### 논문 처리

등록해 둔 대응표의 **H1**("C4 효과 없음")에 해당한다. §Method에 한 문단, 부록에 표.
HOIP 절의 결론은 손대지 않는다. 다만 H1의 순수한 형태는 아니다 -- C4는 아무 일도 안
한 게 아니라 **측정 가능한 일을 했고 그것이 도움이 안 됐다**. 그 기전(flat guard →
feasibility 단독 순위 → 기존 관측 재방문)은 적을 값이 있다.

### 다음 (별도 결정 필요)

C6p가 지지되므로 **pool 기반 restart fallback(C5)** 이 실제 해법일 가능성이 남는다:
C3의 fallback을 균등 random이 아니라 `P(feasible|x)` 상위 catalogue 행에서 뽑는 것.
새 구성요소이므로 논문 범위가 넓어진다.
