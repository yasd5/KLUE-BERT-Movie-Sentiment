# KLUE-BERT 기반 한국어 영화 리뷰 감정 분석

한국어 영화 리뷰를 입력하면 **긍정(1) / 부정(0)**을 분류하고 각각의 확률을 출력하는 프로젝트입니다. 사전학습 모델 `klue/bert-base`를 NSMC(네이버 영화 리뷰)로 Fine-tuning했습니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 모델 | [`klue/bert-base`](https://huggingface.co/klue/bert-base) |
| 데이터 | [NSMC](https://github.com/e9t/nsmc) |
| 환경 | Google Colab (GPU 권장) |
| 프레임워크 | PyTorch, Hugging Face Transformers |
| 평가 | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| 노트북 | `KLUE_BERT_movie.ipynb` |

## 실행 방법

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasd5/KLUE-BERT-Movie-Sentiment/blob/main/KLUE_BERT_movie.ipynb)

1. Google Colab에서 `KLUE_BERT_movie.ipynb`를 엽니다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
3. 필요한 패키지를 설치합니다(노트북 실행 환경에 맞춰 조정).

   ```python
   %pip install pandas==2.2.3 transformers==4.57.6 datasets==4.4.2 accelerate==1.12.0 huggingface-hub==0.36.2 scikit-learn matplotlib seaborn tqdm
   ```

4. 패키지 변경 후 필요한 경우 런타임을 재시작하고, 노트북 셀을 위에서 아래로 실행합니다. NSMC는 코드에서 자동으로 복제합니다.
5. 학습 후 `results/`의 그래프·지표·오답 분석 결과를 확인하고, 마지막 셀에서 새로운 리뷰를 입력해 예측합니다.

> 패키지 설치 셀은 실행 환경에 따라 조정해야 합니다. Google Colab의 기본 패키지와 충돌 경고가 발생할 수 있습니다.

## 데이터 및 학습 설정

| 설정 | 값 |
|---|---:|
| Train / Validation / Test | 20,000 / 3,000 / 5,000 |
| 최대 입력 길이 | 128 |
| Batch size | 16 |
| 최대 Epoch | 5 |
| Learning rate | 2e-5 |
| Early Stopping patience | 2 |
| 모델 선택 기준 | Validation F1-score |

결측치·빈 리뷰·중복 문장을 정리한 뒤 학습·검증 데이터를 분리합니다. 원본 NSMC 테스트 데이터는 학습과 겹치는 리뷰를 제외하고 별도로 사용합니다.

## 구현 내용

1. NSMC 데이터 로드 및 전처리
2. KLUE-BERT Tokenizer로 `input_ids`, `attention_mask` 생성
3. `AutoModelForSequenceClassification`을 이용한 이진 분류 Fine-tuning
4. 검증 F1 기반 최고 모델 저장 및 Early Stopping
5. 테스트 성능 측정, Loss/성능 곡선 및 Confusion Matrix 시각화
6. 새로운 영화 리뷰 입력 시 긍정·부정 확률 출력
7. 오답 추출 및 신뢰도 순 정렬

## 실험 결과

아래 수치는 이번 Colab 실행에서 확인한 **테스트 5,000개** 기준입니다.

| 지표 | 결과 |
|---|---:|
| Accuracy | **87.98%** |
| 정분류 | 4,399 |
| 오분류 | 601 (12.02%) |

### Confusion Matrix

| 실제 \ 예측 | 부정 | 긍정 |
|---|---:|---:|
| 부정 | 2,158 | 315 |
| 긍정 | 286 | 2,241 |

학습 Loss는 약 0.39에서 0.04로 감소했지만 검증 Loss는 약 0.32에서 0.74로 증가하여 과적합 경향이 관찰되었습니다. 검증 Accuracy/F1은 3 Epoch 부근에서 최고 수준을 기록했습니다. **Precision, Recall, F1의 정확한 수치는 노트북의 테스트 평가 출력 및 `results/test_metrics.csv`에서 확인할 수 있습니다.**

### 시각화 결과

노트북 실행 시 다음 파일이 생성됩니다.

- `results/training_curves.png`: Epoch별 Train/Validation Loss 및 Validation Accuracy/F1
- `results/confusion_matrix.png`: 테스트 데이터 혼동 행렬
- `results/training_history.csv`: Epoch별 기록
- `results/test_metrics.csv`: 테스트 지표
- `results/error_analysis.csv`: 오답 및 예측 확률

> `results/` 파일은 노트북 실행 후 생성되며, 이 저장소에는 학습 전 생성되지 않은 결과물을 임의로 포함하지 않았습니다.

## 새로운 문장 예측

```python
prediction('정말 재미있고 배우들의 연기도 좋았어요.')
prediction('시간이 아까울 정도로 지루했습니다.')
```

각 문장에 대해 예측 감정과 긍정/부정 Softmax 확률을 출력합니다.

## 한계 및 개선 방향

- **과적합:** 학습 Loss 감소와 달리 검증 Loss가 증가함 → 최적 Epoch 선택 및 Early Stopping 활용
- **반어법:** `완전내취향. 이게 왜 재미없을까?!`를 부정으로 오분류
- **복합 감정:** 긍정과 부정 평가가 함께 포함된 리뷰의 최종 감정 판단이 어려움
- **높은 확신의 오답:** 오답 중 예측 확률 99% 이상 사례 존재 → Calibration 검토
- **개선 방안:** 데이터 확대, 오답 유형별 보강, 학습률/정규화 조정

자세한 분석은 노트북 마지막의 **6. 모델의 문제점 및 개선 방법 분석**에 정리했습니다.

## 저장소 구조

```text
.
├── KLUE_BERT_movie.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── results/              # 실행 후 생성 (그래프/CSV)
```

`model/`의 체크포인트와 `data/`의 원본 데이터는 용량 및 재현성을 고려하여 Git에 포함하지 않습니다. 데이터는 [NSMC 원본 저장소](https://github.com/e9t/nsmc)를 참조합니다.
