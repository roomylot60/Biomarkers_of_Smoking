<!-- 기간 : 22.9.20~10.7 -->
# 9/20 ~ 22일 회의록
## 9/21 아이디어 회의
- 성기호
    - [심장병 예측](https://www.kaggle.com/code/pantanjali/analysis-of-heart-attack-with-machine-learning/notebook)
    - 담배 예측
    - [뇌졸증 예측](https://www.kaggle.com/code/adhefirmansyah/brainstroke-classifier-predict/data)
    - [뇌종양 예측](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
    - [고객 성격 분류](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- 이중훈
    - [차량 가격 예측](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/overview)
    - 의료보험 비용 예측
    - [의류 거래량 예측](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview)
- 박준수
    - [우크라이나 - 러시아 전쟁에서 러시아인의 장비손실 및 사망자수 예측](https://www.kaggle.com/datasets/piterfm/2022-ukraine-russian-war)
    - [전세계의 행복도 분류](https://www.kaggle.com/datasets/unsdsn/world-happiness):gdp, 기대수명, 정부에 대한 신뢰도 등을 바탕으로 행복도가 높은 순위의 국가를 분류
    - [역사상 가장 영향력 있는 50인의 그림 컬랙션](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time):인공신경망을 통해 아티스트를 식별
- 민홍기
    - [뇌졸증 예측](https://www.kaggle.com/code/adhefirmansyah/brainstroke-classifier-predict/data)
    - [뇌종양 예측](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
    - [고객 성격 분류](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- 정서연
    - [자동차 가격 예측](https://www.kaggle.com/code/goyalshalini93/car-price-prediction-linear-regression-rfe)
    - [의료보험비-선형회귀](https://www.kaggle.com/code/mariapushkareva/medical-insurance-cost-with-linear-regression)
    - [고객거래 예측](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview)

## 9/22 2차 아이디어 회의(추가 아이디어)
- 성기호
    - [날씨 예측](https://www.kaggle.com/datasets/ananthr1/weather-prediction)
- 박준수
    - [흡연에 대한 신체 반응](https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking)
    - [노트북 가격 예측](https://www.kaggle.com/datasets/muhammetvarl/laptop-price?select=laptop_price.csv)
    - [웨이트 트레이닝 무게 예측 ](https://www.kaggle.com/datasets/kukuroo3/powerlifting-benchpress-weight-predict)
- 민홍기
    - [파산 감지 모델 (1999~2009 대만)](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)
    - [구인 공고 사기 여부 예측](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction?datasetId=533871&sortBy=voteCount)
- 이중훈
    - [항공승객 만족도 예측](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
    - [고객의 신용카드 해지여부 예측](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- 정서연
    - [폭스바겐 가격 결정요인 회귀분석](https://www.kaggle.com/code/gireeshs/volkswagen-price-regression-r-2-0-9555) : 데이터 특성 중 최적의 특성 선정

## 주제 기획 마무리 및 기획서 제출
**주제 : 흡연에 대한 신체 반응**
- [데이터 준비](https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking)
    - 데이터 탐색(EDA) 시각화
    - 시각화 보고서
- 도메인 정의
    - 피처 도메인 정의서 작성
---
# 9/28 데이터 전처리
## EDA
- 일반 특성
    |index|Feature|Data type|Description|
    |--|----|---|------|
    |0|gender|object|성별 : F/M|
    |1|age|int64|연령|
    |2|height(cm)|int64|신장|
    |3|weight(kg)|int64|체중|
    |4|waist(cm)|float64|허리둘레|
    |5|eyesight(left)|float64|시력(왼)|
    |6|eyesight(right)|float64|시력(오)|
    |7|hearing(left)|float64|청력(왼)|
    |8|hearing(right)|float64|청력(오)|

- 혈관 및 호흡
    |index|Feature|Data type|Description|
    |--|----|---|------|
    |9|systolic|float64|수축기압 : 심장이 수축되어 혈액이 대동맥으로 압출되면 동맥 내로 보내지는 심실 수축기의 혈압|
    |10|relaxation|float64|이완|
    |11|fasting blood sugar|float64|공복혈당 : 식사를 한 후 8시간 이후에 측정한 혈당 농도. 공복 혈당이 높아지면 여럴 성인병으로 이어질 수 있으며, 정상적인 공복 혈당 수치는 100mg/dL 미만.100~125mg/dL를 혈당 장애, 126이상에 당화혈색소가 6.5 이상일 경우 당뇨병으로 분류.|
    |12|Cholesterol|float64|콜레스테롤 : 전신에 존재하는 모든 세포의 막을 형성하는 지질의 한 종류로 생명에 필수적인 물질로,  체내의 막 표면에 있으면서 막을 보호하고, 혈관벽이 찢어지는 것을 예방하며 적혈구의 수명을 오래 보전한다.<br>콜레스테롤이 부족한 경우 적혈구의 수명이 짧아져 빈혈이 생기기 쉽고, 출혈성 질환의 위험도 증가하나, 혈중 콜레스테롤에 높은 경우에는 동맥벽에 침전물을 형성하여 동맥경화증을 일으킬 수 있으므로 적정량의 콜레스테롤을 유지하는 것이 중요함.|
    |13|triglyceride|float64|트리글리세리드(콜레스테롤과 함께 동맥 경화를 일으키는 혈중 지방 성분)<br>다른 명칭으로는 중성지방으로, 음식물로서 섭취하는 지방질의 95% 이상이 이에 해당한다.<br>트리글리세리드 수치가 흰 빵과 파스타와 같은 탄수화물이 풍부한 음식을 너무 많이 섭취한 결과 수치가 높아짐<br>적정 수치는 150 미만. 150~199, 200~499, 500을 기준으로 경계, 높음, 매우 높음 순.|
    |14|HDL|float64|고밀도 리포 단백질(HDL 콜레스테롤). 말초 체조직의 콜레스테롤을 간에 전송하는 작용을 한다. 낮을 경우 동맥경화의 위험성이 증가한다. 성별에 따른 정상 수치는 각각 남성 35~55, 여성 45~65.|
    |15|LDL|float64|저밀도 지방 단백질(LDL 콜레스테롤). 간 따위에서 합성된 콜레스테롤을 체조직에 운반하는 역할을 한다.<br>적정수치는 100미만으로, 정상수치는 100~129. 130~159는 경계, 160~189 높음. 190이상을 매우 높음으로 구분|
    |16|hemoglobin|float64|헤모글로빈 : 적혈구 속에 포함되어 있는 색소 단백질로 산소와 쉽게 결합하여 호흡에 중요한 역할을 함|

- 신장 질환
    |index|Feature|Data type|Description|
    |--|----|---|------|
    |17|Urine protein|float64|뇨 단백질 : 단백뇨의 원인이 되는 신장질환으로는 당뇨병성 신장질환, 원발성 사구체질환 등이 있을 수 있으며, 교원성 질환이나 혈관염 등도 원인이 될 수 있다. 단백뇨의 정의보다 적은 양(하루 30~300mg)의 단백이 배설되는 경우에도 미세 단백뇨라고 하여 당뇨병이나 고혈압, 사구체 신염에 의한 신장 질환의 초기 증세인 경우가 있다. 또한, 신장에 심각한 병이 없어도 간혹 소량의 단백뇨가 나올 수 있고 이러한 경우를 기능성 단백뇨라고 부른다.<br>성인인 경우 하루 500mg 이상, 소아는 1시간 동안 체표면적 1제곱미터당 4mg 이상의 단백이 배설될 때 명백한 단백뇨라고 한다.|
    |18|serum creatinine|float64|혈청크레아티닌 : 크레아티닌은 근육에서 생성되는 노폐물로 대부분 신장을 통해 배출되기 때문에 신장기능의 좋은 지표 보통 신장기능 평가를 위해 크레아티닌 검사와 혈액요소질소 검사를 함께 수행<br>정상범위 0.5-1.4mg/dL(넓게), 남성 : 0.7-1.2ml/dL, 여성 : 0.5-1.0ml/dL으로 진단 기준은 기관에 따라 편차가 있는데 1.5ml/dL이 넘어가면 추가적인 검사가 반드시 필요하다고 볼 수 있다.|

- 간 수치 : 간세포가 파괴되면 AST, ALT 등이 혈액을 돌아다니게 된다. 대개 건강한 사람도 수명이 다한 세포가 죽고 새로운 세포가 만들어지기 때문에 혈액에는 늘 소량의 AST, ALT가 있기 마련이다. 단 간에 염증이 생기거나 다른 이유로 간세포가 많이 파괴되면 혈액 속에 AST, ALT 수치가 올라가게 된다. 따라서 혈액검사에서 간수치가 높다는 것은 보통 AST, ALT라고 하는 간효소 수치가 증가했음을 나타낸다. 어떤 원인이든 간에 간세포가 손상을 받아 세포막이 파괴돼 효소들이 혈액으로 흘러나왔다는 것을 의미한다.
    |index|Feature|Data type|Description|
    |--|----|---|------|
    |19|AST|float64|아스파르테이트아미노전달효소. 간세포(hepatocyte) 외에 적혈구, 골격근(뼈대근육, skeletal muscle) 등에 분포하는 효소로, 세포가 괴사 · 파괴되면 혈중으로 유출된다. 간질환의 지표이며, 옛 명칭은 SGOT이다. (세포의 손상여부를 확인 가능)|
    |20|ALT|float64|알라닌아미노전달효소. 특히 간세포 내에 많이 함유되어 있는 효소로, 간이 장애를 입으면 혈 중 ALT 활성이 상승한다. 옛 명칭은 SGPT이다.
    |21|Gtp|float64|감마지티피는 간세포 내 담관에 존재하는 효소를 말하며 담즙 배설 장애가 있을 경우 증가하는 경향이 있다.<br>보통 술을 많이 마시는 사람이 수치가 높게 나올 가능성이 높고 감마지티피 정상 수치는 남성의 경우 11 ~ 63 IU/L, 여성의 경우 8 ~ 35 IU/L. 

- 구강 건강
    |index|Feature|Data type|Description|
    |--|----|---|------|
    |22|oral|object|구강|
    |23|dental caries|int64|충치 : 1/0|
    |24|tartar|object|치석 : Y/N<br>치석이란 치면세균막(덴탈 플라크)에 타액과 치은열구(치아와 잇몸 사이의 공간)액에서 유래한 칼슘(Ca), 인(P)등의 무기질이 침착되어 단단하게 굳어진 것을 말한다. 구성 성분은 무기물 75%, 유기물 6~15%이고 나머지가 수분이다. 균막(pellicle) 형성을 기초로 세균막(덴탈 플라크)이 성숙이 되고 이어 무기질화(칼슘, 인, 마그네슘, 불소)된 후에 결정체가 형성된다.|

- 흡연 여부
    |index|Feature|Data type|Description|
    |--|----|---|------|
    |25|smoking|int64|흡연여부 : 1/0|


데이터셋의 피처와 흡연과의 관계 탐색

hearing(left), hearing(right) - 청력검사
흡연과 청력손실의 유의한 연관성을 제시하며 금연이 청력을 유지 보호하는데 효과적임
KIM, Gyu-Sang. 소음과 청각 12-건강행태 (음주, 흡연 등) 와 청력영향. 월간산업보건, 2010, 11-24.

height, weight, waist - 체질량 지수
bmi = kg/m^2
대상자 3,501명 중 
비흡연자(725명)
체질량지수와 허리둘레 : 23.98 kg/m2, 84.14 cm
과거흡연자( 1,206명)
체질량지수와 허리둘레 : 24.43 kg/m2, 85.29 cm
흡연자(1,570명)
체질량지수와 허리둘레 : 23.66 kg/m2, 83.97 cm
과거 흡연자는 다른 두 군에 비해 체질량지수와 허리둘레가 의미 있게 컸다. 비만과 관련된 다양한 요소들을 보정했을 때, 
47세 이상에서 현재 흡연자는 비만에 대한 위험도가 비흡연자에 비해 의미 있게 감소하지만(P＜ 0.001), 과거흡연자에서는 비흡연자와 차이가 없었다.

이기헌, et al. 한국 남성 비만과 흡연의 관련성: 제 3 차 및 4 차 국민건강영양조사 자료 분석. 대한금연학회지, 2010, 1.2: 115-123.


fasting blood sugar - 공복혈당 검사
인슐링 저항성 및 2형 당뇨병 발병은 염증과 관련이 있다는 보고가 있음
염증 지표인 백혈구 수 증가와 고혈당증의 발생위험도가 증가함
특히 현재 흡연군과 과거 흡연군에서 고혈당증의 발생 위험이 증가함 이러한 결과는 현재 흡연군자와 과거흡연자에서 만성적인 저강도의 염증이 고혈당증의 위험인자임을 시사함
이용제, et al. 흡연 정도에 따른 백혈구 수 증가와 고혈당증의 관계. 가정의학회지, 2007, 28.1: 32-38.

공복혈당의 정상수치는 100, 100~125구간일 대 당뇨 전단계로 판단됨
제공된 데이터셋에서 당뇨 전단계일 것으로 판단되는 사람은 18381명.
(smoking['fasting blood sugar']>100).sum()
	

Cholesterol, triglyceride, HDL, LDL - 지질검사
흡연의 지질에 대한 영향은 HDL 콜레스테롤을 감소시키고 중성지방을 상승시키며 LDL 콜레스테롤은 변화가 뚜렷하지 않은 것으로 알려져 있다.
김치정, Diet and Life Style Modification for Hypercholesterolemia, 2004. 중앙대학교 의과대학

흡연자 2,421명(35.2%)중 
남자 2,168명(89.6%), 여자 253명(10.4%),
남성 흡연자의 평균혈중지질
총콜레스테롤(Cholesterol) 186.3±36.8mg/dl, 
중성지방(triglyceride) 138.7±67.0mg/dl
고밀도지단백콜레스테롤(HDL) 48.3±12.6mg/dl
저밀도지단백콜레 스테롤(LDL) 110.2±33.8mg/dl
비흡연자 4,456명(64.8%)중 
남자 600명(13.5%), 여자 3856명(86.5)이었다. 
비흡연자의 평균혈중지질
총콜레스테롤(Cholesterol) 185.2±35.1mg/dl
중성지방(triglyceride) 126.6±62.2mg/dl
고밀도지단백콜레스테롤(HDL) 47.2±11.8mg/dl
저밀도지단 백콜레스테롤(LDL) 112.6±32.1mg/dl

김진옥. 현재 흡연자와 비흡연자의 혈중지질 수준 비교. 2003. PhD Thesis. 연세대학교 보건대학원.
데이터 제공자는 흡연과 지질에 대한 영향을 이해하고 위 지표들을 포함시킨 것으로 보임

hemoglobin - 혈색소 검사
남자의 경우 
비 흡연군에서의 혈색소치는 14.60±0.92 g/dl 였고, 
하루 한 갑 이하로 담배를 피는 흡연군의 경우 혈색소치는 14.81±0.90 g/dl, 
1갑 이상 담배를 피는 흡연군 경우 혈색소치는 15.19±0.84 g/dl였다. 
다중회귀분석에서 혈색소의 나이, 하루 흡연량, 체질량지수가 영향을 미쳤고, 나이와 일일흡연량, 흡연 연수를 보정한 상태에서도 혈색소와 흡연의 양과의 연관관계가 있었다. 
YUN, Suk Hyun, et al. Difference in hemoglobin between smokers and non-smokers. Journal of the Korean Academy of Family Medicine, 2002, 23.1: 80-86.
흡연과 헤모글로빈 수치는 양의 상관을 보이는 연구결과

Urine protein, serum creatinine 요단백/크레아티닌 검사
흡연여부에서는 남성의 경우 현재흡연자와 과거흡연자의 요중 크레아티닌 농도가 보정전과 보정 후 모두 유의한 차이를 보였다.
정경식; 김남수; 이병국. 한국인의 요중 크레아티닌 농도에 관한 연구− 국민건강영양조사 4 기 자료 이용−. 한국환경보건학회지, 2012, 38.1: 31-41.
연구팀은 신장긴응 진단 지표인 크레아티닌의 혈중수치를 근거로 신자의 여과기능을 나타내는 사구체여과율을 산출한 다음 니코틴의 대사산물인 코티닌의 혈중수치와 연관성을 분석했다.
그 결과 코티닌 수치가 높아질수록 사구체여과율은 줄어드는 것으로 나타났다
크레아티닌은 체내에서 에너지로 사용된 단백질의 노폐물로 신장의 사구체에서 여과된다.
금연길라잡이 : https://www.nosmokeguide.go.kr/lay2/bbs/S1T98C103/A/26/view.do?article_seq=258227&only_one=Y


AST, ALT, Gtp - 간기능 검사
WHO산하의 국제암연구소 보고서에 따르면 흡연은 1급 발암요인으로 간암을 유발한다는 충분한 과학적 근거가 있다고 밝히고 있음
흡연을 통해 몸에 흡수된 발암물질의 대부분은 간에서 대사되기 때문에 간에 나쁜 영향을 주는 것으로 설명할 수 있다.
간염 바이러스나 음주와 같은 간암의 다른 위험요인을 고려한 경우에도 흡연으로 인한 간암 위험이 60% 증가하는 것으로 여러 연구를 통해 증명됨. 
따라서 간암을 예방하기 위해서는 B형간염 예방접종을 받고, 술을 절제할 뿐만 아니라 흡연을 피해야 한다.
금연길라잡이 : https://www.nosmokeguide.go.kr/lay2/bbs/S1T33C109/H/22/view.do?mode=view&article_seq=278&tag_name=&cpage=1&rows=10&condition=&keyword=&cat=&rn=


oral, dental caries, tartar - 구강 검진
흡연량과 치주질환의 역할기전은 명백하진 않지만 여러 연구들을 통해서 흡연과 치주질환의 빈도와 심도 사이에는 상관관계가 있으며 일반적으로 흡연은 숙주의 정상적인 면역기능을 저해하고 변형시켜 주위의 건강한 치주조직을 파괴시킨다고 알려져 있다.
계승범; 한수부. 흡연량과 흡연 기간에 따른 치주 상태. 대한치주과학회지: Vol, 2001, 31.4.

데이터 제공자는 위 사실을 인지하고있는 것으로 예상되나 치주질환에 대한 특성을 oral(구강검사상태), dantal caries(충치 유무), tartar(치석) 정도로 포함시키는 것으로 그침