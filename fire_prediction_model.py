import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
# 한글 설정
plt.rcParams["font.family"] = "Malgun Gothic"

# 데이터 불러오기
df=pd.read_csv(".\data\소방청_화재현황.csv")

# 데이터 확인
df.info()
unique_values = df['화재유형'].unique()
print(unique_values)

# 필요한 특성 및 타겟 설정
features = ['사망', '부상', '재산피해소계', '인명피해(명)소계', '화재발생(년)', '화재발생(월)', '화재발생(일)', '화재발생(시)']
target = '화재유형'

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# 모델 선택 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 혼동 행렬 계산
cm = confusion_matrix(y_test, y_pred, labels=unique_values)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_values, yticklabels=unique_values)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 클래스별 정밀도, 재현율, F1 점수 계산
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, labels=unique_values)

# 시각화
plt.figure(figsize=(12, 6))
plt.bar(unique_values, precision, label='Precision')
plt.bar(unique_values, recall, label='Recall')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Precision and Recall by Class')
plt.legend()
plt.show()

# 문자열을 더미 변수로 변환
X = pd.get_dummies(df[features])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, df[target], test_size=0.2, random_state=42)

# RandomForestClassifier 모델 선택
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print("Classification Report:")
print(classification_report(y_test, y_pred))

# k-fold 교차 검증
cv_scores = cross_val_score(model, X, df[target], cv=5, scoring='accuracy')
print(f"\n평균 정확도: {cv_scores.mean()}")
