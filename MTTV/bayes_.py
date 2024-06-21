import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 加载数据
data = []
with open('/data/rensisi/HMCAN/MTTV/data/fakeddit/train.json', 'r') as f:
    for line in f:
        sample = json.loads(line)
        data.append({'text': sample['text'], 'label': sample['6_way_label']})

test_data = []
with open('/data/rensisi/HMCAN/MTTV/data/fakeddit/test.json', 'r') as f:
    for line in f:
        sample = json.loads(line)
        test_data.append({'text': sample['text'], 'label': sample['6_way_label']})

# 特征提取
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([sample['text'] for sample in data])
y_train = [sample['label'] for sample in data]

X_test = vectorizer.transform([sample['text'] for sample in test_data])
y_test = [sample['label'] for sample in test_data]

# 训练朴素贝叶斯模型
nb = MultinomialNB()
nb.fit(X_train, y_train)

# 评估模型
y_pred = nb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall: {rec:.4f}')
print(f'F1-score: {f1:.4f}')