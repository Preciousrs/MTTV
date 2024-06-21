

import pandas as pd
import os
import torch
from sklearn.utils import shuffle
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 标签映射
label_map = {"fake": 0, "real": 1}

# 加载训练数据
train_data = []
with open('/data/rensisi/HMCAN/MTTV/data/weibo/train.json', 'r') as f:
    for line in f:
        sample = json.loads(line)
        train_data.append({'text': sample['text'], 'labels': label_map[sample['label']]})  # 将标签转换为数值

# 加载测试数据
test_data = []
with open('/data/rensisi/HMCAN/MTTV/data/weibo/test.json', 'r') as f:
    for line in f:
        sample = json.loads(line)
        test_data.append({'text': sample['text'], 'labels': label_map[sample['label']]})  # 将标签转换为数值

# 创建DataFrame
train_df_clean = pd.DataFrame(train_data)
eval_df_clean = pd.DataFrame(test_data)

# 打印数据大小
print(f"Train Data Size: {len(train_df_clean)}")
print(f"Eval Data Size: {len(eval_df_clean)}")

# 打乱训练数据
train_df_clean = shuffle(train_df_clean).reset_index(drop=True)

# 打印训练数据的前几行
print(train_df_clean.head())

# 定义训练参数
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 1,
    'train_batch_size': 32,
    'eval_batch_size': 64,
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 200,
    'save_eval_checkpoints': False,
    'save_model_every_epoch': False,
    'output_dir': './outputs/',
    'cache_dir': './cache/',
    'no_cache': True,
    'fp16': False,
    'max_seq_length':64,
    'save_steps': -1,
    'logging_steps': 100
}

# 初始化模型
model_BERT = ClassificationModel(
    'bert',
    # 'bert-base-cased',
    'bert-base-chinese',
    num_labels=2,
    args=train_args,
    use_cuda=torch.cuda.is_available()
)

# 训练模型
model_BERT.train_model(train_df_clean, eval_df=eval_df_clean)

# 评估模型
def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 执行评估
result, model_outputs, wrong_predictions = model_BERT.eval_model(
    eval_df_clean,
    acc=accuracy_score,
    prec=precision_score,
    rec=recall_score,
    f1=f1_score,
    verbose=True
)

# 提取并打印评估结果
accuracy = result['acc']
precision = result['prec']
recall = result['rec']
f1 = result['f1']

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")