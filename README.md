# MTTV
Multi-modal transformer using two-level visual features for fake news detection
## 实验环境
    python 3.8.8
    pytorch 1.9.0+cu102
    torchvision 0.2.1
    pytorch-pretrained-bert 0.6.2
## 数据集
### (1)下载数据
数据集：Fakeddit、微博和Style-based Fake  
您需要从以下链接下载原始数据集，以获取图像文件：  
Fakeddit：https://github.com/entitize/Fakeddit   
微博：https://github.com/yaqingwang/EANN-KDD18  
Style-based Fak：https://github.com/junyachen/Data-examples   
### (2)提取数据集的图像特征
```Python
python extract_image_features.py --dataset_dir MTTV/data/weibo/ --image_dir ${your_Fakeddit_image_dir} --feature_dir ./data/weibo/  
python extract_image_features.py --dataset_dir MTTV/data/fakeddit/ --image_dir ${your_weibo_image_dir} --feature_dir ./data/fakeddit/  
python extract_image_features.py --dataset_dir MTTV/data/gossip/ --image_dir ${your_Style-based-Fake_image_dir} --feature_dir ./data/gossip/
```
## 训练
进入目录``MTTV/``，Style-based Fake数据集在MTTV模型上训练
```Python
python train.py --task v3-1_style_based_fake --label_type 2_way_label --batch_sz 32 --gradient_accumulation_steps 20 --max_epochs 20 --name fakeddit_2_way --bert_model bert-base-uncased --global_image_embeds 5 --region_image_embeds 20 --num_image_embeds 25
```  
进入目录``MMBT/``，Style-based Fake数据集在MMBT模型上训练
```Python
python train.py --batch_sz 32 --gradient_accumulation_steps 40 \
 --savedir ./savedir_gossip/ --name mmbt_model_run \
 --data_path MTTV/data/gossip/ \
 --task weibo --task_type classification \
 --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  \
 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1
```

## 结果
Fakeddit  
<img src="https://github.com/Preciousrs/MTTV/blob/main/fakeddit.png" width="800" height="300" />  
微博  
<img src="https://github.com/Preciousrs/MTTV/blob/main/weibo.png" width="800" height="300" />  
Style-based Fake  
<img src="https://github.com/Preciousrs/MTTV/blob/main/Style-based%20Fake.png" width="800" height="300" />  





