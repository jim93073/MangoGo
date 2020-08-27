# Mango
###### tags: `mango`

1. Download Mango dataset
```bash=
https://drive.google.com/file/d/1wuML5BTpjWJokckH-DqGudp4TUOBLeSZ/view?usp=sharing
```
2. Download code
```bash=
git clone https://github.com/jim93073/MangoGo.git
```

- ImgCrop.ipynb
train.csv與dev.csv內有芒果ID、等級以及芒果座標位置
使用ImgCrop.ipynb，將目標芒果切出來，並存成新檔案
- tfrecords/train_TFRecord.ipynb
將芒果圖片存成tfRecord格式
- Mango.ipynb
將tfRecord資料進行訓練