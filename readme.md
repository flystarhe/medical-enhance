# medical-enhance
v0.1

## install
```bash
git clone --recurse-submodules --depth 1 xxx
```

>基础环境：CUDA10.0、GCC7.3、Anaconda(py36)、OpenCV和PyTorch。

## python *.py
```bash
cd $PROJ_HOME
PYTHONPATH=`pwd`/module_mmdetection python *.py
PYTHONPATH=`pwd`/module_mmdetection nohup python *.py >> log.txt 2>&1 &
```

## notes
git:
```bash
git checkout --orphan latest
git add .
git commit -m "v1.0"
git branch -D master
git branch -m master
git push -f origin master
git push --set-upstream origin master

git remote add origin xxx
git remote set-url origin xxx
git push -u origin master
```

bash:
```bash
cd $PROJ_HOME
rm -rf data/coco  # 末尾没有斜杠
ln -s $DATA_ROOT data/coco
cd $DATA_ROOT && rm -rf data_train.json && ln -s $DATA_TRAIN coco_train.json
cd $DATA_ROOT && rm -rf data_test.json && ln -s $DATA_TEST coco_test.json
```

notebook:
```bash
nohup jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook0.ipynb > log.00 2>&1 &
nohup jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook*.ipynb > log.00 2>&1 &
nohup jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook1.ipynb notebook2.ipynb > log.00 2>&1 &
```
