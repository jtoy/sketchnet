## Usage 



#### 1. Generate image/source code pairings

```bash
$ rewrite
```

#### 2. Preprocessing

```bash
$ python build_vocab.py   
$ python resize.py
```

#### 3. Train the model

```bash
$ python train.py    
```

#### 4. Generate captions


```bash
$ python sample.py --image='path_for_image'
```

<br>

## Pretrained model 

If you do not want to train the model yourself, you can use a pretrained model. I have provided the pretrained model as a zip file. You can download the file [here](https://www.dropbox.com/s/b7gyo15as6m6s7x/train_model.zip?dl=0) and extract it to `./models/` directory.
