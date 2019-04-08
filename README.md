# How to Do Text Binary Classification with BERT in Less than Ten Lines of Code?

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-20-03-53-720338.jpg)

# Demand

We all know BERT is a compelling language model which has already been applied to various kinds of downstream tasks, such as sentiment analysis and QA. 

Have you ever tried it on text binary classification?

Honestly, till the beginning of this week, my answer was still **NO**.

Why?

Because the example code on BERT's official GitHub repo was **not very friendly**.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928502.jpeg)

Firstly, I want an IPython Notebook, instead of a Python script file, for I want to get instant feedback when I run a code chunk. Of course, a Google Colab Notebook would be better, for I can use the code right away with the free GPU.

Secondly, I don't want to know the detail except for the ones I care. For example, I want to control the useful parameters, such as the number of epochs and batch size. However, do I need to know all the "processors," "flags" and logging functions?

I am a spoiled machine learning user after I tried all other friendly frameworks.

For example, in Scikit-learn, if you try to build a tree classifier, here is (almost) all your code.

```python
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
```

If you want to do image classification in fast.ai, you need to input these lines.

```python
!git clone https://github.com/wshuyi/demo-image-classification-fastai.git
from fastai.vision import *
path = Path("demo-image-classification-fastai/imgs/")
data = ImageDataBunch.from_folder(path, test='test', size=224)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(8, 8))
```

Not only you can get the classification result, but an activation map as well.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928501.png)

Why on earth cannot Google Developers give us a similar interface to use BERT for text classification?

On Monday, I found [this Colab Notebook](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb). It's an example of predicting sentiment of movie reviews.

I was so excited, for I learned BERT is now included in Tensorflow Hub.

However, when I opened it, I found there are still too many details for a user who only cares about the application of text classification.

So I tried to refactor the code, and I made it.

# Notebook

Please follow [this link](https://github.com/wshuyi/demo_text_binary_classification_bert/blob/master/demo_text_binary_classification_with_bert.ipynb) and you will see the IPynb Notebook file on github.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-20-03-53-720345.png)

Click the "Open in Colab" Button. Google Colab will be opened automatically.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-20-03-53-720336.png)

You need to **save a copy** to your own Google Drive by clicking on the "COPY TO DRIVE" button.

You only need to do three things after that.

1. Prepare the data in Pandas Data frame format. I guess it's easy for most deep learning users.
2. Adjust four parameters if necessary.
3. Run the notebook and get your result displayed.

I will explain to you in detail.

When you open the notebook, you may feel angry.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928505.png)

> You Liar! You promised there are only less than 10 lines!

Calm down.

You **don't need** to change anything until this line.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928507.png)

So go to this line, and Click the `Run before` button.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928498.png)

Let us focus on the really important part.

You need to get the data ready.

My example is a sample dataset of IMDB reviews. It contains 1000 positive and 1000 negative samples in training set, while the testing set contains 500 positive and 500 negative samples.

```python
!wget https://github.com/wshuyi/info-5731-public/raw/master/imdb-sample.pickle

with open("imdb-sample.pickle", 'rb') as f:
    train, test = pickle.load(f)
```

I used it in my INFO 5731 class at UNT to let students compare the result of textblob package, Bag of Words model, simple LSTM with word embedding, and ULMfit.

Now I think I can add BERT into the list, finally.

You need to run the following line to make sure the training data is shuffled correctly.

```python
train = train.sample(len(train))
```

Now you can look into your data, see if everything goes smoothly.

```python
train.head()
```

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928500.jpeg)

Your dataset should be stored in Pandas Data Frame. There should be one training set, called `train` and one testing set, called `test`.

Both of them should at least contain two columns. One column is for the text, and the other one is for the binary label. It is highly recommended to select 0 and 1 as label values.

Now that your data is ready, you can set the parameters.

```python
myparam = {
        "DATA_COLUMN": "text",
        "LABEL_COLUMN": "sentiment",
        "LEARNING_RATE": 2e-5,
        "NUM_TRAIN_EPOCHS":10
    }
```

The first two parameters are just the name of columns of your data frame. You can change them accordingly.

The third parameter is the learning rate. You need to read the original paper to figure out how to select it wisely. Alternatively, you can use this default setting.

The last parameter is to set how many epochs you want BERT to run. I chose 10 here, for the training dataset is very small, and I don't want it overfits.

Okay. Now you can run BERT!

```python
result, estimator = run_on_dfs(train, test, **myparam)
```

Warning! This line takes you **some time** to run.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928499.png)

When you see some message like this, you know the training phase has finished.

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-20-03-53-720337.png)

So you can run the last line to get evaluation result of your classification model (on BERT) in a pretty form.

```python
pretty_print(result)
```

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-20-03-53-720339.png)

For such a small training set, I think the result is quite good.

That's all.

Now you can use the **state of the art** language modeling technique to train your text binary classifier too!

By the way, if you are interested, please help me to package the code before that line so that it looks even more straightforward. Thanks!

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928506.png)

# Related Blogs

If you are interested in this blog article, you may also want to read the following ones:


- [How to Practice Python with Google Colab? ](https://towardsdatascience.com/how-to-practice-python-with-google-colab-45fc6b7d118b)
- [How to Predict Severe Traffic Jams with Python and Recurrent Neural Networks?](https://towardsdatascience.com/how-to-predict-severe-traffic-jams-with-python-and-recurrent-neural-networks-e53b6d411e8d)
- [Deep Learning with Python and fast.ai, Part 1: Image classification with pre-trained model](https://medium.com/datadriveninvestor/deep-learning-with-python-and-fast-ai-part-1-image-classification-with-pre-trained-model-cd9364107872)
- [Deep Learning with Python and fast.ai, Part 2: NLP Classification with Transfer Learning](https://medium.com/datadriveninvestor/deep-learning-with-python-and-fast-ai-part-2-nlp-classification-with-transfer-learning-e7aaf7514e04)



