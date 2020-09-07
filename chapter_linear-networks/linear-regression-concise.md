# Doğrusal Regresyonunun Kısa Uygulaması
:label:`sec_linear_concise`

Son birkaç yıldır derin öğrenmeye olan geniş ve yoğun ilgi, gradyan tabanlı öğrenme algoritmalarını uygulamanın tekrarlayan işyükünü otomatikleştirmek için şirketler, akademisyenler ve amatör geliştiricilere çeşitli olgun açık kaynak çerçeveleri geliştirmeleri için ilham verdi. :numref: `sec_linear_scratch`'de, biz sadece (i) veri depolama ve doğrusal cebir için tensörlere; ve (ii) gradyanları hesaplamak için otomatik türev almaya güvendik. Pratikte, veri yineleyiciler, kayıp işlevleri, optimize ediciler ve sinir ağı katmanları çok yaygın olduğu için, modern kütüphaneler bu bileşenleri bizim için de uygular.

Bu bölümde, derin öğrenme çerçevelerinin üst düzey API'lerini kullanarak :numref: `sec_linear_scratch`daki doğrusal regresyon modelini kısaca nasıl uygulayacağınızı göstereceğiz.

## Veri Kümesini Oluşturma

Başlamak için, şuradaki aynı veri kümesini oluşturacağız: :numref:`sec_linear_scratch`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data

true_w = torch.Tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
labels = labels.reshape(-1,1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf

true_w = tf.constant([2, -3.4], shape=(2, 1))
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
labels = tf.reshape(labels, (-1, 1))
```

## Veri Kümesini Okuma

Kendi yineleyicimizi döndürmek yerine, verileri okumak için bir çerçevedeki mevcut API'yi çağırabiliriz. `Öznitelikler` (`features`) ve `etiketler`'i (`labels`) bağımsız değişken olarak iletiriz ve bir veri yineleyici nesnesi başlatırken `batch_size` (grup boyutu) belirtiriz. Ayrıca, mantıksal veri tipi (boolean) değeri `is_train`, veri yineleyici nesnesinin her bir dönemdeki (epoch) verileri karıştırmasını isteyip istemediğimizi gösterir (veri kümesinden geçer).

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Şimdi, `data_iter`i, `data_iter` işlevini :numref:`sec_linear_scratch`'de çağırdırdığımız şekilde kullanabiliriz. Çalıştığını doğrulamak için, örneklerin ilk mini grubunu okuyabilir ve yazdırabiliriz. :numref:`sec_linear_scratch`'deki ile karşılaştırıldığında, burada bir Python yineleyici oluşturmak için `iter` kullanıyoruz ve yineleyiciden ilk öğeyi elde etmek için `next`'i kullanıyoruz.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## Modeli Tanımlama

Doğrusal regresyonu :numref:`sec_linear_scratch`'da sıfırdan uyguladığımızda, model parametrelerimizi açık bir şekilde tanımladık ve temel doğrusal cebir işlemlerini kullanarak çıktı üretmek için hesaplamalarımızı kodladık. Bunu nasıl yapacağınızı *bilmelisiniz*. Ancak modelleriniz daha karmaşık hale geldiğinde ve bunu neredeyse her gün yapmanız gerektiğinde, bu yardım için memnun olacaksınız. Durum, kendi blogunuzu sıfırdan kodlamaya benzer. Bunu bir veya iki kez yapmak ödüllendirici ve öğreticidir, ancak bir bloga her ihtiyaç duyduğunuzda tekerleği yeniden icat etmek için bir ay harcarsanız kötü bir web geliştiricisi olursunuz.

Standart işlemler için, uygulamaya (kodlamaya) odaklanmak yerine özellikle modeli oluşturmak için kullanılan katmanlara odaklanmamızı sağlayan bir çerçevenin önceden tanımlanmış katmanlarını kullanabiliriz. Önce, `Sequential` (ardışık, sıralı) sınıfının bir örneğini ifade edecek `net` (ağ) model değişkenini tanımlayacağız. `Sequential` sınıfı, birbirine zincirlenecek birkaç katman için bir kap (container) tanımlar. Girdi verileri verildiğinde, `Sequential` bir örnek, bunu birinci katmandan geçirir, ardından onun çıktısını ikinci katmanın girdisi olarak geçirir ve böyle devam eder. Aşağıdaki örnekte, modelimiz yalnızca bir katmandan oluşuyor, bu nedenle gerçekten `Sequential` örneğe ihtiyacımız yok. Ancak, gelecekteki modellerimizin neredeyse tamamı birden fazla katman içereceği için, sizi en standart iş akışına alıştırmak için yine de kullanacağız.

Tek katmanlı bir ağın mimarisini şurada gösterildiği gibi hatırlayın :numref:`fig_singleneuron`. Katmanın *tamamen bağlı* olduğu söylenir, çünkü girdilerinin her biri, bir matris-vektör çarpımı yoluyla çıktılarının her birine bağlanır.

:begin_tab:`mxnet`
Gluon'da tamamen bağlı katman `Dense` (Yoğun) sınıfında tanımlanır. Sadece tek bir skaler çıktı üretmek istediğimiz için, bu sayıyı 1 olarak ayarladık.

Kolaylık sağlamak için Gluon'un her katman için girdi şeklini belirlememizi gerektirmediğini belirtmek gerekir. Yani burada, Gluon'a bu doğrusal katmana kaç girdi girdiğini söylememize gerek yok. Modelimizden ilk veri geçirmeye çalıştığımızda, örneğin, daha sonra `net(X)`'i çalıştırdığımızda, Gluon otomatik olarak her katmana girdi sayısını çıkaracaktır. Bunun nasıl çalıştığını daha sonra daha ayrıntılı olarak anlatacağız.
:end_tab:

:begin_tab:`pytorch`
PyTorch'ta, tam bağlantılı katman `Linear` sınıfında tanımlanır. `nn.Linear`'e iki bağımsız değişken aktardığımıza dikkat edin. Birincisi, 2 olan girdi öznitelik boyutunu belirtir ve ikincisi, tek bir skaler olan ve dolayısıyla 1 olan çıktı öznitelik boyutudur.
:end_tab:

:begin_tab:`tensorflow`
Keras'ta tamamen bağlı katman `Dense` (Yoğun) sınıfında tanımlanır. Sadece tek bir skaler çıktı üretmek istediğimiz için, bu sayıyı 1 olarak ayarladık.

Kolaylık sağlamak için Gluon'un her katman için girdi şeklini belirlememizi gerektirmediğini belirtmek gerekir. Yani burada, Keras'a bu doğrusal katmana kaç girdi girdiğini söylememize gerek yok. Modelimizden ilk veri geçirmeye çalıştığımızda, örneğin, daha sonra `net(X)`'i çalıştırdığımızda, Keras otomatik olarak her katmana girdi sayısını çıkaracaktır. Bunun nasıl çalıştığını daha sonra daha ayrıntılı olarak anlatacağız.
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## Model Parametrelerini İlkletme

`net`'i kullanmadan önce, doğrusal regresyon modelindeki ağırlıklar ve ek girdi gibi model parametrelerini ilkletmemiz gerekir. Derin öğrenme çerçeveleri genellikle parametreleri ilklemek için önceden tanımlanmış bir yola sahiptir. Burada, her ağırlık parametresinin, ortalama 0 ve standart sapma 0.01 ile normal bir dağılımdan rastgele örneklenmesi gerektiğini belirtiyoruz. Ek girdi parametresi sıfır olarak başlatılacaktır.

:begin_tab:`mxnet`
MXNet'ten `initializer` modülünü içe aktaracağız. Bu modül, model parametresi ilkletme için çeşitli yöntemler sağlar. Gluon, `init`'i `initializer` paketine erişmek için bir kısayol (kısaltma) olarak kullanılabilir hale getirir. Ağırlığın nasıl ilkleneceğini sadece `init.Normal(sigma=0.01)`'i çağırarak belirtiyoruz. Ek girdi parametreleri varsayılan olarak sıfıra başlatılır.
:end_tab:

:begin_tab:`pytorch`
`nn.Linear` oluştururken girdi ve çıktı boyutlarını belirttik. Şimdi, başlangıç değerlerini belirtmek için parametrelere doğrudan erişiyoruz. İlk olarak  ağdaki ilk katmanı `net[0]` ile buluruz ve ardından parametrelere erişmek için `weight.data` ve `bias.data` yöntemlerini kullanırız. Daha sonra, parametre değerlerinin üzerine yazmak için `uniform_` ve `fill_` değiştirme yöntemlerini kullanırız.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow'daki `initializers` modülü, model parametresi ilkletme için çeşitli yöntemler sağlar. Keras'ta ilkletme yöntemini belirlemenin en kolay yolu, katmanı `kernel_initializer` belirterek oluşturmaktır. Burada `net`'i yeniden oluşturuyoruz.
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.uniform_(0.0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
Yukarıdaki kod basit görünebilir, ancak burada tuhaf bir şeylerin olduğunu fark etmelisiniz. Gluon, girdinin kaç boyuta sahip olacağını henüz bilmese de, bir ağ için parametreleri ilkletebiliyoruz! Örneğimizdeki gibi 2 de olabilir veya 2000 de olabilir. Gluon bunun yanına kalmamıza izin veriyor çünkü sahnenin arkasında, ilk değerleri atama aslında *ertelendi*. Gerçek ilkleme, yalnızca verileri ağ üzerinden ilk kez geçirmeye çalıştığımızda gerçekleşecektir. Unutmayın ki, parametreler henüz başlatılmadığı için bunlara erişemeyiz veya onları değiştiremeyiz.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
Yukarıdaki kod basit görünebilir, ancak burada tuhaf bir şeylerin olduğunu fark etmelisiniz. Keras, girdinin kaç boyuta sahip olacağını henüz bilmese de, bir ağ için parametreleri ilkletebiliyoruz! Örneğimizdeki gibi 2 de olabilir veya 2000 de olabilir. Keras bunun yanına kalmamıza izin veriyor çünkü sahnenin arkasında, ilk değerleri atama aslında *ertelendi*. Gerçek ilkleme, yalnızca verileri ağ üzerinden ilk kez geçirmeye çalıştığımızda gerçekleşecektir. Unutmayın ki, parametreler henüz başlatılmadığı için bunlara erişemeyiz veya onları değiştiremeyiz.
:end_tab:

## Defining the Loss Function

:begin_tab:`mxnet`
In Gluon, the `loss` module defines various loss functions. In this example, we will use the Gluon implementation of squared loss (`L2Loss`).
:end_tab:

:begin_tab:`pytorch`
The `MSELoss` class computes the mean squared error, also known as squared L2 norm. By default it returns the average loss over examples.
:end_tab:

:begin_tab:`tensorflow`
The `MeanSquaredError` class computes the mean squared error, also known as squared L2 norm. By default it returns the average loss over examples.
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## Defining the Optimization Algorithm

:begin_tab:`mxnet`
Minibatch stochastic gradient descent is a standard tool for optimizing neural networks and thus Gluon supports it alongside a number of variations on this algorithm through its `Trainer` class. When we instantiate `Trainer`, we will specify the parameters to optimize over (obtainable from our model `net` via `net.collect_params()`), the optimization algorithm we wish to use (`sgd`), and a dictionary of hyperparameters required by our optimization algorithm. Minibatch stochastic gradient descent just requires that we set the value `learning_rate`, which is set to 0.03 here.
:end_tab:

:begin_tab:`pytorch`
Minibatch stochastic gradient descent is a standard tool for optimizing neural networks and thus PyTorch supports it alongside a number of variations on this algorithm in the `optim` module. When we instantiate an `SGD` instance, we will specify the parameters to optimize over (obtainable from our net via `net.parameters()`), with a dictionary of hyperparameters required by our optimization algorithm. Minibatch stochastic gradient descent just requires that we set the value `lr`, which is set to 0.03 here.
:end_tab:

:begin_tab:`tensorflow`
Minibatch stochastic gradient descent is a standard tool for optimizing neural networks and thus Keras supports it alongside a number of variations on this algorithm in the `optimizers` module. Minibatch stochastic gradient descent just requires that we set the value `learning_rate`, which is set to 0.03 here.
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## Training

You might have noticed that expressing our model through high-level APIs of a deep learning framework requires comparatively few lines of code. We did not have to individually allocate parameters, define our loss function, or implement minibatch stochastic gradient descent. Once we start working with much more complex models, advantages of high-level APIs will grow considerably. However, once we have all the basic pieces in place, the training loop itself is strikingly similar to what we did when implementing everything from scratch.

To refresh your memory: for some number of epochs, we will make a complete pass over the dataset (`train_data`), iteratively grabbing one minibatch of inputs and the corresponding ground-truth labels. For each minibatch, we go through the following ritual:

* Generate predictions by calling `net(X)` and calculate the loss `l` (the forward pass).
* Calculate gradients by running the backpropagation.
* Update the model parameters by invoking our optimizer.

For good measure, we compute the loss after each epoch and print it to monitor progress.

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

Below, we compare the model parameters learned by training on finite data and the actual parameters that generated our dataset. To access parameters, we first access the layer that we need from `net` and then access that layer's weights and bias. As in our from-scratch implementation, note that our estimated parameters are close to their ground-truth counterparts.

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## Summary

:begin_tab:`mxnet`
* Using Gluon, we can implement models much more concisely.
* In Gluon, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers, and the `loss` module defines many common loss functions.
* MXNet's module `initializer` provides various methods for model parameter initialization.
* Dimensionality and storage are automatically inferred, but be careful not to attempt to access parameters before they have been initialized.
:end_tab:

:begin_tab:`pytorch`
* Using PyTorch's high-level APIs, we can implement models much more concisely.
* In PyTorch, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers and common loss functions.
* We can initialize the parameters by replacing their values with methods ending with `_`.
:end_tab:

:begin_tab:`tensorflow`
* Using TensorFlow's high-level APIs, we can implement models much more concisely.
* In TensorFlow, the `data` module provides tools for data processing, the `keras` module defines a large number of neural network layers and common loss functions.
* TensorFlow's module `initializers` provides various methods for model parameter initialization.
* Dimensionality and storage are automatically inferred (but be careful not to attempt to access parameters before they have been initialized).
:end_tab:

## Exercises

:begin_tab:`mxnet`
1. If we replace `l = loss(output, y)` with `l = loss(output, y).mean()`, we need to change `trainer.step(batch_size)` to `trainer.step(1)` for the code to behave identically. Why?
1. Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules `gluon.loss` and `init`. Replace the loss by Huber's loss.
1. How do you access the gradient of `dense.weight`?

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. If we replace `nn.MSELoss(reduction='sum')` with `nn.MSELoss()`, how can we change the learning rate for the code to behave identically. Why?
1. Review the PyTorch documentation to see what loss functions and initialization methods are provided. Replace the loss by Huber's loss.
1. How do you access the gradient of `net[0].weight`?

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. Review the TensorFlow documentation to see what loss functions and initialization methods are provided. Replace the loss by Huber's loss.

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
