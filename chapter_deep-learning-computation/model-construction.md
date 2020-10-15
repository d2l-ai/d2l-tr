# Katmanlar ve Bloklar
:label:`sec_model_construction`

Sinir ağlarını ilk tanıttığımızda, tek çıktılı doğrusal modellere odaklandık. Burada tüm model sadece tek bir nörondan oluşuyor. Tek bir nöronun (i) bazı girdiler aldığını; (ii) karşılık gelen (*skaler*) bir çıktı ürettiğini; ve (iii) ilgili bazı amaç işlevlerini optimize etmek için güncellenebilen bir dizi ilişkili parametreye sahip olduğunu unutmayın. Sonra, birden çok çıktıya sahip ağları düşünmeye başladığımızda, tüm bir nöron *katmanını* karakterize etmek için vektörleştirilmiş aritmetikten yararlandık. Tıpkı bireysel nöronlar gibi, katmanlar (i) bir dizi girdi alır, (ii) karşılık gelen çıktıları üretir ve (iii) bir dizi ayarlanabilir parametre ile açıklanır. Softmaks bağlanımı üzerinde çalıştığımızda, tek bir *katmanın* kendisi *modeldi*. Bununla birlikte, daha sonra çok katmanlı algılayıcıları devreye soktuğumuzda bile, modelin bu aynı temel yapıyı koruduğunu düşünebiliriz.

İlginç bir şekilde, çok katmanlı algılayıcılar için hem *tüm model* hem de *kurucu katmanları* bu yapıyı paylaşır. (Tam) Model ham girdileri (öznitelikleri) alır, çıktılar (tahminler) üretir ve parametrelere (tüm kurucu katmanlardan birleşik parametreler) sahiptir. Benzer şekilde, her bir katman girdileri alır (önceki katman tarafından sağlanır) çıktılar (sonraki katmana girdiler) üretir ve sonraki katmandan geriye doğru akan sinyale göre güncellenen bir dizi ayarlanabilir parametreye sahiptir.

Nöronların, katmanların ve modellerin bize işimiz için yeterince soyutlama sağladığını düşünseniz de, genellikle tek bir katmandan daha büyük ancak tüm modelden daha küçük olan bileşenler hakkında konuşmayı uygun bulduğumuz ortaya çıkıyor. Örneğin, bilgisayarla görmede aşırı popüler olan ResNet-152 mimarisi, yüzlerce katmana sahiptir. Bu katmanlar, *katman gruplarının* tekrar eden desenlerinden oluşur. Böyle bir ağın her seferinde bir katman olarak uygulanması sıkıcı bir hal alabilir. Bu endişe sadece varsayımsal değildir---bu tür tasarım modelleri pratikte yaygındır. Yukarıda bahsedilen ResNet mimarisi, hem tanıma hem de tespit için 2015 ImageNet ve COCO bilgisayarlı görme yarışmalarını kazandı :cite:`He.Zhang.Ren.ea.2016` ve birçok görme görevi için bir ilk tatbik edilen mimari olmaya devam ediyor. Katmanların çeşitli yinelenen desenlerde düzenlendiği benzer mimariler artık doğal dil işleme ve konuşma dahil olmak üzere diğer alanlarda da her yerde mevcuttur.

Bu karmaşık ağları uygulamak için, bir sinir ağı *bloğu* kavramını sunuyoruz. Bir blok, tek bir katmanı, birden çok katmandan oluşan bir bileşeni veya tüm modelin kendisini tanımlayabilir! Blok soyutlamayla çalışmanın bir yararı, bunların genellikle yinelemeli olarak daha büyük yapay nesnelerle birleştirilebilmeleridir (şekile bakınız :numref:`fig_blocks`).

![Çoklu katmanlar bloklara birleştiriliyor](../img/blocks.svg)
:label:`fig_blocks`

İsteğe bağlı keyfi karmaşıklıkta bloklar oluşturmak için kod tanımlayarak şaşırtıcı derecede öz kod yazabilir ve yine de karmaşık sinir ağları uygulayabiliriz.

Yazılım açısından, bir blok *sınıf* ile temsil edilir. Onun herhangi bir alt sınıfı, girdisini çıktıya dönüştüren ve gerekli parametreleri depolayan bir ileriye doğru yöntem tanımlamalıdır. Bazı blokların herhangi bir parametre gerektirmediğini unutmayın! Son olarak, gradyanları hesaplamak için bir blok geriye doğru bir yönteme sahip olmalıdır. Neyse ki, kendi bloğumuzu tanımlarken otomatik türev almanın sağladığı bazı perde arkası sihir sayesinde (:numref:`sec_autograd`da tanıtıldı), sadece parametreler ve ileriye doğru işlev hakkında endişelenmemiz gerekir.

Başlamak için, çok katmanlı algılayıcıları uygulamak için kullandığımız kodları yeniden gözden geçiriyoruz (:numref:`sec_mlp_concise`). Aşağıdaki kod, 256 birim ve ReLU aktivasyonuna sahip tam bağlı bir gizli katmana sahip bir ağ oluşturur ve ardından 10 birimle (etkinleştirme işlevi yok) tam bağlı *çıktı katmanı* gelir.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.random.uniform(size=(2, 20))

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F


x = torch.randn(2,20)
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(x)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

x = tf.random.uniform((2, 20))
net(x)
```

:begin_tab:`mxnet`
This is actually just shorthand for `net.forward(X)`, a slick Python trick achieved via the Block class's `__call__` function.

Bu örnekte, modelimizi bir `nn.Sequential` örneğini oluşturarak, döndürülen nesneyi `net` değişkenine atayarak oluşturduk. Daha sonra, katmanları çalıştırılmaları gereken sırayla eklemek için tekrar tekrar `add` yöntemini çağırıyoruz. Kısaca, `nn.Sequential`, Gluon'da bir blok sunan sınıf olan özel bir `Block` türünü tanımlar. Oluşturucu `Block`ların sıralı bir listesini tutar. `add` yöntemi, basitçe her ardışık `Block`'un listeye eklenmesini kolaylaştırır. Her katmanın, kendisinin `Block`'un alt sınıfı olan `Dense` sınıfının bir örneği olduğuna dikkat edin. `Forward` işlevi de oldukça basittir: Listedeki her bloğu birbirine zincirleyerek her birinin çıktısını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktılarını elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında Block sınıfının `__call__` işlevi aracılığıyla elde edilen ustaca bir Python marifeti olan `net.forward(X)` için kısaltmadır.
:end_tab:

:begin_tab:`pytorch`
Bu örnekte, modelimizi bir `nn.Sequential` örneğini oluşturarak, katmanları bağımsız değişken olarak iletilmeleri gereken sırayla oluşturduk. Kısaca, `nn.Sequential`, PyTorch'ta bir blok sunan özel bir sınıf `Module` türünü tanımlar, ki kurucu `Module`lerin sıralı bir listesini tutar. Tam bağlı iki katmanın her birinin, kendisi de `Module`ün bir alt sınıfı olan `Linear` sınıfının bir örneği olduğuna dikkat edin. `forward` işlevi de oldukça basittir: Listedeki her bloğu birbirine zincirleyerek her birinin çıktısını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktıları elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında Block sınıfının `__call__` işlevi aracılığıyla elde edilen ustaca bir Python marifeti olan `net.forward(X)` için kısaltmadır.
:end_tab:

:begin_tab:`tensorflow`
Bu örnekte, modelimizi bir `keras.models.Sequential` örneğini oluşturarak, katmanları bağımsız değişken olarak iletilmeleri gereken sırayla oluşturduk. Kısaca `Sequential`, Keras'ta bir blok sunan özel bir sınıf `keras.Model`'i tanımlar, ki kurucu `Model`lerin sıralı bir listesini tutar. İki tam bağlı katmanın her birinin, kendisi `Model`'in alt sınıfı olan `Dense` sınıfının bir örneği olduğuna dikkat edin. `forward` işlevi de oldukça basittir: Listedeki her bloğu birbirine zincirleyerek her birinin çıkışını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktılarını elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında Block sınıfının `__call__` işlevi aracılığıyla elde edilen ustaca bir Python marifeti olan `net.call(X)` için kısaltmadır.
:end_tab:

## A Custom Block

Perhaps the easiest way to develop intuition about how a block works is to implement one ourselves. Before we implement our own custom block, we briefly summarize the basic functionality that each block must provide:

1. Ingest input data as arguments to its forward method.
1. Generate an output by having forward return a value. Note that the output may have a different shape from the input. For example, the first fully-connected layer in our model above ingests an input of arbitrary dimension but returns an output of dimension 256.
1. Calculate the gradient of its output with respect to its input, which can be accessed via its backward method. Typically this happens automatically.
1. Store and provide access to those parameters necessary to execute the forward computation.
1. Initialize these parameters as needed.

In the following snippet, we code up a block from scratch corresponding to a multilayer perceptron with one hidden layer with 256 hidden nodes, and a 10-dimensional output layer. Note that the `MLP` class below inherits the class represents a block. We will rely heavily on the parent class's methods, supplying only our own `__init__` and forward methods.

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input `x`
    def forward(self, x):
        return self.out(self.hidden(x))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__()
        self.hidden = nn.Linear(20,256)  # Hidden layer
        self.out = nn.Linear(256,10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input `x`
    def forward(self, x):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(x)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input `x`
    def call(self, x):
        return self.out(self.hidden((x)))
```

To begin, let us focus on the forward method. Note that it takes `x` as input, calculates the hidden representation (`self.hidden(x)`) with the activation function applied, and outputs its logits (`self.out( ... )`). In this MLP implementation, both layers are instance variables. To see why this is reasonable, imagine instantiating two MLPs, `net1` and `net2`, and training them on different data. Naturally, we would expect them to represent two different learned models.

We instantiate the MLP's layers in the `__init__` method (the constructor) and subsequently invoke these layers on each call to the forward method. Note a few key details. First, our customized `__init__` method invokes the parent class's `__init__` method via `super().__init__()` sparing us the pain of restating boilerplate code applicable to most Blocks. We then instantiate our two fully-connected layers, assigning them to `self.hidden` and `self.out`. Note that unless we implement a new operator, we need not worry about backpropagation (the backward method) or parameter initialization. The system will generate these methods automatically. Let us try this out:

```{.python .input}
net = MLP()
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(x)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(x)
```

A key virtue of the block abstraction is its versatility. We can subclass the block class to create layers (such as the fully-connected layer class), entire models (such as the `MLP` above), or various components of intermediate complexity. We exploit this versatility throughout the following chapters, especially when addressing convolutional neural networks.


## The Sequential Block

We can now take a closer look at how the `Sequential` class works. Recall that `Sequential` was designed to daisy-chain other blocks together. To build our own simplified `MySequential`, we just need to define two key methods:
1. A method to append blocks one by one to a list.
2. A forward method to pass an input through the chain of Blocks (in the same order as they were appended).

The following `MySequential` class delivers the same
functionality the default `Sequential` class:

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume it has
        # a unique name. We save it in the member variable _children of the
        # Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize function, the system automatically
        # initializes all members of `_children`
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            x = block(x)
        return x
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # Here, block is an instance of a Module subclass. We save it in
            # the member variable _modules of the Module class, and its type
            # is OrderedDict
            self._modules[block] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            x = block(x)
        return x
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, block is an instance of a tf.keras.layers.Layer subclass
            self.modules.append(block)

    def call(self, x):
        for module in self.modules:
            x = module(x)
        return x
```

:begin_tab:`mxnet`
The `add` method adds a single block to the ordered dictionary `_children`. You might wonder why every Gluon `Block` possesses a `_children` attribute and why we used it rather than just defining a Python list ourselves. In short the chief advantage of `_children` is that during our block's parameter initialization, Gluon knows to look in the `_children`dictionary to find sub-Blocks whose parameters also need to be initialized.
:end_tab:

:begin_tab:`pytorch`
In the `__init__` method, we add every block to the ordered dictionary `_modules` one by one. You might wonder why every `Module` possesses a `_modules` attribute and why we used it rather than just defining a Python list ourselves. In short the chief advantage of `_modules` is that during our block's parameter initialization, the system knows to look in the `_modules` dictionary to find sub-blocks whose parameters also need to be initialized.
:end_tab:

:begin_tab:`tensorflow`
FIXME, don't use `Sequential` to implement `MySequential`.
:end_tab:

When our `MySequential`'s forward method is invoked, each added block is executed in the order in which they were added. We can now reimplement an MLP using our `MySequential` class.

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(x)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(x)
```

Note that this use of `MySequential` is identical to the code we previously wrote for the `Sequential` class (as described in :numref:`sec_mlp_concise`).


## Executing Code in the forward Method

The `Sequential` class makes model construction easy, allowing us to assemble new architectures without having to define our own class. However, not all architectures are simple daisy chains. When greater flexibility is required, we will want to define our own blocks. For example, we might want to execute Python's control flow within the forward method. Moreover we might want to perform arbitrary mathematical operations, not simply relying on predefined neural network layers.

You might have noticed that until now, all of the operations in our networks have acted upon our network's activations and its parameters. Sometimes, however, we might want to incorporate terms that are neither the result of previous layers nor updatable parameters. We call these *constant* parameters. Say for example that we want a layer that calculates the function $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$, where $\mathbf{x}$ is the input, $\mathbf{w}$ is our parameter, and $c$ is some specified constant that is not updated during optimization.

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the get_constant are not
        # iterated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # Use the constant parameters created, as well as the relu and dot
        # functions
        x = npx.relu(np.dot(x, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while np.abs(x).sum() > 1:
            x /= 2
        return x.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # Use the constant parameters created, as well as the relu and dot
        # functions
        x = F.relu(torch.mm(x, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.linear(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while x.norm().item() > 1:
            x /= 2
        return x.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        x = self.flatten(inputs)
        # Use the constant parameters created, as well as the relu and dot
        # functions
        x = tf.nn.relu(tf.matmul(x, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while tf.norm(x) > 1:
            x /= 2
        return tf.reduce_sum(x)
```

In this `FixedHiddenMLP` model, we implement a hidden layer whose weights (`self.rand_weight`) are initialized randomly at instantiation and are thereafter constant. This weight is not a model parameter and thus it is never updated by backpropagation. The network then passes the output of this *fixed* layer through a fully-connected layer.

Note that before returning output, our model did something unusual. We ran a `while` loop, testing on the condition it's norm is larger than 1, and dividing our output vector by $2$ until it satisfied the condition. Finally, we returned the sum of the entries in `x`. To our knowledge, no standard neural network performs this operation. Note that this particular operation may not be useful in any real world task. Our point is only to show you how to integrate arbitrary code into the flow of your neural network computations.

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(x)
```

We can mix and match various ways of assembling blocks together. In the following example, we nest blocks in some creative ways.

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())

chimera.initialize()
chimera(x)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, x):
        return self.linear(self.net(x))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(x)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(x)
```

## Compilation

:begin_tab:`mxnet, tensorflow` 
The avid reader might start to worry about the efficiency of some of these operations. After all, we have lots of dictionary lookups, code execution, and lots of other Pythonic things taking place in what is supposed to be a high performance deep learning library. The problems of Python's [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. In the context of deep learning, we worry that our extremely fast GPU(s) might have to wait until a puny CPU runs Python code before it gets another job to run. The best way to speed up Python is by avoiding it altogether.
:end_tab:

:begin_tab:`mxnet`
One way that Gluon does this by allowing for hybridization (:numref:`sec_hybridize`). Here, the Python interpreter executes a Block the first time it is invoked. The Gluon runtime records what is happening and the next time around it short-circuits calls to Python. This can accelerate things considerably in some cases but care needs to be taken when control flow (as above) leads down different branches on different passes through the net. We recommend that the interested reader check out the hybridization section (:numref:`sec_hybridize`) to learn about compilation after finishing the current chapter.
:end_tab:

## Summary

* Layers are blocks.
* Many layers can comprise a block.
* Many blocks can comprise a block.
* A block can contain code.
* Blocks take care of lots of housekeeping, including parameter initialization and backpropagation.
* Sequential concatenations of layers and blocks are handled by the `Sequential` Block.


## Exercises

1. What kinds of problems will occur if you change `MySequential` to store blocks in a Python list.
1. Implement a block that takes two blocks as an argument, say `net1` and `net2` and returns the concatenated output of both networks in the forward pass (this is also called a parallel block).
1. Assume that you want to concatenate multiple instances of the same network. Implement a factory function that generates multiple instances of the same block and build a larger network from it.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
