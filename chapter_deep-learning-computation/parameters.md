# Parametre Yönetimi

Bir mimari seçip hiperparametrelerimizi belirledikten sonra, hedefimiz amaç işlevimizi en aza indiren parametre değerlerini bulmak olduğu eğitim döngüsüne geçiyoruz. Eğitimden sonra, gelecekteki tahminlerde bulunmak için bu parametrelere ihtiyacımız olacak. Ek olarak, bazen parametreleri başka bir bağlamda yeniden kullanmak, modelimizi başka bir yazılımda yürütülebilmesi için diske kaydetmek veya bilimsel anlayış kazanma umuduyla incelemek için dışarı çıkarmak isteyeceğiz.

Çoğu zaman, ağır işlerin üstesinden gelmek için çerçeveye dayanarak, parametrelerin nasıl beyan edildiğine ve değiştirildiğine dair işin esas ayrıntıları görmezden gelebileceğiz. Bununla birlikte, standart katmanlara sahip yığılmış mimarilerden uzaklaştığımızda, bazen parametreleri bildirme ve onların üstünde oynama yabani otlarına girmemizi gerekecektir. Bu bölümde aşağıdakileri ele alıyoruz:

* Hata ayıklama, teşhis ve görselleştirmeler için parametrelere erişim.
* Parametre ilkletme.
* Parametreleri farklı model bileşenleri arasında paylaşma.

Tek bir gizli katmana sahip bir MLP'ye odaklanarak başlıyoruz.

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

x = np.random.uniform(size=(2, 4))
net(x)  # Forward computation
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
x = torch.randn(2, 4)
net(x)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

x = tf.random.uniform((2, 4))
net(x)
```

## Parametre Erişimi

Zaten bildiğiniz modellerden parametrelere nasıl erişileceğiyle başlayalım. Bir model `Sequential` sınıfı aracılığıyla tanımlandığında, ilk olarak herhangi bir katmana, bir listeymiş gibi modelde üzerinden indisleme ile erişebiliriz. Her katmanın parametreleri, niteliğinde uygun bir şekilde bulunur. Tam bağlı ikinci katmanın parametrelerini aşağıdaki gibi irdeleyebiliriz.

```{.python .input}
print(net[1].params)
```

```{.python .input}
#@tab pytorch
print(net[2].state_dict())
```

```{.python .input}
#@tab tensorflow
print(net.layers[2].weights)
```

Çıktı bize birkaç önemli şey gösteriyor. İlk olarak, bu tam bağlı katman, sırasıyla o katmanın ağırlıklarına ve ek girdilerine karşılık gelen iki parametre içerir. Her ikisi de tek hassas basamaklı kayan virgüllü sayı olarak saklanır. Parametrelerin adlarının, yüzlerce katman içeren bir ağda bile, her katmanın parametrelerini *benzersiz şekilde* tanımlamamıza izin verdiğini unutmayın.


### Hedeflenen Parametreler

Her parametrenin, parametre sınıfının bir örneği olarak temsil edildiğine dikkat edin. Parametrelerle yararlı herhangi bir şey yapmak için önce altta yatan sayısal değerlere erişmemiz gerekir. Bunu yapmanın birkaç yolu var. Bazıları daha basitken diğerleri daha geneldir. Aşağıdaki kod, bir parametre sınıfı örneği döndüren ikinci sinir ağı katmanından ek girdiyi dışarı çıkarır ve bu parametrenin değerine daha ileri erişim sağlar.

```{.python .input}
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

```{.python .input}
#@tab pytorch
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

```{.python .input}
#@tab tensorflow
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))
```

:begin_tab:`mxnet,pytorch`
Parametreler; veri, gradyanlar ve ek bilgiler içeren karmaşık nesnelerdir. Bu nedenle verileri açıkça talep etmemiz gerekiyor.

`data`'ya ek olarak, her `Parameter` gradyana erişmek için bir `grad` yöntemi sağlar. Henüz bu ağ için geri yaymayı çağırmadığımız için, ağ ilk durumundadır.
:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch
net[2].weight.grad == None
```

### Bütün Parametrelere Bir Kerede Erişim

Tüm parametrelerde işlem yapmamız gerektiğinde, bunlara tek tek erişmek bıktırıcı olabilir. Her bir alt bloğun parametrelerini dışarı çıkarmayı tüm ağaç boyunca tekrarlamamız gerekeceğinden, daha karmaşık bloklarla (örneğin, iç içe geçmiş bloklarla) çalıştığımızda durum özellikle kullanışsızca büyüyebilir. Aşağıda, ilk tam bağlı katmanın parametrelerine erişmeye karşılık tüm katmanlara erişmeyi gösteriyoruz.

```{.python .input}
print(net[0].collect_params())
print(net.collect_params())
```

```{.python .input}
#@tab pytorch
print(net[1].state_dict())
print(net.state_dict())
```

```{.python .input}
#@tab tensorflow
print(net.layers[1].weights)
print(net.get_weights())
```

Bu bize ağın parametrelerine erişmenin başka bir yolunu sağlar:

```{.python .input}
net.collect_params()['dense1_bias'].data()
```

```{.python .input}
#@tab pytorch
net.state_dict()['2.bias'].data
```

```{.python .input}
#@tab tensorflow
net.get_weights()[1]
```

### İçiçe Bloklardan Parametreleri Toplama

Birden çok bloğu iç içe yerleştirirsek, parametre adlandırma kurallarının nasıl çalıştığını görelim. Bunun için önce blok üreten bir fonksiyon tanımlıyoruz (tabiri caizse bir blok fabrikası) ve sonra bunları daha büyük bloklar içinde birleştiriyoruz.

```{.python .input}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(x)
```

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(x)
```

```{.python .input}
#@tab tensorflow
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(x)
```

Ağı tasarladığımıza göre, şimdi nasıl düzenlendiğini görelim.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

```{.python .input}
#@tab pytorch
print(rgnet)
```

```{.python .input}
#@tab tensorflow
print(rgnet.summary())
```

Katmanlar hiyerarşik olarak iç içe olduğundan, iç içe geçmiş listeler aracılığıyla dizinlenmiş gibi bunlara da erişebiliriz. Örneğin, birinci ana bloğa, içindeki ikinci alt bloğa ve bunun içindeki ilk katmanın ek girdisine aşağıdaki şekilde erişebiliriz:

```{.python .input}
rgnet[0][1][0].bias.data()
```

```{.python .input}
#@tab pytorch
rgnet[0][1][0].bias.data
```

```{.python .input}
#@tab tensorflow
rgnet.layers[0].layers[1].layers[1].weights[1]
```

## Parametre İlkleme

Artık parametrelere nasıl erişeceğimizi bildiğimize göre, onları nasıl doğru şekilde ilkleteceğimze bakalım. İlkletme ihtiyacını :numref:`sec_numerical_stability`'de tartıştık. Çerçeve, katmanlarına varsayılan rastgele ilkletmeler sağlar. Bununla birlikte, ağırlıklarımızı öteki farklı protokollere göre ilkletmek istiyoruz. Çerçeve, en sık kullanılan protokolleri sağlar ve ayrıca bir tüketici ilkletici oluşturmaya izin verir.

:begin_tab:`mxnet`
Varsayılan olarak, MXNet ağırlık matrislerini $U[-0.07, 0.07]$'den çekerek eşit şekilde başlatır, MXNet'in `init` modülü önceden ayarlı çeşitli ilkleme yöntemleri sağlar.
:end_tab:

:begin_tab:`pytorch`
Varsayılan olarak PyTorch, girdi ve çıktı boyutuna göre hesaplanan bir aralıktan çekerek ağırlık ve ek girdi matrislerini tekdüze olarak başlatır. PyTorch'un `nn.init` modülü önceden ayarlı çeşitli ilkleme yöntemleri sağlar.
:end_tab:

:begin_tab:`tensorflow`
Varsayılan olarak Keras, girdi ve çıktı boyutuna göre hesaplanan bir aralıktan çekerek ağırlık matrislerini tekdüze olarak başlatır ve ek girdi parametrelerinin tümü $0$ olarak atanır. TensorFlow, hem kök modülde hem de `keras.initializers` modülünde çeşitli ilkleme yöntemleri sağlar.
:end_tab:

### Yerleşik İlkletme

Yerleşik ilketicileri çağırarak başlayalım. Aşağıdaki kod, tüm ağırlık parametrelerini standart sapması $.01$ olan Gauss rastgele değişkenler olarak başlatırken, ek girdi parametreleri 0 olarak ayarlanmıştır.

```{.python .input}
# force_reinit ensures that variables are freshly initialized
# even if they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(x)
net.weights[0], net.weights[1]
```

Ayrıca tüm parametreleri belirli bir sabit değere ilkleyebiliriz (örneğin, $1$ gibi).

```{.python .input}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(x)
net.weights[0], net.weights[1]
```

Ayrıca belirli bloklar için farklı ilkleyiciler uygulayabiliriz. Örneğin, aşağıda ilk katmanı `Xavier` ilkleyicisi ile ilkliyoruz ve ikinci katmanı sabit bir 42 değeri ile ilkliyoruz.

```{.python .input}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
#@tab pytorch
def xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        torch.nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(1)),
])

net(x)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### Custom Initialization

Sometimes, the initialization methods we need are not provided by the framework. In the example below, we define an initializer for the following strange distribution:

$$
\begin{aligned}
    w \sim \begin{cases}
        U[5, 10] & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U[-10, -5] & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Here we define a subclass of `Initializer`. Usually, we only need to implement the `_init_weight` function which takes a tensor argument (`data`) and assigns to it the desired initialized values.
:end_tab:

:begin_tab:`pytorch`
Again, we implement a `my_init` function to apply to `net`.
:end_tab:

:begin_tab:`tensorflow`
Here we define a subclass of `Initializer` and implement the `__call__` function that return a desired tensor given the shape and data type.
:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[0:2]
```

```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, dtype=dtype)

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(x)
print(net.layers[1].weights[0])
```

Note that we always have the option of setting parameters directly.

:begin_tab:`mxnet`
A note for advanced users: if you want to adjust parameters within an `autograd` scope, you need to use `set_data` to avoid confusing the automatic differentiation mechanics.
:end_tab:

```{.python .input}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
#@tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Tied Parameters

Often, we want to share parameters across multiple layers. Later we will see that when learning word embeddings, it might be sensible to use the same parameters both for encoding and decoding words. We discussed one such case when we introduced :numref:`sec_model_construction`. Let us see how to do this a bit more elegantly. In the following we allocate a dense layer and then use its parameters specifically to set those of another layer.

```{.python .input}
net = nn.Sequential()
# We need to give the shared layer a name such that we can reference its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = np.random.uniform(size=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
#@tab pytorch
# We need to give the shared layer a name such that we can reference its
# parameters
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(x)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
#@tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(x)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

This example shows that the parameters of the second and third layer are tied. They are not just equal, they are represented by the same exact tensor. Thus, if we change one of the parameters, the other one changes, too. You might wonder, *when parameters are tied what happens to the gradients?* Since the model parameters contain gradients, the gradients of the second hidden layer and the third hidden layer are added together during backpropagation.

## Summary

* We have several ways to access, initialize, and tie model parameters.
* We can use custom initialization.


## Exercises

1. Use the FancyMLP defined in :numref:`sec_model_construction` and access the parameters of the various layers.
1. Look at the initialization module document to explore different initializers.
1. Construct a multilayer perceptron containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
1. Why is sharing parameters a good idea?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:
