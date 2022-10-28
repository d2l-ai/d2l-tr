# Parametre Yönetimi

Bir mimari seçip hiper parametrelerimizi belirledikten sonra, hedefimiz yitim işlevimizi en aza indiren parametre değerlerini bulmak olduğu eğitim döngüsüne geçiyoruz. Eğitimden sonra, gelecekteki tahminlerde bulunmak için bu parametrelere ihtiyacımız olacak. Ek olarak, bazen parametreleri başka bir bağlamda yeniden kullanmak, modelimizi başka bir yazılımda yürütülebilmesi için diske kaydetmek veya bilimsel anlayış kazanma umuduyla incelemek için ayıklamak isteyeceğiz.

Çoğu zaman, ağır işlerin üstesinden gelmek için derin öğrenme çerçevelerine dayanarak, parametrelerin nasıl beyan edildiğine ve değiştirildiğine dair işin esas ayrıntıları görmezden gelebileceğiz. Bununla birlikte, standart katmanlara sahip yığılmış mimarilerden uzaklaştığımızda, bazen parametreleri bildirme ve onların üstünde oynama yabani otlarına girmemizi gerekecektir. Bu bölümde aşağıdakileri ele alıyoruz:

* Hata ayıklama, teşhis ve görselleştirmeler için parametrelere erişim.
* Parametre ilkletme.
* Parametreleri farklı model bileşenleri arasında paylaşma.

(**Tek bir gizli katmana sahip bir MLP'ye odaklanarak başlıyoruz.**)

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Varsayılan ilkleme yöntemini kullan

X = np.random.uniform(size=(2, 4))
net(X)  # İleri hesaplama
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

## [**Parametre Erişimi**]

Zaten bildiğiniz modellerden parametrelere nasıl erişileceğiyle başlayalım. Bir model `Sequential` sınıfı aracılığıyla tanımlandığında, ilk olarak herhangi bir katmana, bir listeymiş gibi modelde üzerinden indeksleme ile erişebiliriz. Her katmanın parametreleri, özelliklerinde uygun bir şekilde bulunur. Tam bağlı ikinci katmanın parametrelerini aşağıdaki gibi irdeleyebiliriz.

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

Çıktı bize birkaç önemli şey gösteriyor. İlk olarak, bu tam bağlı katman, sırasıyla o katmanın ağırlıklarına ve ek girdilerine karşılık gelen iki parametre içerir. Her ikisi de tek hassas basamaklı kayan virgüllü sayı (float32) olarak saklanır. Parametrelerin adlarının, yüzlerce katman içeren bir ağda bile, her katmanın parametrelerini benzersiz şekilde tanımlamamıza izin verdiğini unutmayın.


### [**Hedeflenen Parametreler**]

Her parametrenin, parametre sınıfının bir örneği olarak temsil edildiğine dikkat edin. Parametrelerle yararlı herhangi bir şey yapmak için önce altta yatan sayısal değerlere erişmemiz gerekir. Bunu yapmanın birkaç yolu var. Bazıları daha basitken diğerleri daha geneldir. Aşağıdaki kod, bir parametre sınıfı örneği döndüren ikinci sinir ağı katmanından ek girdiyi dışarı çıkarır ve dahası bu parametrenin değerine erişim sağlar.

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
Parametreler; değer, gradyanlar ve ek bilgiler içeren karmaşık nesnelerdir. Bu nedenle değerleri açıkça talep etmemiz gerekiyor.

Değere ek olarak, her parametre gradyana erişmemize de izin verir. Henüz bu ağ için geri yaymayı çağırmadığımız için, ağ ilk durumundadır.
:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch
net[2].weight.grad == None
```

### [**Bütün Parametrelere Bir Kerede Erişim**]

Tüm parametrelerde işlem yapmamız gerektiğinde, bunlara tek tek erişmek bıktırıcı olabilir. Her bir alt bloğun parametrelerini dışarı çıkarmayı tüm ağaç boyunca tekrarlamamız gerekeceğinden, daha karmaşık bloklarla (örneğin, iç içe geçmiş bloklarla) çalıştığımızda durum özellikle kullanışsızca büyüyebilir. Aşağıda, ilk tam bağlı katmanın parametrelerine erişmeye karşılık tüm katmanlara erişmeyi gösteriyoruz.

```{.python .input}
print(net[0].collect_params())
print(net.collect_params())
```

```{.python .input}
#@tab pytorch
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

```{.python .input}
#@tab tensorflow
print(net.layers[1].weights)
print(net.get_weights())
```

Bu bize ağın parametrelerine erişmenin aşağıdaki gibi başka bir yolunu sağlar:

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

### [**İçiçe Bloklardan Parametreleri Toplama**]

Birden çok bloğu iç içe yerleştirirsek, parametre adlandırma kurallarının nasıl çalıştığını görelim. Bunun için önce blok üreten bir fonksiyon tanımlıyoruz (tabiri caizse bir blok fabrikası) ve sonra bunları daha büyük bloklar içinde birleştiriyoruz.

```{.python .input}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        # Burada iç içe konuyor
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(X)
```

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Burada iç içe konuyor
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
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
        # Burada iç içe konuyor
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)
```

[**Ağı tasarladığımıza göre, şimdi nasıl düzenlendiğini görelim.**]

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

Katmanlar hiyerarşik olarak iç içe olduğundan, iç içe geçmiş listeler aracılığıyla dizinlenmiş gibi bunlara da erişebiliriz. Örneğin, birinci ana bloğa, içindeki ikinci alt bloğa ve bunun içindeki ilk katmanın ek girdisine aşağıdaki şekilde erişebiliriz.

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

Artık parametrelere nasıl erişeceğimizi bildiğimize göre, onları nasıl doğru şekilde ilkleteceğimize bakalım. Uygun ilkleme ihtiyacını :numref:`sec_numerical_stability` içinde tartıştık. Derin öğrenme çerçevesi katmanlarına varsayılan rastgele ilklemeler sağlar. Bununla birlikte, ağırlıklarımızı öteki farklı protokollere göre ilkletmek istiyoruz. Çerçeve, en sık kullanılan protokolleri sağlar ve ayrıca özelleştirilmiş bir ilkletici oluşturmaya izin verir.

:begin_tab:`mxnet`
Varsayılan olarak, MXNet ağırlık parametrelerini $U[-0.07, 0.07]$ tekdüze dağılımdan rasgele çekerek ilkler ve ek girdileri sıfırlar. MXNet'in `init` modülü önceden ayarlı çeşitli ilkleme yöntemleri sağlar.
:end_tab:

:begin_tab:`pytorch`
Varsayılan olarak PyTorch, girdi ve çıktı boyutuna göre hesaplanan bir aralıktan çekerek ağırlık ve ek girdi matrislerini tekdüze olarak başlatır. PyTorch'un `nn.init` modülü önceden ayarlı çeşitli ilkleme yöntemleri sağlar.
:end_tab:

:begin_tab:`tensorflow`
Varsayılan olarak Keras, girdi ve çıktı boyutuna göre hesaplanan bir aralıktan çekerek ağırlık matrislerini tekdüze olarak ilkletir ve ek girdi parametrelerinin tümü sıfır olarak atanır. TensorFlow, hem kök modülde hem de `keras.initializers` modülünde çeşitli ilkleme yöntemleri sağlar.
:end_tab:

### [**Yerleşik İlkletme**]

Yerleşik ilkleticileri çağırarak ilkleyelim. Aşağıdaki kod, tüm ağırlık parametrelerini standart sapması 0.01 olan Gauss rastgele değişkenler olarak ilkletirken ek girdi parametreleri de 0 olarak atanır.

```{.python .input}
# Burada 'force_reinit', daha önce ilklenmiş olsalar bile parametrelerin 
# yeniden ilklenmesini sağlar
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

net(X)
net.weights[0], net.weights[1]
```

Ayrıca tüm parametreleri belirli bir sabit değere ilkleyebiliriz (örneğin, 1 gibi).

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

net(X)
net.weights[0], net.weights[1]
```

[**Ayrıca belirli bloklar için farklı ilkleyiciler uygulayabiliriz.**] Örneğin, aşağıda ilk katmanı Xavier ilkleyicisi ile ilkliyoruz ve ikinci katmanı sabit 42 değeri ile ilkliyoruz.

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
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

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
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### [**Özelleştirilmiş İlkleme**]

Bazen, ihtiyaç duyduğumuz ilkletme yöntemleri derin öğrenme çerçevesi tarafından sağlanmayabilir. Aşağıdaki örnekte, herhangi bir ağırlık parametresi $w$ için aşağıdaki garip dağılımı kullanarak bir ilkleyici tanımlıyoruz:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ olasılık değeri } \frac{1}{4} \\
            0    & \text{ olasılık değeri } \frac{1}{2} \\
        U(-10, -5) & \text{ olasılık değeri } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Burada `Initializer` sınıfının bir alt sınıfını tanımlıyoruz. Genellikle, yalnızca bir tensör bağımsız değişkeni (`data`) alan ve ona istenen ilkletilmiş değerleri atayan `_init_weight` işlevini uygulamamız gerekir.
:end_tab:

:begin_tab:`pytorch`
Yine, `net`'e uygulamak için bir `my_init` işlevi uyguluyoruz.
:end_tab:

:begin_tab:`tensorflow`
Burada `Initializer`'ın bir alt sınıfını tanımlıyoruz ve şekil ve veri türüne göre istenen bir tensörü döndüren `__call__` işlevini uyguluyoruz.
:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) 
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor 

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

Her zaman parametreleri doğrudan ayarlama seçeneğimiz olduğunu unutmayın.

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

:begin_tab:`mxnet`
İleri düzey kullanıcılar için bir hatırlatma: Parametreleri bir `autograd` (otomatik türev) kapsamında ayarlamak istiyorsanız, otomatik türev alma mekanizmalarının karıştırılmasını önlemek için `set_data`'yı kullanmanız gerekir.
:end_tab:

## [**Bağlı Parametreler**]

Genellikle, parametreleri birden çok katmanda paylaşmak isteriz. Bunu biraz daha zekice bir şekilde nasıl yapacağımızı görelim. Aşağıda yoğun (dense) bir katman ayırıyoruz ve ardından onun parametrelerini de özellikle başka bir katmanınkileri ayarlamak için kullanıyoruz.

```{.python .input}
net = nn.Sequential()
# Parametrelerine atıfta bulunabilmemiz için paylaşılan katmana bir ad 
# vermemiz gerekiyor
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Parametrelerin aynı olup olmadığını kontrol edin
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Aynı değere sahip olmak yerine aslında aynı nesne olduklarından emin olun
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
#@tab pytorch
# Parametrelerine atıfta bulunabilmemiz için paylaşılan katmana bir ad 
# vermemiz gerekiyor
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Parametrelerin aynı olup olmadığını kontrol edin
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Aynı değere sahip olmak yerine aslında aynı nesne olduklarından emin olun
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
#@tab tensorflow
# tf.keras biraz farklı davranır. Yinelenen katmanı otomatik olarak kaldırır.
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Parametrelerin farklı olup olmadığını kontrol edin
print(len(net.layers) == 3)
```

:begin_tab:`mxnet,pytorch`
Bu örnek, ikinci ve üçüncü katmanın parametrelerinin birbirine bağlı olduğunu göstermektedir. Sadece eşit değiller, tamamen aynı tensörle temsil ediliyorlar. Bu yüzden parametrelerden birini değiştirirsek diğeri de değişir. Merak edebilirsiniz, parametreler bağlı olduğunda gradyanlara ne olur? Model parametreleri gradyanlar içerdiğinden, ikinci ve üçüncü gizli katmanların gradyanları geri yayma sırasında birbiriyle toplanır.
:end_tab:

## Özet

* Model parametrelerine erişmek, ilklemek ve onları bağlamak için birkaç farklı yol var.
* Özelleştirilmiş ilkleme kullanabiliriz.


## Alıştırmalar

1. :numref:`sec_model_construction` içinde tanımlanan `FancyMLP` modelini kullanınız ve çeşitli katmanların parametrelerine erişiniz.
1. Farklı ilkleyicileri keşfetmek için ilkleme modülü dökümanına bakınız.
1. Paylaşılan bir parametre katmanı içeren bir MLP oluşturunuz ve onu eğitiniz. Eğitim sürecinde, her katmanın model parametrelerini ve gradyanlarını gözlemleyiniz.
1. Parametreleri paylaşmak neden iyi bir fikirdir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/269)
:end_tab:
