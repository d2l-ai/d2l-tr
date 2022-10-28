# Ertelenmiş İlkleme
:label:`sec_deferred_init`

Şimdiye kadar, ağlarımızı kurarken özensiz davranışlarımızdan yakayı kurtarmış görünebiliriz. Özellikle, çalışmaması gerektiği düşündüğümüz, aşağıdaki sezgisel olmayan şeyleri yaptık:

* Ağ mimarilerini girdi boyutlarını belirlemeden tanımladık.
* Bir önceki katmanın çıktı boyutunu belirtmeden katmanlar ekledik.
* Hatta modellerimizin kaç parametre içermesi gerektiğini belirlemek için yeterli bilgi sağlamadan önce bu parametreleri "ilklettik".

Kodumuzun çalışmasına şaşırabilirsiniz. Sonuçta, derin öğrenme çerçevesinin bir ağın girdi boyutluluğunun ne olacağını söylemesi mümkün değildir. Buradaki püf nokta, çerçevenin *ilklemeyi ertelemesidir*, verileri modelden ilk kez geçirene kadar bekleyerek koşma anında her katmanın boyutunu çıkarır.

Daha sonra, evrişimli sinir ağları ile çalışırken, bu teknik daha da uygun hale gelecektir, çünkü girdi boyutluluğu (yani, bir imgenin çözünürlüğü) sonraki her katmanın boyutluluğunu etkileyecektir. Bu nedenle, kodun yazılması sırasında boyutluluğun ne olduğunu bilmeye gerek kalmadan parametreleri ayarlama becerisi, modellerimizi belirleme ve daha sonra değiştirme görevini büyük ölçüde basitleştirebilir. Ardından, ilkleme mekaniğinin daha derinine ineceğiz.

## Bir Ağ Örneği Yaratma

Bir MLP örneği yaratarak başlayalım.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

Bu noktada, ağ muhtemelen girdi katmanının ağırlıklarının boyutlarını bilemez çünkü girdi boyutu bilinmemektedir. Sonuç olarak, çerçeve henüz herhangi bir parametreyi ilklemedi. Aşağıdaki parametrelere erişmeyi deneyerek bunu test ediyoruz.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Parametre nesneleri varken, her katmanın girdi boyutunun -1 olarak listelendiğini unutmayın. MXNet, parametre boyutunun bilinmediğini belirtmek için -1 özel değerini kullanır. Bu noktada, `net[0].weight.data()` erişim girişimleri, parametrelere erişilmeden önce ağın ilkletilmesi gerektiğini belirten bir çalışma zamanı hatasını tetikleyecektir. Şimdi parametreleri `initialize` işlevi ile ilklemeye çalıştığımızda ne olacağını görelim.
:end_tab:

:begin_tab:`tensorflow`
Her katman nesnesinin var olduğunu ancak ağırlıkların boş (atanmamış) olduğunu unutmayın. `net.get_weights()`'ın kullanılması, ağırlıklar henüz ilkletilmediği için bir hata oluşturacaktır.
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Gördüğümüz gibi hiçbir şey değişmedi. Girdi boyutları bilinmediğinde, ilkleme çağrıları parametreleri gerçekten ilkletmez. Bunun yerine, bu çağrı, parametreleri ilklemek istediğimizi (ve isteğe bağlı olarak, hangi dağılım olduğuna göre) MXNet'e kaydeder.
:end_tab:

Sonrasında çerçevenin en sonunda parametreleri ilklemesi için ağ üzerinden veriyi geçirelim.

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

Girdi boyutluluğunu, 20'yi, öğrenir öğrenmez, 20 değerini yerine koyarak, çerçeve ilk katmanın ağırlık matrisinin şeklini belirleyebilir. İlk katman şeklini tanımladıktan sonra çerçeve boyutsallığı ikinci katmana ilerletir, ta ki tüm şekiller bilinene kadar hesaplama çizgesi üzerinden devam eder. Bu durumda, yalnızca ilk katmanın ertelenmiş ilkleme gerektirdiğini, ancak çerçevenin sırayla ilklettiğini unutmayın. Tüm parametre şekilleri bilindikten sonra, çerçeve sonunda parametreleri ilkleyebilir.

## Özet

* Ertelenmiş ilkleme, çerçevenin parametre şekillerini otomatik olarak çıkarmasına izin vermede, mimarileri değiştirmeyi kolaylaştırmada ve genel bir hata kaynağını ortadan kaldırmada kullanışlı olabilir.
* Çerçevenin en sonunda parametreleri ilklemesi sağlamak için verileri modelden geçirebiliriz.

## Alıştırmalar

1. İlk katmanda girdi boyutlarını belirtirseniz, ancak sonraki katmanlarda girmezseniz ne olur? Anında ilklemeyi elde eder misiniz?
1. Uyumsuz boyutlar belirtirseniz ne olur?
1. Boyutluluğu değişken bir girdiye sahipseniz ne yapmanız gerekir? İpucu: Bağlı parametrelere bakınız.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/281)
:end_tab:
