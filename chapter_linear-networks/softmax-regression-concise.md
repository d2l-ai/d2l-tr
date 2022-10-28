# Softmaks Regresyonunun Kısa Uygulaması
:label:`sec_softmax_concise`

:numref:`sec_linear_concise` içindeki derin öğrenme çerçevelerinin (**yüksek seviyeli API'leri**) (**doğrusal regresyon uygulamasını çok daha kolay hale getirdi**), (~~burada~~) onu sınıflandırma modellerini uygulamada (**benzer şekilde**) (veya muhtemelen daha fazla) uygun bulacağız. Fashion-MNIST veri kümesine bağlı kalalım ve iş boyutunu :numref:`sec_softmax_scratch` gibi 256'da tutalım.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Model Parametrelerini İlkleme

:numref:`sec_softmax` içinde bahsedildiği gibi, [**softmaks regresyonunun çıktı katmanı tam bağlı bir katmandır**]. Bu nedenle, modelimizi uygulamak için, `Sequential`'a 10 çıktılı tam bağlı bir katman eklememiz yeterlidir. Yine burada, `Sequential` gerçekten gerekli değildir, ancak derin modelleri uygularken her yerde bulunacağından bu alışkanlığı oluşturalım. Yine, ağırlıkları sıfır ortalama ve 0.01 standart sapma ile rastgele ilkliyoruz.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch, girdileri dolaylı olarak yeniden şekillendirmez. Bu yüzden, 
# ağımızdaki doğrusal katmandan önceki girdileri yeniden şekillendirmek için 
# düzleştirilmiş katmanı tanımlarız.
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## Softmaks Uygulamasına Yeniden Bakış
:label:`subsec_softmax-implementation-revisited`

Önceki örneğimiz :numref:`sec_softmax_scratch` içinde, modelimizin çıktısını hesapladık ve sonra bu çıktıyı çapraz entropi kaybıyla çalıştırdık. Matematiksel olarak bu, yapılacak son derece makul bir şeydir. Bununla birlikte, hesaplama açısından, üs alma, sayısal kararlılık sorunlarının bir kaynağı olabilir.

Softmaks fonksiyonunun $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$'yi hesapladığını hatırlayın; burada $\hat y_j$  tahmin edilen olasılık dağılımı $\hat {\mathbf y}$'nin $j.$ öğesidir ve $o_j$, $\mathbf{o}$ logitlerinin $j.$ öğesidir. $o_k$ değerlerinden bazıları çok büyükse (yani çok pozitifse), o zaman $\exp(o_k)$ belirli veri türleri için sahip olabileceğimiz en büyük sayıdan daha büyük olabilir (yani *taşar*). Bu, paydayı (ve/veya payı) `inf` (sonsuz) yapar ve o zaman $\hat y_j$ için 0, `inf` veya `nan` (sayı değil) ile karşılaşırız. Bu durumlarda çapraz entropi için iyi tanımlanmış bir dönüş değeri elde edemeyiz.

Bunu aşmanın bir yolu, softmaks hesaplamasına geçmeden önce ilk olarak $\max(o_k)$'yı tüm $o_k$'dan çıkarmaktır. Burada her bir $o_k$'nın sabit faktörle kaydırılmasının softmaksın dönüş değerini değiştirmediğini görebilirsiniz. 

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$

Çıkarma ve normalleştirme adımından sonra, bazı $o_j - \max(o_k)$ büyük negatif değerlere sahip olabilir ve bu nedenle karşılık gelen $\exp(o_j - \max(o_k))$ sıfıra yakın değerler alacaktır. Bunlar, sonlu kesinlik (yani, *küçümenlik*) nedeniyle sıfıra yuvarlanabilir, $\hat y_j$ sıfır yapar ve $\log(\hat y_j)$ için bize `-inf` verir. Geri yaymada yolun birkaç adım aşağısında, kendimizi korkutucu `nan` sonuçlarıyla karşı karşıya bulabiliriz.

Neyse ki, üstel fonksiyonları hesaplasak bile, nihayetinde onların loglarını (çapraz entropi kaybını hesaplarken) almayı planladığımız gerçeğiyle kurtulduk. Bu iki softmaks ve çapraz entropi operatörünü bir araya getirerek, aksi takdirde geri yayma sırasında başımıza bela olabilecek sayısal kararlılık sorunlarından kaçabiliriz. Aşağıdaki denklemde gösterildiği gibi, $\exp(o_j - \max(o_k))$'yi hesaplamaktan kaçınırız ve bunun yerine $\log(\exp(\cdot))$ içini iptal ederek doğrudan $o_j - \max(o_k)$ kullanabiliriz:

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$

Modelimizin çıktı olasılıklarını değerlendirmek istememiz durumunda, geleneksel softmaks işlevini el altında tutmak isteyeceğiz. Ancak softmaks olasılıklarını yeni kayıp fonksiyonumuza geçirmek yerine, akıllılık yapıp ["LogSumExp numarası"](https://en.wikipedia.org/wiki/LogSumExp) kullanarak [**logitleri geçireceğiz, bütün softmaks ve logaritma değerlerini çapraz entropi kaybında hesaplayacağız**].

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## Optimizasyon Algoritması

Burada, optimizasyon algoritması olarak 0.1 öğrenme oranıyla (**minigrup rasgele gradyan inişini kullanıyoruz**). Bunun doğrusal regresyon örneğinde uyguladığımızla aynı olduğuna ve optimize edicilerin genel uygulanabilirliğini gösterdiğine dikkat edin.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## Eğitim

Ardından modeli eğitirken (~~önceden~~)  :numref:`sec_softmax_scratch` içinde tanımlanan [**eğitim işlevini çağırıyoruz**].

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

Daha önce olduğu gibi, bu algoritma, bu sefer öncekinden daha az kod satırı ile, iyi bir doğruluk sağlayan bir çözüme yakınsıyor.

## Özet

* Yüksek seviyeli API'leri kullanarak softmaks regresyonunu çok daha öz uygulayabiliriz.
* Hesaplama bakış açısından, softmaks regresyonunun uygulanmasının karmaşıklıkları vardır. Pek çok durumda, bir derin öğrenme çerçevesinin sayısal kararlılığı sağlamak için bu çok iyi bilinen hilelerin ötesinde ek önlemler aldığını ve bizi pratikte tüm modellerimizi sıfırdan kodlamaya çalıştığımızda karşılaşacağımız daha da fazla tuzaktan kurtardığını unutmayın.

## Alıştırmalar

1. Sonuçların ne olduğunu görmek için grup boyutu, dönem sayısı ve öğrenme oranı gibi hiper parametreleri ayarlamayı deneyin.
1. Eğitim için dönem sayısını artırın. Test doğruluğu neden bir süre sonra düşüyor olabilir? Bunu nasıl düzeltebiliriz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/260)
:end_tab:
