# Doğrusal Regresyonunun Kısa Uygulaması
:label:`sec_linear_concise`

Son birkaç yıldır derin öğrenmeye olan geniş ve yoğun ilgi, gradyan tabanlı öğrenme algoritmalarını uygulamanın tekrarlayan iş yükünü otomatikleştirmek için şirketler, akademisyenler ve amatör geliştiricilere çeşitli olgun açık kaynak çerçeveleri geliştirmeleri için ilham verdi. :numref:`sec_linear_scratch` içinde, biz sadece (i) veri depolama ve doğrusal cebir için tensörlere; ve (ii) gradyanları hesaplamak için otomatik türev almaya güvendik. Pratikte, veri yineleyiciler, kayıp işlevleri, optimize ediciler ve sinir ağı katmanları çok yaygın olduğu için, modern kütüphaneler bu bileşenleri bizim için de uygular.

Bu bölümde, derin öğrenme çerçevelerinin (**üst düzey API'lerini kullanarak :numref:`sec_linear_scratch` içindeki doğrusal regresyon modelini kısaca nasıl uygulayacağınızı göstereceğiz**).

## Veri Kümesini Oluşturma

Başlamak için, şuradaki aynı veri kümesini oluşturacağız: :numref:`sec_linear_scratch`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## Veri Kümesini Okuma

Kendi yineleyicimizi döndürmek yerine, [**verileri okumak için bir çerçevedeki mevcut API'yi çağırabiliriz**]. `features` (öznitelikleri tutan değişken) ve `labels`'i (etiketleri tutan değişken) bağımsız değişken olarak iletiriz ve bir veri yineleyici nesnesi başlatırken `batch_size` (grup boyutu) belirtiriz. Ayrıca, mantıksal veri tipi (boolean) değeri `is_train`, veri yineleyici nesnesinin her bir dönemdeki (epoch) verileri karıştırmasını isteyip istemediğimizi gösterir (veri kümesinden geçer).

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Bir Gluon veri yineleyici oluşturun."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Bir PyTorch veri yineleyici oluşturun."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Bir TensorFlow veri yineleyici oluşturun."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Şimdi, `data_iter`i, `data_iter` işlevini :numref:`sec_linear_scratch` içinde çağırdığımız şekilde kullanabiliriz. Çalıştığını doğrulamak için, örneklerin ilk minigrubunu okuyabilir ve yazdırabiliriz. :numref:`sec_linear_scratch` içindeki ile karşılaştırıldığında, burada bir Python yineleyici oluşturmak için `iter` kullanıyoruz ve yineleyiciden ilk öğeyi elde etmek için `next`'i kullanıyoruz.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## Modeli Tanımlama

Doğrusal regresyonu :numref:`sec_linear_scratch` içinde sıfırdan uyguladığımızda, model parametrelerimizi açık bir şekilde tanımladık ve temel doğrusal cebir işlemlerini kullanarak çıktı üretmek için hesaplamalarımızı kodladık. Bunu nasıl yapacağınızı *bilmelisiniz*. Ancak modelleriniz daha karmaşık hale geldiğinde ve bunu neredeyse her gün yapmanız gerektiğinde, bu yardım için memnun olacaksınız. Durum, kendi blogunuzu sıfırdan kodlamaya benzer. Bunu bir veya iki kez yapmak ödüllendirici ve öğreticidir, ancak bir bloga her ihtiyaç duyduğunuzda tekerleği yeniden icat etmek için bir ay harcarsanız kötü bir web geliştiricisi olursunuz.

Standart işlemler için, uygulamaya (kodlamaya) odaklanmak yerine özellikle modeli oluşturmak için kullanılan katmanlara odaklanmamızı sağlayan [**bir çerçevenin önceden tanımlanmış katmanlarını**] kullanabiliriz. Önce, `Sequential` (ardışık, sıralı) sınıfının bir örneğini ifade edecek `net` (ağ) model değişkenini tanımlayacağız. `Sequential` sınıfı, birbirine zincirlenecek birkaç katman için bir kapsayıcı (container) tanımlar. Girdi verileri verildiğinde, `Sequential` bir örnek, bunu birinci katmandan geçirir, ardından onun çıktısını ikinci katmanın girdisi olarak geçirir ve böyle devam eder. Aşağıdaki örnekte, modelimiz yalnızca bir katmandan oluşuyor, bu nedenle gerçekten `Sequential` örneğe ihtiyacımız yok. Ancak, gelecekteki modellerimizin neredeyse tamamı birden fazla katman içereceği için, sizi en standart iş akışına alıştırmak için yine de kullanacağız.

Tek katmanlı bir ağın mimarisini şurada gösterildiği gibi hatırlayın: :numref:`fig_single_neuron`. Katmanın *tamamen bağlı* olduğu söylenir, çünkü girdilerinin her biri, bir matris-vektör çarpımı yoluyla çıktılarının her birine bağlanır.

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
# `nn` sinir ağları için kısaltmadır
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` sinir ağları için kısaltmadır
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` TensorFlow'un üst-seviye API'sidir
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## Model Parametrelerini İlkleme

`net`'i kullanmadan önce, doğrusal regresyon modelindeki ağırlıklar ve ek girdi gibi (**model parametrelerini ilkletmemiz**) gerekir. Derin öğrenme çerçeveleri genellikle parametreleri ilklemek için önceden tanımlanmış bir yola sahiptir. Burada, her ağırlık parametresinin, ortalama 0 ve standart sapma 0.01 ile normal bir dağılımdan rastgele örneklenmesi gerektiğini belirtiyoruz. Ek girdi parametresi sıfır olarak ilklenecektir.

:begin_tab:`mxnet`
MXNet'ten `initializer` modülünü içe aktaracağız. Bu modül, model parametresi ilkletme için çeşitli yöntemler sağlar. Gluon, `init`'i `initializer` paketine erişmek için bir kısayol (kısaltma) olarak kullanılabilir hale getirir. Ağırlığın nasıl ilkleneceğini sadece `init.Normal(sigma=0.01)`'i çağırarak belirtiyoruz. Ek girdi parametreleri sıfır olarak ilklenecektir.
:end_tab:

:begin_tab:`pytorch`
`nn.Linear` oluştururken girdi ve çıktı boyutlarını belirttik, şimdi, onların ilk değerlerini belirtmek için parametrelere doğrudan erişebiliriz. İlk olarak  ağdaki ilk katmanı `net[0]` ile buluruz ve ardından parametrelere erişmek için `weight.data` ve `bias.data` yöntemlerini kullanırız. Daha sonra, parametre değerlerinin üzerine yazmak için `normal_` ve `fill_` değiştirme yöntemlerini kullanırız.
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
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
Yukarıdaki kod basit görünebilir, ancak burada tuhaf bir şeylerin olduğunu fark etmelisiniz. Gluon, girdinin kaç boyuta sahip olacağını henüz bilmese de, bir ağ için parametreleri ilkletebiliyoruz! Örneğimizdeki gibi 2 de olabilir veya 2000 de olabilir. Gluon bunun yanına kalmamıza izin veriyor çünkü sahnenin arkasında, ilk değerleri atama aslında *ertelendi*. Gerçek ilkleme, yalnızca verileri ağ üzerinden ilk kez geçirmeye çalıştığımızda gerçekleşecektir. Unutmayın ki, parametreler henüz ilkletilmediği için bunlara erişemeyiz veya onları değiştiremeyiz.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
Yukarıdaki kod basit görünebilir, ancak burada tuhaf bir şeylerin olduğunu fark etmelisiniz. Keras, girdinin kaç boyuta sahip olacağını henüz bilmese de, bir ağ için parametreleri ilkletebiliyoruz! Örneğimizdeki gibi 2 de olabilir veya 2000 de olabilir. Keras bunun yanına kalmamıza izin veriyor çünkü sahnenin arkasında, ilk değerleri atama aslında *ertelendi*. Gerçek ilkleme, yalnızca verileri ağ üzerinden ilk kez geçirmeye çalıştığımızda gerçekleşecektir. Unutmayın ki, parametreler henüz ilkletilmediği için bunlara erişemeyiz veya onları değiştiremeyiz.
:end_tab:

## Kayıp Fonksiyonunu Tanımlama

:begin_tab:`mxnet`
Gluon'da `loss` modülü çeşitli kayıp fonksiyonlarını tanımlar. Bu örnekte, Gluon uygulamasının kare kaybını (`L2Loss`) kullanacağız.
:end_tab:

:begin_tab:`pytorch`
[**`MSELoss` sınıfı, ortalama hata karesini hesaplar (:eqref:`eq_mse` içindeki $1/2$ çarpanı olmadan)**]. Varsayılan olarak, örneklerdeki ortalama kaybı döndürür.
:end_tab:

:begin_tab:`tensorflow`
[**`MeanSquaredError` sınıfı, ortalama hata karesini hesaplar (:eqref:`eq_mse` içindeki $1/2$ çarpanı olmadan)**]. Varsayılan olarak, örneklerdeki ortalama kaybı döndürür.
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

## Optimizasyon Algoritmasını Tanımlama

:begin_tab:`mxnet`
Minigrup rasgele gradyan inişi, sinir ağlarını optimize etmek için standart bir araçtır ve bu nedenle Gluon, bunu `Trainer` sınıfı aracılığıyla bu algoritmadaki bir dizi varyasyonla birlikte destekler. `Trainer`'den örnek yarattığımızda, optimize edilecek parametreleri (`net.collect_params()` aracılığıyla `net` modelimizden elde edilebilir), kullanmak istediğimiz optimizasyon algoritması (`sgd`) ve optimizasyon algoritmamızın gerektirdiği hiper parametreleri bir sözlük ile (dictionary veri yapısı) belirleyeceğiz. Minigrup rasgele gradyan inişi, burada 0.03 olarak ayarlanan `learning_rate` (öğrenme oranı) değerini ayarlamamızı gerektirir.
:end_tab:

:begin_tab:`pytorch`
Minigrup rasgele gradyan inişi, sinir ağlarını optimize etmek için standart bir araçtır ve bu nedenle PyTorch, bunu `Trainer` sınıfı aracılığıyla bu algoritmadaki bir dizi varyasyonla birlikte destekler. (**`SGD`'den örnek yarattığımızda**), optimize edilecek parametreleri (`net.collect_params()` aracılığıyla `net` modelimizden elde edilebilir) ve optimizasyon algoritmamızın gerektirdiği hiper parametreleri bir sözlük ile (dictionary veri yapısı) belirteceğiz.
Minigrup rasgele gradyan inişi, burada 0.03 olarak ayarlanan `lr` (öğrenme oranı) değerini ayarlamamızı gerektirir.
:end_tab:

:begin_tab:`tensorflow`
Minigrup rasgele gradyan inişi, sinir ağlarını optimize etmek için standart bir araçtır ve bu nedenle Keras, bunu, `optimizers` modülündeki bu algoritmanın bir dizi varyasyonunun yanında destekler. Minigrup rasgele gradyan inişi, burada 0.03 olarak ayarlanan `learning_rate` (öğrenme oranı) değerini ayarlamamızı gerektirir.
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

## Eğitim

Modelimizi derin öğrenme çerçevesinin yüksek seviyeli API'leri aracılığıyla ifade etmenin nispeten birkaç satır kod gerektirdiğini fark etmiş olabilirsiniz. Parametreleri ayrı ayrı tahsis etmemiz, kayıp fonksiyonumuzu tanımlamamız veya minigrup rasgele gradyan inişini uygulamamız gerekmedi. Çok daha karmaşık modellerle çalışmaya başladığımızda, üst düzey API'lerin avantajları önemli ölçüde artacaktır. Bununla birlikte, tüm temel parçaları bir kez yerine getirdiğimizde, [**eğitim döngüsünün kendisi, her şeyi sıfırdan uygularken yaptığımıza çarpıcı bir şekilde benzer**].

Hafızanızı yenilemek için: Bazı dönemler (epoch) için, veri kümesinin (`train_data`) üzerinden eksiksiz bir geçiş yapacağız ve yinelemeli olarak bir minigrup girdiyi ve ilgili referans gerçek değer etiketleri alacağız. Her minigrup için aşağıdaki ritüeli uyguluyoruz:

* `net(X)`'i çağırarak tahminler oluşturun ve `l` kaybını (ileriye doğru yayma) hesaplayın.
* Geri yaymayı çalıştırarak gradyanları hesaplayın.
* Optimize edicimizi çağırarak model parametrelerini güncelleyin.

İyi bir önlem olarak, her dönemden sonra kaybı hesaplıyor ve ilerlemeyi izlemek için yazdırıyoruz.

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'donem {epoch + 1}, kayip {l.mean().asnumpy():f}')
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
    print(f'donem {epoch + 1}, kayip {l:f}')
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
    print(f'donem {epoch + 1}, kayip {l:f}')
```

Aşağıda, [**sonlu veri üzerinde eğitimle öğrenilen model parametrelerini ve veri kümemizi oluşturan gerçek parametreleri karşılaştırıyoruz**]. Parametrelere erişmek için önce ihtiyacımız olan katmana `net`'ten erişiyoruz ve sonra bu katmanın ağırlıklarına ve ek girdilerine erişiyoruz. Sıfırdan uygulamamızda olduğu gibi, tahmin edilen parametrelerimizin gerçek referans değerlere yakın olduğuna dikkat edin.

```{.python .input}
w = net[0].weight.data()
print(f'w tahmin hatasi: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'b tahmin hatasi: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('w tahmin hatasi:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('b tahmin hatasi:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('w tahmin hatasi', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('b tahmin hatasi', true_b - b)
```

## Özet

:begin_tab:`mxnet`
* Gluon kullanarak modelleri çok daha kısaca uygulayabiliyoruz.
* Gluon'da, `data` modülü veri işleme için araçlar sağlar, `nn` modülü çok sayıda sinir ağı katmanını tanımlar ve `loss` modülü birçok yaygın kayıp işlevini tanımlar.
* MXNet'in `initializer` modülü, model parametresi ilkleme için çeşitli yöntemler sağlar.
* Boyutluluk ve depolama otomatik olarak çıkarılır, ancak parametrelere ilklemeden önce erişmeye çalışmamaya dikkat edin.
:end_tab:

:begin_tab:`pytorch`
* PyTorch'un üst düzey API'lerini kullanarak modelleri çok daha kısa bir şekilde uygulayabiliriz.
* PyTorch'ta, `data` modülü veri işleme için araçlar sağlar, `nn` modülü çok sayıda sinir ağı katmanını ve genel kayıp işlevlerini tanımlar.
* Değerlerini `_` ile biten yöntemlerle değiştirerek parametreleri ilkleyebiliriz.
:end_tab:

:begin_tab:`tensorflow`
* TensorFlow'un üst düzey API'lerini kullanarak modelleri çok daha kısa bir şekilde uygulayabiliriz.
* TensorFlow'da, `data` modülü veri işleme için araçlar sağlar, `keras` modülü çok sayıda sinir ağı katmanını ve genel kayıp işlevlerini tanımlar.
* TensorFlow'un `initializers` modülü, model parametresi ilkleme için çeşitli yöntemler sağlar.
* Boyut ve depolama otomatik olarak çıkarılır (ancak parametrelere ilklemeden önce erişmeye çalışmamaya dikkat edin).
:end_tab:

## Alıştırmalar

:begin_tab:`mxnet`
1. `l = loss(output, y)` yerine `l = loss(output, y).mean()` koyarsak, kodun aynı şekilde davranması için `trainer.step(batch_size)`'i `trainer.step(1)` olarak değiştirmemiz gerekir. Neden?
1. `gluon.loss` ve `init` modüllerinde hangi kayıp işlevlerinin ve ilkleme yöntemlerinin sağlandığını görmek için MXNet belgelerini inceleyin. Kaybı Huber kaybıyla yer değiştirin.
1. `dense.weight`'in gradyanına nasıl erişirsiniz?

[Tartışmalar](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. Eğer `nn.MSELoss()`ı `nn.MSELoss(reduction='sum')` ile değiştirirsek, kodun aynı şekilde davranması için öğrenme oranını nasıl değiştirebiliriz? Neden?
1. Hangi kayıp işlevlerinin ve ilkleme yöntemlerinin sağlandığını görmek için PyTorch belgelerini inceleyin. Kaybı Huber kaybıyla yer değiştirin.
1. `net[0].weight`'in gradyanına nasıl erişirsiniz?

[Tartışmalar](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. Hangi kayıp işlevlerinin ve ilkleme yöntemlerinin sağlandığını görmek için TensorFlow belgelerini inceleyin. Kaybı Huber kaybıyla yer değiştirin.

[Tartışmalar](https://discuss.d2l.ai/t/204)
:end_tab:
