# Minigrup Rasgele Gradyan İnişi
:label:`sec_minibatch_sgd`

Şimdiye kadar gradyan tabanlı öğrenme yaklaşımında iki uç noktaya rastladık: :numref:`sec_gd` gradyanları hesaplamak ve parametreleri güncellemek için her seferinde bir geçiş yaparak tüm veri kümesini kullanır. Tersine :numref:`sec_sgd` ilerleme kaydetmek için bir seferde bir gözlem işler. Her birinin kendi dezavantajları vardır. Gradyan inişi, özellikle veriler çok benzer olduğunda  *veri verimli* değildir. Rasgele gradyan inişi, işlemciler ve GPU'lar vektörleştirmenin tam gücünden yararlanamayacağından, bilhassa *hesaplama açısından verimli* değildir. Bu, mutlu bir ortam olabileceğini gösteriyor ve aslında, şimdiye kadar tartıştığımız örneklerde bunu kullanıyorduk. 

## Vektörleştirme ve Önbellekler

Minigruplar kullanma kararının merkezinde hesaplama verimliliği vardır. Bu, en kolay şekilde birden çok GPU ve birden çok sunucuya paralelleştirme düşünüldüğünde anlaşılır. Bu durumda, her GPU'ya en az bir imge göndermemiz gerekiyor. 16 sunucu ve sunucu başına 8 GPU ile zaten 128'lik minigrup boyutuna ulaşıyoruz. 

Tek GPU'lar ve hatta CPU'lar söz konusu olduğunda işler biraz daha inceliktir. Bu cihazların birden çok bellek türü, genellikle birden fazla işlem birimi türü ve aralarında farklı bant genişliği kısıtlamaları vardır. Örneğin, bir CPU'da az sayıda yazmaç (register) ve daha sonra L1, L2 ve hatta bazı durumlarda L3 önbellek (farklı işlemci çekirdekleri arasında paylaşılır) vardır. Bu önbellekler boyut ve gecikme süresini artırmaktadır (ve aynı zamanda bant genişliğini azaltmaktadır). Diyebiliriz ki, aslında işlemci ana bellek arayüzünün sağlayabildiğinden çok daha fazla işlem gerçekleştirebilir. 

* 16 çekirdeğe ve AVX-512 vektörleştirmesine sahip 2 GHz CPU, saniyede $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ bayta kadar işleyebilir. GPU'ların kapasitesi bu sayının 100 katını kolayca aşar. Öte yandan, orta düzey bir sunucu işlemcisi 100 GB/s'den fazla bant genişliğine sahip olmayabilir, yani işlemcinin beslenmesini sağlamak için gerekenlerin onda birinden azı olabilir. İşleri daha da kötüleştirmek adına, tüm bellek erişimi eşit oluşturulmaz: Öncelikle, bellek arabirimleri genellikle 64 bittir ya da daha geniştir (örneğin, GPU'larda 384 bite kadar), bu nedenle tek bir bayt okumak çok daha geniş bir erişim maliyetini doğurur.
* İlk erişim için önemli bir maliyet varken sıralı erişim nispeten ucuzdur (buna genellikle çoğuşma denir). Birden fazla soket, yonga ve diğer yapılara sahip olduğumuzda önbelleğe alma gibi akılda tutulması gereken çok daha fazla şey vardır. Bunun ayrıntılı bir tartışması, bu bölümün kapsamı dışındadır. Daha ayrıntılı bir tartışma için bu [Wikipedia makalesi](https://en.wikipedia.org/wiki/Cache_hierarchy)'ne bakın.

Bu kısıtlamaları hafifletmenin yolu, işlemciye veri sağlamak için yeterince hızlı olan CPU önbellekleri hiyerarşisini kullanmaktır. Bu, derin öğrenmede toplu işlemenin arkasındaki itici güçtür. Konuları basit tutmak için, matris ile matris çarpımını düşünün, $\mathbf{A} = \mathbf{B}\mathbf{C}$ diyelim. $\mathbf{A}$'yı hesaplamak için bir dizi seçeneğimiz var. Örneğin aşağıdakileri deneyebiliriz: 

1. $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$'ü hesaplayabiliriz, yani nokta çarpımları vasıtasıyla eleman yönlü hesaplayabiliriz.
1. $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$'yi hesaplayabiliriz, yani, her seferinde bir sütun hesaplayabiliriz. Aynı şekilde $\mathbf{A}$'yı bir seferde bir satır, $\mathbf{A}_{i,:}$, hesaplamak da olabilir.
1. Sadece $\mathbf{A} = \mathbf{B} \mathbf{C}$'yi hesaplayabiliriz.
1. $\mathbf{B}$ ve $\mathbf{C}$'yi daha küçük blok matrislerine parçalayıp  $\mathbf{A}$'yı her seferde bir blok hesaplayabiliriz.

İlk seçeneği izlersek, $\mathbf{A}_{ij}$ öğesini hesaplamak istediğimiz her seferinde bir satır ve bir sütun vektörünü CPU'ya kopyalamalıyız. Daha da kötüsü, matris elemanlarının sıralı olarak hizalanması nedeniyle, bellekten okurken iki vektörden biri için birçok ayrık konuma erişmemiz gerekiyor. İkinci seçenek çok daha elverişlidir. İçinde $B$ üzerinden geçiş yapmaya devam ederken sütun vektörünü $\mathbf{C}_{:,j}$'ü CPU önbelleğinde tutabiliyoruz. Bu, daha hızlı erişim ile bellek bant genişliği gereksinimini yarıya indirir. Tabii ki, seçenek 3 en çok arzu edilendir. Ne yazık ki, çoğu matris önbelleğe tamamen sığmayabilir (sonuçta tartıştığımız şey budur). Bununla birlikte, seçenek 4 pratik olarak kullanışlı bir alternatif sunar: Matrisin bloklarını önbelleğe taşıyabilir ve yerel olarak çoğaltabiliriz. Optimize edilmiş kütüphaneler bunu bizim için halledeceklerdir. Bu operasyonların pratikte ne kadar verimli olduğuna bir göz atalım. 

Hesaplama verimliliğinin ötesinde, Python ve derin öğrenme çerçevesinin kendisi tarafından getirilen yük de düşündürücüdür. Python yorumlayıcısı her komutu çalıştırdığımızda MXNet motoruna, hesaplamalı çizgeye eklemesi ve zamanlama sırasında onunla ilgilenmesi gereken bir komut gönderdiğini hatırlayın. Bu tür yükler oldukça bezdirici olabilir. Kısacası, mümkün olduğunca vektörleştirme (ve matrisler) kullanılması şiddetle tavsiye edilir.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

timer = d2l.Timer()
A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np

timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

timer = d2l.Timer()
A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

Eleman yönlü değer atama, $\mathbf{A}$'ya değer atamak için sırasıyla $\mathbf{B}$ ve $\mathbf{C}$'nin tüm satır ve sütunlarını yineler.

```{.python .input}
# Her keresinde bir eleman olacak şekilde A = BC hesapla
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Her keresinde bir eleman olacak şekilde A = BC hesapla
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Her keresinde bir eleman olacak şekilde A = BC hesapla
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

Daha hızlı bir strateji sütun yönlü atama gerçekleştirmektir.

```{.python .input}
# Her keresinde bir sutün olacak şekilde A = BC hesapla
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Her keresinde bir sutün olacak şekilde A = BC hesapla
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

Son olarak, en etkili yol, tüm işlemi bir blokta gerçekleştirmektir. İşlemlerin göreceli hızının ne olduğunu görelim.

```{.python .input}
# Bir seferde A = BC hesapla
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Çarpma ve toplama ayrı işlemler olsun (uygulamada kaynaşıktır)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Bir seferde A = BC hesapla
timer.start()
A = torch.mm(B, C)
timer.stop()

# Çarpma ve toplama ayrı işlemler olsun (uygulamada kaynaşıktır)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# Çarpma ve toplama ayrı işlemler olsun (uygulamada kaynaşıktır)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minigruplar
:label:`sec_minibatches` 

Geçmişte, parametreleri güncellemek için tek gözlemler yerine verilerin *minigruplar*ı okuyacağımızı düşünmeden kabul ettik. Şimdi bunun için kısa bir gerekçe veriyoruz. Tek gözlemlerin işlenmesi, oldukça pahalıdır ve altta yatan derin öğrenme çerçevesi adına önemli bir yük oluşturan birçok tek matris ile vektör (hatta vektör ile vektör) çarpımı gerçekleştirmemizi gerektirir. Bu, hem verilere uygulandığında bir ağın değerlendirilmesi (genellikle çıkarım olarak adlandırılır) hem de parametreleri güncellemek için gradyanları hesaplarken geçerlidir. Yani, bu $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ uyguladığımız her zaman geçerlidir:

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Bu işlemin *hesaplama* verimliliğini bir seferde bir minigrup gözlem uygulayarak artırabiliriz. Yani, $\mathbf{g}_t$'yi tek bir gözlem yerine minigrup üzerinden gradyan ile değiştiriyoruz:

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Bunun $\mathbf{g}_t$'nin istatistiksel özelliklerine ne yaptığını görelim: Hem $\mathbf{x}_t$ hem de minigrup $\mathcal{B}_t$'nin tüm elemanları eğitim kümesinden tekdüze rastgele bir şekilde çekildiğinden, gradyanın beklentisi değişmeden kalır. Öte yandan varyans önemli ölçüde azaltılır. Minigrup gradyanı, ortalaması alınan $b := |\mathcal{B}_t|$ bağımsız gradyanlardan oluştuğu için, standart sapması $b^{-\frac{1}{2}}$ çarpanı kadar azalır. Bu, tek başına, iyi bir şeydir, çünkü güncellemelerin tam gradyan ile daha güvenilir bir şekilde hizalandığı anlamına gelir. 

Safça bu, büyük bir minigrup $\mathcal{B}_t$ seçmenin evrensel olarak arzu edileceğini gösterir. Ne yazık ki, bir noktadan sonra, standart sapmadaki ek azalma, hesaplama maliyetindeki doğrusal artışa kıyasla minimumdur. Pratikte, bir GPU belleğine uyarken iyi hesaplama verimliliği sunacak kadar büyük bir minigrup seçiyoruz. Tasarrufları göstermek için koda biraz göz atalım. İçerisinde aynı matris matris çarpımını gerçekleştiriyoruz, ancak bu sefer bir seferde 64 sütunlu "minigruplar"a parçaladık.

```{.python .input}
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

Gördüğümüz gibi, minigrup üzerindeki hesaplama aslında tam matris kadar etkilidir. Dikkat edilmesi gereken unsur şudur: :numref:`sec_batch_norm` içinde bir minigrup içindeki varyans miktarına büyük ölçüde bağımlı olan bir düzenlilik türü kullandık. İkincisini arttırdıkça, varyans azalır ve bununla birlikte toplu normalleşme nedeniyle gürültü aşılamanın faydası olur. Uygun şartların nasıl yeniden ölçekleneceği ve hesaplanacağı ile ilgili ayrıntılar için bkz. :cite:`Ioffe.2017`. 

## Veri Kümesini Okuma

Minigrupların verilerden nasıl verimli bir şekilde üretildiğine bir göz atalım. Aşağıda bu optimizasyon algoritmaları karşılaştırmak için [farklı uçakların kanat gürültüsü](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)nü test etmek için NASA tarafından geliştirilmiş bir veri kümesi kullanıyoruz. Kolaylık olsun diye sadece ilk $1,500$ örneği kullanıyoruz. Veriler ön işleme için beyazlatılır, yani ortalamaları kaldırır ve varyansı koordinat başına $1$'e yeniden ölçeklendiririz.

```{.python .input}
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Sıfırdan Uygulama

:numref:`sec_linear_scratch` içindeki minigrup rasgele gradyan inişi uygulamasını hatırlayın. Aşağıda biraz daha genel bir uygulama sağlıyoruz. Kolaylık sağlamak için, bu bölümde daha sonra tanıtılan diğer optimizasyon algoritmalarıyla aynı çağrı imzasına sahiptir. Özellikle, `states` durum girdisini ekliyoruz ve hiper parametreyi `hyperparams` sözlüğüne yerleştiriyoruz. Buna ek olarak, eğitim işlevindeki her minigrup örneğinin kaybını ortalayacağız, böylece optimizasyon algoritmasındaki gradyanın iş boyutuna bölünmesi gerekmez.

```{.python .input}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

Daha sonra, bu bölümün ilerleyen bölümlerinde tanıtılan diğer optimizasyon algoritmalarının kullanımını kolaylaştırmak için genel bir eğitim işlevi uyguluyoruz. Doğrusal regresyon modelini ilkletir ve modeli minigrup rasgele gradyan inişi ve daha sonra tanıtılan diğer algoritmalarla eğitmek için kullanılabilir.

```{.python .input}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # İlkleme
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Eğitim
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # İlkleme
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Eğitim
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # İlkleme
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Eğitim
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Toplu iş gradyan inişi için optimizasyonun nasıl ilerlediğini görelim. Bu, minigrup boyutunu 1500'e (yani toplam örnek sayısına) ayarlayarak elde edilebilir. Sonuç olarak model parametreleri dönem başına yalnızca bir kez güncellenir. Çok az ilerleme var. Aslında 6 adımdan sonra ilerleme durur.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Toplu iş boyutu 1'e eşit olduğunda, optimizasyon için rasgele gradyan inişi kullanıyoruz. Uygulamanın basitliği için sabit (küçük olsa da) bir öğrenme oranı seçtik. Rasgele gradyan inişinde, örnek her işlendiğinde model parametreleri güncelleştirilir. Bizim durumumuzda bu, dönem başına 1500 güncellemedir. Gördüğümüz gibi, amaç fonksiyonunun değerindeki düşüş bir dönemden sonra yavaşlar. Her iki yöntem de bir dönem içinde 1500 örnek işlemiş olsa da, rasgele gradyan inişi bizim deneyde gradyan inişinden daha fazla zaman tüketir. Bunun nedeni, rasgele gradyan inişinin parametreleri daha sık güncellemesi ve tek tek gözlemleri birer birer işlemenin daha az verimli olmasıdır.
```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Son olarak, toplu iş boyutu 100'e eşit olduğunda, optimizasyon için minigrup rasgele gradyan inişi kullanıyoruz. Dönem başına gereken süre, rasgele gradyan inişi için gereken süreden ve toplu gradyan inişi için gereken süreden daha kısadır.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

Topl iş boyutu 10'a düşürüldüğünde, her toplu iş yükünün yürütülmesi daha az verimli olduğundan, her dönemin süresi artar.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Şimdi önceki dört deney için zamanı ve kaybı karşılaştırabiliriz. Görüldüğü gibi, rasgele gradyan inişi, işlenen örneklerin sayısı açısından GD'den daha hızlı yakınsasa da, gradyanın her örnek için hesaplanması o kadar verimli olmadığından, GD'ye görece aynı kayba ulaşmak için daha fazla zaman kullanır. Minigrup rasgele gradyan inişi yakınsama hızını ve hesaplama verimliliğini değiştirebilir. 10 'luk minigrup boyutu, rasgele gradyan inişinden daha etkilidir; 100'luk minigrup boyutu çalışma zamanı açısından GD'den daha iyi performans gösterir.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Özlü Uygulama

Gluon'da, optimizasyon algoritmalarını çağırmak için `Trainer` sınıfını kullanabiliriz. Bu, genel bir eğitim işlevini uygulamak için kullanılır. Bunu mevcut bölüm boyunca kullanacağız.

```{.python .input}
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # İlkleme
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # İlkleme
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # İlkleme
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `MeanSquaredError` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

Son deneyi tekrarlarken Gluon'u kullanmak aynı davranışı gösterir.

```{.python .input}
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## Özet

* Vektorleştirme, derin öğrenme çerçevesinden kaynaklanan azaltılmış ek yükü ve CPU ile GPU'larda daha iyi bellek yerelliği ve önbelleğe alma nedeniyle kodu daha verimli hale getirir.
* Rasgele gradyan inişinden kaynaklanan istatistiksel verimlilik ile aynı anda büyük veri yığınlarının işlenmesinden kaynaklanan hesaplama verimliliği arasında bir ödün verme vardır.
* Minigrup rasgele gradyan inişi her iki dünyanın en iyisini sunar: Hesaplama ve istatistiksel verimlilik.
* Minigrup rasgele gradyan inişinde, eğitim verilerinin rastgele bir yer değiştirmesi ile elde edilen veri yığınlarını işleriz (yani, her gözlem, rastgele sırada da olsa, her dönemde sadece bir kez işlenir).
* Eğitim sırasında öğrenme oranlarının sönümlenmesi tavsiye edilir.
* Genel olarak, minigrup rasgele gradyan inişi, saat süresi açısından ölçüldüğünde, daha küçük bir riske yakınsama için rasgele gradyan inişi ve gradyan inişinden daha hızlıdır.

## Alıştırmalar

1. Toplu iş boyutunu ve öğrenme oranını değiştirin ve amaç fonksiyonun değerini ve her dönemde tüketilen süreye ilişkin düşüş oranını gözlemleyin.
1. MXNet belgelerini okuyun ve `Trainer` sınıfı `set_learning_rate` işlevini kullanarak minigrup rasgele gradyan inişinin öğrenme hızını her dönemden sonraki önceki değerinin 1/10'una düşürün.
1. Minigrup rasgele gradyan inişini, eğitim kümesinden *değiştirmeli örneklemler* kullanan bir sürümle karşılaştırın. Ne olur?
1. Kötü bir cin, size haber vermeden veri kümenizi çoğaltıyor (yani, her gözlem iki kez gerçekleşir ve veri kümeniz orijinal boyutunun iki katına çıkar, ancak size söylenmedi). Rasgele gradyan inişinin, minigrup rasgele gradyan inişinin ve gradyan inişinin davranışı nasıl değişir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1069)
:end_tab:
