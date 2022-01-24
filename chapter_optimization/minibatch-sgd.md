# Minibatch Stokastik Degrade İniş
:label:`sec_minibatch_sgd`

Şimdiye kadar degrade tabanlı learning: :numref:`sec_gd` yaklaşımında iki uç noktaya rastladık, degradeleri hesaplamak ve parametreleri güncellemek için her seferinde bir geçiş yapmak için tam veri kümesini kullanır. Tersine :numref:`sec_sgd` ilerleme kaydetmek için bir seferde bir gözlem işler. Her birinin kendi dezavantajları vardır. Degrade Descent, veriler çok benzer olduğunda özellikle *veri verimliliği* değildir. Stokastik Gradyan Descent, işlemciler ve GPU'lar vektörleştirmenin tam gücünden yararlanamayacağından, bilhase*hesaplama açısından verimli değildir. Bu, mutlu bir ortam olabileceğini gösteriyor ve aslında, şimdiye kadar tartıştığımız örneklerde bunu kullanıyorduk. 

## Vektorizasyon ve Önbellekler

Minibatches kullanma kararının merkezinde hesaplama verimliliği vardır. Bu, birden çok GPU ve birden çok sunucuya paralelleştirme düşünüldüğünde en kolay şekilde anlaşılır. Bu durumda, her GPU'ya en az bir görüntü göndermemiz gerekiyor. Sunucu başına 8 GPU ve 16 sunucu ile zaten 128 minibatch boyutuna ulaşıyoruz. 

Tek GPU'lar ve hatta CPU'lar söz konusu olduğunda işler biraz daha inceliktir. Bu aygıtların birden çok bellek türü, genellikle birden fazla işlem birimi türü ve aralarında farklı bant genişliği kısıtlamaları vardır. Örneğin, bir CPU az sayıda kayıt ve daha sonra L1, L2 ve hatta bazı durumlarda L3 önbellek (farklı işlemci çekirdekleri arasında paylaşılır) vardır. Bu önbellekler boyut ve gecikme süresini artırmaktadır (ve aynı zamanda bant genişliğini azaltmaktadır). Diyelim ki, işlemci ana bellek arayüzünün sağlayabildiğinden çok daha fazla işlem gerçekleştirebilir. 

* 16 çekirdeğe ve AVX-512 vektorizasyonuna sahip 2 GHz CPU, saniyede $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ bayta kadar işleyebilir. GPU'ların kapasitesi bu sayıyı 100 faktörle kolayca aşar. Öte yandan, orta düzey bir sunucu işlemcisi 100 GB/s'den fazla bant genişliğine sahip olmayabilir, yani işlemcinin beslenmesini sağlamak için gerekenlerin onda birinden azı olabilir. İşleri daha da kötüleştirmek için, tüm bellek erişimi eşit oluşturulmaz: Birincisi, bellek arabirimleri genellikle 64 bit genişliğinde veya daha geniştir (örneğin, 384 bit'e kadar GPU'larda), bu nedenle tek bir bayt okumak çok daha geniş bir erişim maliyetini doğurur.
* İlk erişim için önemli bir yük varken sıralı erişim nispeten ucuzdur (buna genellikle bir patlama okuma denir). Birden fazla soket, yonga ve diğer yapılara sahip olduğumuzda önbelleğe alma gibi akılda tutulması gereken çok daha fazla şey vardır. Bunun ayrıntılı bir tartışması, bu bölümün kapsamı dışındadır. Daha ayrıntılı bir tartışma için bu [Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy)'e bakın.

Bu kısıtlamaları hafifletmenin yolu, işlemciye veri sağlamak için yeterince hızlı olan CPU önbellekleri hiyerarşisini kullanmaktır. Bu, derin öğrenmede toplu işlemenin arkasında* itici güçtür. Konuları basit tutmak için, matris matris çarpımını düşünün, $\mathbf{A} = \mathbf{B}\mathbf{C}$ diyelim. $\mathbf{A}$ hesaplamak için bir dizi seçeneğimiz var. Örneğin aşağıdakileri deneyebiliriz: 

1. $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$'ü hesaplayabiliriz, yani noktalı ürünler vasıtasıyla elemente hesaplayabiliriz.
1. $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$'yı hesaplayabiliriz, yani, her seferinde bir sütun hesaplayabiliriz. Aynı şekilde biz hesaplamak olabilir $\mathbf{A}$ bir satır $\mathbf{A}_{i,:}$ bir seferde.
1. Sadece $\mathbf{A} = \mathbf{B} \mathbf{C}$'ü hesaplayabiliriz.
1. Biz daha küçük blok matrisleri içine $\mathbf{B}$ ve $\mathbf{C}$ kırmak ve aynı anda bir blok $\mathbf{A}$ hesaplamak olabilir.

İlk seçeneği izlersek, $\mathbf{A}_{ij}$ öğesini hesaplamak istediğimiz her seferinde bir satır ve bir sütun vektörünü CPU'ya kopyalamalıyız. Daha da kötüsü, matris elemanlarının sıralı olarak hizalanması nedeniyle, bellekten okurken iki vektörden biri için birçok ayrık konuma erişmemiz gerekiyor. İkinci seçenek çok daha elverişlidir. İçinde $B$ üzerinden geçiş yapmaya devam ederken sütun vektörünü $\mathbf{C}_{:,j}$'ü CPU önbelleğinde tutabiliyoruz. Bu, daha hızlı erişim ile bellek bant genişliği gereksinimini yarıya indirir. Tabii ki, seçenek 3 en çok arzu edilir. Ne yazık ki, çoğu matris önbelleğe tamamen sığmayabilir (sonuçta tartıştığımız şey budur). Bununla birlikte, seçenek 4 pratik olarak kullanışlı bir alternatif sunar: matrisin bloklarını önbelleğe taşıyabilir ve yerel olarak çoğaltabiliriz. Optimize edilmiş kütüphaneler bunu bizim için halledeceklerdir. Bu operasyonların pratikte ne kadar verimli olduğuna bir göz atalım. 

Hesaplama verimliliğinin ötesinde, Python ve derin öğrenme çerçevesinin kendisi tarafından getirilen yükü önemli ölçüde. Python yorumlayıcısı her komutu çalıştırdığımızda MXNet motoruna, hesaplama grafiğine eklemesi ve zamanlama sırasında onunla ilgilenmesi gereken bir komut gönderdiğini hatırlayın. Bu tür yükler oldukça zararlı olabilir. Kısacası, mümkün olduğunca vektorizasyon (ve matrisler) kullanılması şiddetle tavsiye edilir.

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

Öğe bilge atama, değeri $\mathbf{A}$ olarak atamak için sırasıyla $\mathbf{B}$ ve $\mathbf{C}$'in tüm satır ve sütunlarını yineleyir.

```{.python .input}
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

Daha hızlı bir strateji sütun olarak atama gerçekleştirmektir.

```{.python .input}
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
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

Son olarak, en etkili yol, tüm işlemi bir blokta gerçekleştirmektir. Operasyonların ilgili hızının ne olduğunu görelim.

```{.python .input}
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minibatchler

:label:`sec_minibatches` 

Geçmişte, parametreleri güncellemek için tek gözlemler yerine verilerin*minibatches* okuyacağımızı kabul ettik. Şimdi bunun için kısa bir gerekçe veriyoruz. Tek gözlemlerin işlenmesi, oldukça pahalıdır ve altta yatan derin öğrenme çerçevesi adına önemli bir yük oluşturan birçok tek matris vektör (hatta vektör-vektör) çarpımı gerçekleştirmemizi gerektirir. Bu, hem verilere uygulandığında bir ağın değerlendirilmesi (genellikle çıkarım olarak adlandırılır) hem de parametreleri güncellemek için degradeleri hesaplarken geçerlidir. Yani, bu $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ gerçekleştirdiğimiz her zaman geçerlidir 

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Bu işlemin*hesaplama verimliliğini bir seferde bir mini toplu gözlem alanına uygulayarak artırabiliriz. Yani, $\mathbf{g}_t$'ü küçük bir parti üzerinden tek bir gözlem üzerine degradeyi değiştiriyoruz 

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Bunun $\mathbf{g}_t$'in istatistiksel özelliklerine ne yaptığını görelim: hem $\mathbf{x}_t$ hem de minibatch $\mathcal{B}_t$'ün tüm unsurları eğitim setinden rastgele düzgün bir şekilde çizildiğinden, degradenin beklentisi değişmeden kalır. Öte yandan varyans önemli ölçüde azaltılır. Minibatch degradesinin ortalama alınmakta olan $b := |\mathcal{B}_t|$ bağımsız degradelerinden oluştuğundan standart sapması $b^{-\frac{1}{2}}$ faktörü azaltılır. Bu, tek başına, iyi bir şeydir, çünkü güncellemelerin tam degradeyle daha güvenilir bir şekilde hizalandığı anlamına gelir. 

Naively bu, büyük bir minibatch $\mathcal{B}_t$ seçmenin evrensel olarak arzu edileceğini gösterir. Ne yazık ki, bir noktadan sonra, standart sapmadaki ek azalma, hesaplama maliyetindeki doğrusal artışa kıyasla minimumdur. Pratikte, bir GPU belleğine uyurken iyi hesaplama verimliliği sunacak kadar büyük bir mini batch seçiyoruz. Tasarrufları göstermek için bize bazı kodlara bir göz atalım. İçinde aynı matris matris çarpımını gerçekleştiriyoruz, ancak bu sefer bir seferde 64 sütunlu “minibatches” a bölündü.

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

Gördüğümüz gibi, minibatch üzerindeki hesaplama aslında tam matris kadar etkilidir. Bir uyarı sözcüğü sırayla. :numref:`sec_batch_norm`'te bir minibatch içindeki varyans miktarına büyük ölçüde bağımlı olan bir düzenlilik türü kullandık. İkincisini arttırdıkça, varyans azalır ve bununla birlikte parti normalleşmesi nedeniyle gürültü enjeksiyonunun faydası olur. Uygun şartların nasıl yeniden ölçekleneceği ve hesaplanacağı ile ilgili ayrıntılar için bkz. :cite:`Ioffe.2017`. 

## Veri Kümesini Okuma

Minibüslerin verilerden nasıl verimli bir şekilde üretildiğine bir göz atalım. Aşağıda bu optimizasyon algoritmaları karşılaştırmak için kanat [noise from different aircraft](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) test etmek için NASA tarafından geliştirilen bir veri kümesi kullanın. Kolaylık sağlamak için sadece ilk $1,500$ örneklerini kullanıyoruz. Veriler ön işleme için beyazlatılır, yani ortalamaları kaldırır ve varyansı koordinat başına $1$'e yeniden ölçeklendiririz.

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

## Çizilmelerden Uygulama

:numref:`sec_linear_scratch` den minibatch stokastik degrade iniş uygulamasını hatırlayın. Aşağıda biraz daha genel bir uygulama sağlıyoruz. Kolaylık sağlamak için, bu bölümde daha sonra tanıtılan diğer optimizasyon algoritmalarıyla aynı çağrı imzasına sahiptir. Özellikle, `states` durum girişini ekliyoruz ve hiperparametreyi `hyperparams` sözlüğüne yerleştiriyoruz. Buna ek olarak, eğitim işlevindeki her minibatch örneğinin kaybını ortalayacağız, bu nedenle optimizasyon algoritmasındaki degradenin toplu boyutuna bölünmesi gerekmez.

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

Daha sonra, bu bölümün ilerleyen bölümlerinde tanıtılan diğer optimizasyon algoritmalarının kullanımını kolaylaştırmak için genel bir eğitim işlevi uyguluyoruz. Doğrusal regresyon modelini başlatır ve modeli minibatch stokastik degrade iniş ve daha sonra tanıtılan diğer algoritmalarla eğitmek için kullanılabilir.

```{.python .input}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
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
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
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
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
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

Toplu degrade iniş için optimizasyonun nasıl ilerlediğini görelim. Bu, minibatch boyutunu 1500'e (yani toplam örnek sayısına) ayarlayarak elde edilebilir. Sonuç olarak model parametreleri çağı başına yalnızca bir kez güncellenir. Çok az ilerleme var. Aslında, 6 adım ilerleme tezgahları sonra.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Toplu boyut 1'e eşit olduğunda, optimizasyon için stokastik degrade iniş kullanıyoruz. Uygulamanın basitliği için sabit (küçük olsa da) bir öğrenme oranı seçtik. Stokastik degrade inişinde, örnek her işlendiğinde model parametreleri güncelleştirilir. Bizim durumumuzda bu, çak başına 1500 güncellemedir. Gördüğümüz gibi, objektif fonksiyonun değerindeki düşüş bir devirden sonra yavaşlar. Her iki prosedür de bir dönem içinde 1500 örnek işlenmiş olsa da, stokastik degrade iniş bizim deneyde degrade iniş daha fazla zaman tüketir. Bunun nedeni, stokastik degrade iniş parametreleri daha sık güncellenir ve tek bir gözlemleri teker teker işlemek daha az verimli olduğu için.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Son olarak, parti boyutu 100'e eşit olduğunda, optimizasyon için minibatch stokastik degrade iniş kullanıyoruz. Çak başına gereken süre, stokastik degrade iniş için gereken süreden ve toplu degrade iniş süresinden daha kısadır.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

Toplu iş boyutunu 10'a düşürülerek, her bir parti için iş yükünün yürütülmesi daha az verimli olduğundan, her dönem için zaman artar.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Şimdi önceki dört deney için zaman ve kaybını karşılaştırabiliriz. Görüldüğü gibi, stokastik degrade iniş, işlenen örneklerin sayısı açısından GD'den daha hızlı yakınsa da, örneğin degrade örneğinin hesaplanması o kadar verimli olmadığından, GD'den aynı kayba ulaşmak için daha fazla zaman kullanır. Minibatch stokastik degrade iniş yakınsama hızını ve hesaplama verimliliğini değiştirebilir. 10 minibatch boyutu, stokastik degrade inişinden daha etkilidir; 100 mini batch boyutu çalışma zamanı açısından GD'den daha iyi performans gösterir.

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
    # Initialization
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
    # Initialization
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
    # Initialization
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

Son deneyi tekrarlamak için Gluon'u kullanmak aynı davranışı gösterir.

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

* Vektorizasyon, derin öğrenme çerçevesinden kaynaklanan azaltılmış ek yükü ve CPU ve GPU'larda daha iyi bellek yerelliği ve önbelleğe alma nedeniyle kodu daha verimli hale getirir.
* Stokastik degrade inişinden kaynaklanan istatistiksel verimlilik ile aynı anda büyük veri yığınlarının işlenmesinden kaynaklanan hesaplama verimliliği arasında bir işlem vardır.
* Minibatch stokastik degrade iniş her iki dünyanın en iyisini sunar: hesaplama ve istatistiksel verimlilik.
* Mini batch stokastik degrade inişinde, eğitim verilerinin rastgele bir permütasyonu ile elde edilen veri yığınlarını işleriz (yani, her gözlem, rastgele sırada da olsa, her bir gözlem sadece bir kez işlenir).
* Eğitim sırasında öğrenme oranlarının bozulması tavsiye edilir.
* Genel olarak, mini batch stokastik degrade iniş, saat süresi açısından ölçüldüğünde, daha küçük bir riske yakınsama için stokastik degrade iniş ve degrade inişinden daha hızlıdır.

## Egzersizler

1. Toplu iş boyutunu ve öğrenme oranını değiştirin ve objektif fonksiyonun değeri ve her dönemde tüketilen süreye ilişkin düşüş oranını gözlemleyin.
1. MXNet belgelerini okuyun ve `Trainer` sınıfı `set_learning_rate` işlevini kullanarak mini batch stokastik degrade inişinin öğrenme hızını her dönemden sonraki önceki değerinin 1/10'una düşürün.
1. Mini batch stokastik degrade iniş aslında bir varyant ile karşılaştırın*değiştirile* eğitim kümesinden* örnekleri. Ne oluyor?
1. Kötü bir cin size söylemeden veri kümenizi çoğaltır (yani, her gözlem iki kez gerçekleşir ve veri kümeniz orijinal boyutunun iki katına çıkar, ancak kimse size söylemedi). Stokastik degrade iniş, minibatch stokastik degrade iniş ve degrade iniş davranışları nasıl değişir?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:
