# Çekişmeli Üretici Ağlar
:label:`sec_basic_gan`

Bu kitabın çoğunda, nasıl tahminler yapacağımız hakkında konuştuk. O yada bu şekilde, veri noktalarından etiketlere eşleyen derin sinir ağları öğrendik. Bu tür öğrenmeye ayrımcı öğrenme denir, çünkü biz de kedi fotoğrafları ile köpek fotoğrafları arasında ayrım yapabilmek istiyoruz. Sınıflandırıcılar ve bağlanımcılar (regressor), ayrımcı öğrenmenin örnekleridir. Ayrıca geri yayma ile eğitilen sinir ağları, büyük karmaşık veri kümelerinde ayrımcı öğrenme hakkında bildiğimizi düşündüğümüz her şeyi altüst etti. Yüksek çözünürlüklü görüntülerde sınıflandırma doğruluğu, sadece 5-6 yıl içinde işe yaramazlıktan insan düzeyine (bazı kısıtlamalarla) geçti. Derin sinir ağlarının şaşırtıcı derecede iyi yaptığı diğer tüm ayrımcı görevler hakkında bir tartışma yapacağız.

Ancak makine öğrenmesinde ayrımcı görevleri çözmekten daha fazlası var. Örneğin, herhangi bir etiket içermeyen büyük bir veri kümesi verildiğinde, bu verilerin karakterini kısaca özetleyen bir model öğrenmek isteyebiliriz. Böyle bir model verildiğinde, eğitim verilerinin dağılımına benzeyen sentetik veri noktalarını örnekleyebiliriz. Örneğin, büyük bir yüz fotoğrafı külliyatı verildiğinde, makul bir şekilde aynı veri kümesinden gelmiş gibi görünen yeni bir foto-gerçekçi imge oluşturabilmeyi isteyebiliriz. Bu tür öğrenmeye üretici modelleme denir.

Yakın zamana kadar, yeni foto-gerçekçi görüntüleri sentezleyebilecek bir yöntemimiz yoktu. Ancak derin sinir ağlarının ayrımcı öğrenmedeki başarısı yeni olasılıkları ortaya çıkardı. Son üç yıldaki büyük bir yönelim, genellikle gözetimli öğrenme problemleri olarak düşünmediğimiz problemlerdeki zorlukların üstesinden gelmek için ayrımcı derin ağların uygulanması olmuştur. Yinelemeli sinir ağı dili modelleri, bir kez eğitildikten sonra üretici bir model olarak davranabilen ayrımcı bir ağ (sonraki karakteri tahmin etmek için eğitilmiş) kullanmanın bir örneğidir.

2014'te çığır açan bir makale, Çekişmeli Üretici Ağları (GAN'lar) tanıttı :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`, iyi üretici modeller elde etmek için ayrımcı modellerin gücünden yararlanmanın akıllıca yeni bir yolu. GAN'lar, eğer gerçek veriler ile sahte veriler ayrılamıyorsa, o zaman veri üreticinin iyi olduğu fikrine dayanırlar. İstatistikte buna ikili-örneklem testi denir -  $X=\{x_1,\ldots, x_n\}$ ve $X'=\{x'_1,\ldots, x'_n\}$ veri kümelerinin aynı dağılımdan alınmış olup olmadığı sorusuna cevap vermek için yapılan bir testtir. Çoğu istatistik makalesi ile GAN arasındaki temel fark, ikincisinin bu fikri yapıcı bir şekilde kullanmasıdır. Başka bir deyişle, "hey, bu iki veri kümesi aynı dağılımdan gelmiş gibi görünmüyor" demek için bir model eğitmek yerine, [ikili-örneklem testi](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) üretici bir modele eğitim sinyalleri sağlamak için kullandılar. Bu, gerçek verilere benzeyen bir şey üretene kadar veri üreticiyi iyileştirmemize olanak tanır. En azından sınıflandırıcıyı kandırması gerekiyor. Sınıflandırıcımız son teknoloji bir derin sinir ağı olsa bile.

![Çekişmeli Üretici Ağlar](../img/gan.svg)
:label:`fig_gan`

GAN mimarisi şu şekilde gösterilmektedir :numref:`fig_gan`.
Gördüğünüz gibi, GAN mimarisinde iki parça vardır - ilk olarak, potansiyel olarak gerçek gibi görünen verileri üretebilecek bir cihaza ihtiyacımız var (bir derin ağ diyelim, ancak oyun oluşturma motoru gibi herhangi bir şey de olabilir). İmgelerle uğraşıyorsak, bu imge üretmeyi gerektirir. Mesela konuşmayla uğraşıyorsak, ses dizileri üretmesi gerekir. Buna üretici ağ diyoruz. İkinci bileşen, ayrımcı ağdır. Sahte ve gerçek verileri birbirinden ayırt etmeye çalışır. Her iki ağ da birbiriyle rekabet halindedir. Üretici ağ, ayrımcı ağı kandırmaya çalışır. Bu noktada, ayrımcı ağ yeni sahte verilere uyum sağlar. Bu bilgi de, üretici ağı geliştirmek için kullanılır.

Ayrımcı, $x$ girdisinin gerçek mi (gerçek verilerden) yoksa sahte mi (üreticiden) olduğunu ayırt eden ikili bir sınıflandırıcıdır. Tipik olarak, ayrımcı $\mathbf x$ girdisi için bir skaler tahmin $o \in \mathbb R$ verir, mesela gizli boyutu 1 olan yoğun bir katman kullanılır ve ardından tahmin olasılığı, $D(\mathbf x) = 1/(1+e^{-o})$, için sigmoid fonksiyonu uygulanır. $y$ etiketinin gerçek veriler için $1$ ve sahte veriler için $0$ olduğunu varsayın. Ayrımcıyı, çapraz entropi kaybını en aza indirecek şekilde eğitiyoruz, *yani*,

$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$

Üretici için, önce bir rasgelelik kaynağından bir $\mathbf z\in \mathbb R^d$ parametresi çekilir, *örneğin*, normal bir dağılımdan, $\mathbf z \sim \mathcal{N} (0, 1)$. Gizli değişkene genellikle $\mathbf z$ diyoruz. Daha sonra $\mathbf x'=G(\mathbf z)$ oluşturmak için bir işlev uygulanır. Üreticinin amacı, ayrımcıyı $\mathbf x'= G(\mathbfz)$'yı gerçek veri olarak *yani*, $D( G(\mathbf z)) \approx 1$ olarak sınıflandırması için kandırmaktır. Başka bir deyişle, belirli bir $D$ ayrımcısı için, $y = 0$, olduğunda çapraz entropi kaybını maksimize etmek için $G$ üreticisinin parametrelerini güncelliyoruz, *yani* 

$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$

Üretici mükemmel bir iş çıkarırsa, o zaman $D(\mathbf x')\approx 1$ olur, yani yukarıdaki kayıp 0'a yaklaşır, bu da gradyanların ayrımcı için anlamlı bir iyileştirme sağlayamayacak kadar küçük olmasına neden olur. Bu nedenle, genellikle aşağıdaki kaybı en aza indiririz:

$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$

Bu da $y = 1$ etiketini vererek $\mathbf x'=G(\mathbf z)$'yı ayrımcıya beslemektir.

Özetle, $D$ ve $G$, kapsamlı amaç işlevine sahip bir "minimax" oyunu oynuyorlar:

$$min_D max_G \{ -E_{x \sim \text{Data}} log D(\mathbf x) - E_{z \sim \text{Noise}} log(1 - D(G(\mathbf z))) \}.$$

GAN uygulamalarının çoğu imge bağlamındadır. Gösterme amaçlı olarak, önce çok daha basit bir dağılım oturtmakla yetineceğiz. Bir Gaussian için dünyanın en verimsiz parametre tahmincisini oluşturmak için GAN'ları kullanırsak ne olacağını göstereceğiz. Hadi başlayalım.

```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

## Bir miktar "gerçek" veri üretme

Bu dünyanın en yavan örneği olacağından, basitçe bir Gauss'tan çekilen verileri üretiyoruz.

```{.python .input  n=2}
X = np.random.normal(size=(1000, 2))
A = np.array([[1, 2], [-0.1, 0.5]])
b = np.array([1, 2])
data = X.dot(A) + b
```

Bakalım elimizde ne var. Bu, ortalamasi $b$ ve kovaryans matrisi $A^TA$ olan keyfi bir şekilde kaydırılmış bir Gaussian olmalıdır.

```{.python .input  n=3}
d2l.set_figsize()
d2l.plt.scatter(data[:100, 0].asnumpy(), data[:100, 1].asnumpy());
print(f'The covariance matrix is\n{np.dot(A.T, A)}')
```

```{.python .input  n=4}
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```

## Üretici

Üretici ağımız mümkün olan en basit ağ olacak - tek katmanlı bir doğrusal model. Bunun nedeni, bu doğrusal ağı bir Gauss veri üretici ile yölendirecek olmamız. Böylece, kelimenin tam anlamıyla, sadece nesneleri mükemmel bir şekilde taklit etmek için parametreleri öğrenmesi gerekecektir.

```{.python .input  n=5}
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

## Ayrımcı

Ayrımcı için biraz daha ayırt edici olacağız: İşleri biraz daha ilginç hale getirmek için 3 katmanlı bir MLP kullanacağız.

```{.python .input  n=6}
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

## Eğitim

İlk olarak ayrımcıyı güncellemek için bir işlev tanımlarız.

```{.python .input  n=7}
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Do not need to compute gradient for `net_G`, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

Üretici de benzer şekilde güncellenir. Burada çapraz entropi kaybını tekrar kullanıyoruz ama sahte verinin etiketini $0$'dan $1$'e çeviriyoruz.

```{.python .input  n=8}
def update_G(Z, net_D, net_G, loss, trainer_G):  #@save
    """Update generator."""
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

Hem ayrımcı hem de üretici, çapraz entropi kaybıyla ikili lojistik regresyon uygular. Eğitim sürecini kolaylaştırmak için Adam'ı kullanıyoruz. Her yinelemede, önce ayrımcıyı ve ardından üreticiyi güncelliyoruz. Hem kayıpları hem de üretilen örnekleri görselleştiriyoruz.

```{.python .input  n=9}
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(),
                              'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(),
                              'adam', {'learning_rate': lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['generator', 'discriminator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs+1):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

Şimdi, Gauss dağılımına oturacak hiper parametreleri belirliyoruz.

```{.python .input  n=10}
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, data[:100].asnumpy())
```

## Özet

* Çekişmeli üretici ağlar (GAN'lar), iki derin ağdan oluşur: Üretici ve ayrımcı.
* Üretici, çapraz entropi kaybını maksimize ederek *yani*, $\max \log(D(\mathbf{x'}))$ yoluyla, ayrımcıyı kandırmak için gerçek imgeye olabildiğince yakın bir imge oluşturur.
* Ayrımcı, çapraz entropi kaybını en aza indirerek, oluşturulan imgeleri gerçek imgelerden ayırt etmeye çalışır, *yani*, $\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$.

## Alıştırmalar

* Üreticinin kazandığı yerde bir denge var mıdır, *mesela* ayrımcının sonlu örnekler üzerinden iki dağılımı ayırt edemediği?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/408)
:end_tab:
