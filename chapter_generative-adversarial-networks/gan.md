# Çekişmeli Üretici Ağlar
:label:`sec_basic_gan`

Bu kitabın çoğunda, nasıl tahmin yapabileceğimiz hakkında konuştuk. O yada bu şekilde, veri noktalarıni etiketlere eşleyen derin sinir ağları öğrendik. Bu tür öğrenmeye ayrımcı öğrenme denir, çünkü kedi ve köpek fotoğrafları arasında ayrım yapabilmelerini istiyoruz. Sınıflandırıcılar ve bağlanımcılar (regressor), ayrımcı öğrenmenin örnekleridir. Ggiteri yayma ile eğitilen sinir ağları, büyük karmaşık veri kümelerinde ayrımcı öğrenme hakkında bildiğimizi düşündüğümüz her şeyi altüst etmeyi başardı. Yüksek çözünürlüklü görüntülerdeki sınıflandırma doğruluk oranı, sadece 5-6 yıl içinde işe yaramazlık düzeyinden insan düzeyine (bazı kısıtlamalarla) geldi. Bu bölümde, derin sinir ağlarının şaşırtıcı derecede iyi yaptığı diğer ayrımcı görevler hakkında tartışacağız.

Ancak makine öğrenmesinde ayrımcı problemler çözmekten daha ötesi vardır. Örneğin, herhangi bir etiket içermeyen büyük bir veri kümesi verildiğinde, bu verinin karakterini kısaca özetleyen bir model öğrenmek isteyebiliriz. Böyle bir model verildiğinde, eğitim verisinin dağılımına benzeyen sentetik veri noktaları örnekleyebiliriz. Örneğin, büyük bir yüz fotoğrafı külliyatı verildiğinde, makul bir şekilde, aynı veri kümesinden gelmiş gibi görünen yeni bir foto-gerçekçi imge oluşturabilmeyi isteyebiliriz. Bu tür öğrenmeye üretici modelleme denir.

Yakın zamana kadar, yeni foto-gerçekçi görüntüleri sentezleyebilecek bir yöntem yoktu. Ancak derin sinir ağlarının ayrımcı öğrenmedeki başarısı yeni olasılıkların ortaya çıkmasına vesile oldu. Son üç yıldaki büyük bir yönelim de genellikle gözetimli öğrenme problemi gibi düşünmediğimiz problemlerdeki zorlukların üstesinden gelmede ayrımcı derin ağların kullanması olmuştur. Yinelemeli sinir ağı dil modelleri, bir kez eğitildikten sonra üretici bir model olarak davranabilen  (sonraki karakteri tahmin etmek için eğitilmiş) ayrımcı ağ kullanmanın iyi bir örneğidir.

2014'te çığır açan bir makalede, Çekişmeli Üretici Ağlar (GAN'lar) tanıtıldı :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`, iyi üretici modeller elde etmek için ayrımcı modellerin gücünden yararlanmada akıllıca yeni bir yol gösterildi. GAN'lar, eğer gerçek veriler ile sahte veriler ayrılamıyorsa, o zaman veri üreticinin başarılı olduğu fikrine dayanırlar. İstatistikte buna ikili-örneklem testi denir -  $X=\{x_1,\ldots, x_n\}$ ve $X'=\{x'_1,\ldots, x'_n\}$ veri kümelerinin aynı dağılımdan çekilmiş olup olmadığı sorusuna cevap vermek için yapılan bir testtir. Çoğu istatistik makalesi ile GAN arasındaki temel fark, ikincisinin bu fikri yapıcı bir şekilde kullanmasıdır. Başka bir deyişle, "hey, bu iki veri kümesi aynı dağılımdan gelmiş gibi görünmüyor" demek için bir model eğitmek yerine, [ikili-örneklem testini](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) üretici bir modele eğitim sinyalleri sağlamak için kullandılar. Bu bize gerçek verilere benzeyen bir şey üretene kadar veri üreticiyi iyileştirme olanağını tanır. Sınıflandırıcımız son teknoloji bir derin sinir ağı da olsa bile üreticinin en azından sınıflandırıcıyı kandırabilmesi gerekir.

![Çekişmeli Üretici Ağlar](../img/gan.svg)
:label:`fig_gan`

GAN mimarisi şu şekilde gösterilmektedir :numref:`fig_gan`.
Gördüğünüz gibi, GAN mimarisinde iki parça bulunur - öncelikle, potansiyel olarak gerçek gibi görünen verileri üretebilecek bir cihaza ihtiyacımız vardır (bir derin ağ düşünebiliriz, fakat oyun oluşturma motoru gibi herhangi bir şey de olabilir). İmgelerle uğraşıyorsak, bu imge üretmeyi gerektirir. Mesela konuşmayla uğraşıyorsak, ses dizileri üretmeyi gerektirir. Buna üretici ağ diyoruz. İkinci bileşen, ayrımcı ağdır. Sahte ve gerçek verileri birbirinden ayırt etmeye çalışır. Her iki ağ da birbiriyle rekabet içindedir. Üretici ağ, ayrımcı ağı kandırmaya çalışır. Bir noktada ayrımcı ağ yeni sahte verilere uyum sağlar. Bu bilgi de üretici ağı iyileştirmek için kullanılır.

Ayrımcı, $x$ girdisinin gerçek mi (gerçek veriden) yoksa sahte mi (üreticiden) olduğunu ayırt eden bir ikili sınıflandırıcıdır. Tipik olarak, ayrımcı $\mathbf x$ girdisi için bir skaler tahmin, $o \in \mathbb R$, verir ve bunun için mesela gizli boyutu 1 olan yoğun bir katman kullanabilir ve ardından da tahmin olasılığı, $D(\mathbf x) = 1/(1+e^{-o})$, için sigmoid fonksiyonunu uygulayabilir. $y$ etiketinin gerçek veriler için $1$ ve sahte veriler için $0$ olduğunu varsayalım. Ayrımcıyı, çapraz entropi kaybını en aza indirecek şekilde eğitiriz, *yani*,

$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$

Üretici için, önce bir rasgelelik kaynağından bir $\mathbf z\in \mathbb R^d$ parametresi çekeriz, *örneğin*, normal bir dağılım, $\mathbf z \sim \mathcal{N} (0, 1)$, kullanabiliriz. Gizli değişkeni genellikle $\mathbf z$ ile gösteriyoruz. Daha sonra $\mathbf x'=G(\mathbf z)$ oluşturmak için bir işlev uygularız. Üreticinin amacı, ayrımcıyı $\mathbf x'= G(\mathbfz)$'yı gerçek veri olarak *yani*, $D( G(\mathbf z)) \approx 1$ diye, sınıflandırması için kandırmaktır. Başka bir deyişle, belirli bir $D$ ayrımcısı için, $y = 0$, olduğunda çapraz entropi kaybını maksimize etmek için $G$ üreticisinin parametrelerini güncelleriz, *yani* 

$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$

Üretici mükemmel bir iş çıkarırsa, o zaman $D(\mathbf x')\approx 1$ olur, böylece yukarıdaki kayıp 0'a yaklaşır, bu da gradyanların ayrımcı için anlamlı bir iyileştirme sağlayamayacak kadar küçük olmasına neden olur. Bu yüzden, genellikle aşağıdaki kaybı en aza indirmeyi deneriz:

$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$

Bu da $y = 1$ etiketini vererek $\mathbf x'=G(\mathbf z)$'yı ayrımcıya beslemek demektir.

Özetle, $D$ ve $G$, kapsamlı amaç işlevine sahip bir "minimax" oyunu oynuyorlar:

$$min_D max_G \{ -E_{x \sim \text{Data}} log D(\mathbf x) - E_{z \sim \text{Noise}} log(1 - D(G(\mathbf z))) \}.$$

GAN uygulamalarının çoğu imge bağlamındadır. Sizlere gösterme amaciyla, önce çok daha basit bir dağılım oturtmakla yetineceğiz. Bir Gaussian için dünyanın en verimsiz parametre tahmincisini oluşturmak için GAN'ları kullanırsak ne olacağını göreceğiz. Hadi başlayalım.

```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

## Bir miktar "gerçek" veri üretme

Bu dünyanın en yavan örneği olacağından, basit bir Gauss'tan çekilen verileri üretiyoruz.

```{.python .input  n=2}
X = np.random.normal(size=(1000, 2))
A = np.array([[1, 2], [-0.1, 0.5]])
b = np.array([1, 2])
data = X.dot(A) + b
```

Bakalım elimizde neler var. Bu, ortalamasi $b$ ve kovaryans matrisi $A^TA$ olan keyfi bir şekilde kaydırılmış bir Gaussian olacaktır.

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

Üretici ağımız mümkün olan en basit ağ olacak - tek katmanlı bir doğrusal model. Bunun nedeni, bu doğrusal ağı bir Gauss veri üreticisi ile yönlendirecek olmamız. Böylece, kelimenin birebir anlamıyla, sadece nesneleri mükemmel bir şekilde taklit etmek için gerekli parametreleri öğrenmesi gerekecektir.

```{.python .input  n=5}
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

## Ayrımcı

Ayrımcı için biraz daha hassas olacağız: İşleri biraz daha ilginç hale getirmek için 3 katmanlı bir MLP kullanacağız.

```{.python .input  n=6}
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

## Eğitim

İlk olarak ayrımcıyı güncellemek için bir işlev tanımlayalım.

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

Üreticiyi de benzer şekilde güncelleiyoruz. Burada çapraz entropi kaybını tekrar kullanıyoruz ama sahte verinin etiketini $0$'dan $1$'e çeviriyoruz.

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

Hem ayrımcı hem de üretici, çapraz entropi kaybıyla ikili lojistik bağlanım uygular. Eğitim sürecini kolaylaştırmak için Adam'ı kullanıyoruz. Her yinelemede, önce ayrımcıyı ve ardından da üreticiyi güncelliyoruz. Ayrıca hem kayıpları hem de üretilen örnekleri görselleştiriyoruz.

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
* Üretici, çapraz entropi kaybını maksimize ederek *yani*, $\max \log(D(\mathbf{x'}))$ yoluyla, ayrımcıyı kandırmak için gerçek imgeye olabildiğince yakın imgeler oluşturur.
* Ayrımcı, çapraz entropi kaybını en aza indirerek, oluşturulan imgeleri gerçek imgelerden ayırt etmeye çalışır, *yani*, $\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$ optimize edilir.

## Alıştırmalar

* Üreticinin kazandığı yerde bir denge var mıdır, *mesela* ayrımcının sonlu örnekler üzerinden iki dağılımı ayırt edemediği gibi?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/408)
:end_tab:
