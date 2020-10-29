# Derin Evrişimli Çekişmeli Üretici Ağlar
:label:`sec_dcgan`

:numref:`sec_basic_gan`'de, GAN'ların nasıl çalıştığına dair temel fikirleri tanıttık. Tekdüze veya normal dağılım gibi basit, örneklenmesi kolay bir dağılımdan örnekler alabileceklerini ve bunları bazı veri kümelerinin dağılımıyla eşleşiyor gibi görünen örneklere dönüştürebileceklerini gösterdik. Burada bir 2B Gauss dağılımını eşleştirme örneğimiz amaca uygunken, özellikle heyecan verici bir örnek değil.

Bu bölümde, foto-gerçekçi imgeler oluşturmak için GAN'ları nasıl kullanabileceğinizi göstereceğiz. Modellerimizi derin evrişimli GAN'lara (DCGAN) dayandıracağız :cite:`Radford.Metz.Chintala.2015`. Ayrımcı bilgisayarla görme problemleri için çok başarılı olduğu kanıtlanmış evrişimli mimariyi ödünç alacağız ve GAN'lar aracılığıyla foto-gerçekçi imgeler oluşturmak için nasıl kullanılabileceklerini göstereceğiz.

```{.python .input  n=1}
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

## Pokemon Veri Kümesi

Kullanacağımız veri kümesi, [pokemondb](https://pokemondb.net/sprites) adresinden elde edilen Pokemon görselleri koleksiyonudur. İlk önce bu veri kümesini indirelim, çıkaralım ve yükleyelim.

```{.python .input  n=2}
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
pokemon = gluon.data.vision.datasets.ImageFolderDataset(data_dir)
```

Her bir imgeyi $64 \times 64$ şekilde yeniden boyutlandırıyoruz. `ToTensor` dönüşümü piksel değerini $[0, 1]$ aralığına izdüşürürken bizim üreticimiz $[- 1, 1]$ aralığından çıktılar elde etmek için tanh işlevini kullanacak. Bu nedenle, verileri değer aralığına uyması için $0.5$ ortalama ve $0.5$ standart sapma ile normalize ediyoruz.

```{.python .input  n=3}
batch_size = 256
transformer = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(64),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize(0.5, 0.5)
])
data_iter = gluon.data.DataLoader(
    pokemon.transform_first(transformer), batch_size=batch_size,
    shuffle=True, num_workers=d2l.get_dataloader_workers())
```

İlk 20 imgeyi görselleştirelim.

```{.python .input  n=4}
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[0:20,:,:,:].transpose(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break
```

## Üretici

Üreticinin, $d$-uzunluklu vektör olan $\mathbf z \in \mathbb R^d$ gürültü değişkenini, genişliği ve yüksekliği $64 \times 64$ olan bir RGB imgesine eşlemesi gerekir. :numref:`sec_fcn` bölümünde, girdi boyutunu büyütmek için devrik evrişim katmanı (bakınız :numref:`sec_transposed_conv`) kullanan tam evrişimli ağı tanıttık. Üreticinin temel bloğu, devrik bir evrişim katmanı, ardından toptan normalleştirme ve ReLU etkinleştirmesi içerir.

```{.python .input  n=5}
class G_block(nn.Block):
    def __init__(self, channels, kernel_size=4,
                 strides=2, padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.Conv2DTranspose(
            channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = nn.BatchNorm()
        self.activation = nn.Activation('relu')

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

Varsayılan olarak, devrik evrişim katmanı $k_h = k_w = 4$'lük çekirdek, $s_h = s_w = 2$'lik uzun adımlar (stride) ve $p_h = p_w = 1$'lik dolgu (padding) kullanır. $n_h^{'} \times n_w^{'} = 16 \times 16$ girdi şekli ile, üretici bloğu girdinin genişliğini ve yüksekliğini iki katına çıkaracaktır.

$$
\begin{aligned}
n_h^{'} \times n_w^{'} &= [(n_h k_h - (n_h-1)(k_h-s_h)- 2p_h] \times [(n_w k_w - (n_w-1)(k_w-s_w)- 2p_w]\\
  &= [(k_h + s_h (n_h-1)- 2p_h] \times [(k_w + s_w (n_w-1)- 2p_w]\\
  &= [(4 + 2 \times (16-1)- 2 \times 1] \times [(4 + 2 \times (16-1)- 2 \times 1]\\
  &= 32 \times 32 .\\
\end{aligned}
$$

```{.python .input  n=6}
x = np.zeros((2, 3, 16, 16))
g_blk = G_block(20)
g_blk.initialize()
g_blk(x).shape
```

Devrik evrişim katmanını $4 \times 4$ çekirdek, $1 \times 1$ uzun adımları ve sıfır dolgu olarak değiştirelim. $1 \times 1$ girdi boyutuyla çıktının genişliği ve yüksekliği sırasıyla 3 artar.

```{.python .input  n=7}
x = np.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk.initialize()
g_blk(x).shape
```

Üretici, girdinin hem genişliğini hem de yüksekliğini 1'den 32'ye çıkaran dört temel bloktan oluşur. Aynı zamanda, önce saklı değişkeni $64 \times 8$ kanala izdüşürür ve ardından her seferinde kanalları yarıya indirir. Sonunda, çıktının üretilmesi için bir devrik evrişim katmanı kullanılır. Genişliği ve yüksekliği istenen $64 \times 64$ şekline uyacak şekilde ikiye katlar ve kanal sayısını $3$'e düşürür. Çıktı değerlerini $(- 1, 1)$ aralığına yansıtmak için tanh etkinleştirme işlevi uygulanır.

```{.python .input  n=8}
n_G = 64
net_G = nn.Sequential()
net_G.add(G_block(n_G*8, strides=1, padding=0),  # Output: (64 * 8, 4, 4)
          G_block(n_G*4),  # Output: (64 * 4, 8, 8)
          G_block(n_G*2),  # Output: (64 * 2, 16, 16)
          G_block(n_G),   # Output: (64, 32, 32)
          nn.Conv2DTranspose(
              3, kernel_size=4, strides=2, padding=1, use_bias=False,
              activation='tanh'))  # Output: (3, 64, 64)
```

Üreticinin çıktı şeklini doğrulamak için 100 boyutlu bir saklı değişken oluşturalım.

```{.python .input  n=9}
x = np.zeros((1, 100, 1, 1))
net_G.initialize()
net_G(x).shape
```

## Ayrımcı

Ayrımcı, etkinleştirme işlevi olarak sızıntılı (leaky) ReLU kullanması dışında normal bir evrişimli ağdır. $\alpha \in[0, 1]$ verildiğinde, tanım şöyledir:

$$\textrm{leaky ReLU}(x) = \begin{cases}x & \text{if}\ x > 0\\ \alpha x &\text{otherwise}\end{cases}.$$

Görüldüğü gibi, $\alpha = 0$ ise normal ReLU, $\alpha = 1$ ise bir birim fonksiyondur. $\alpha \in (0, 1)$ için, sızıntılı ReLU, negatif bir girdi için sıfır olmayan bir çıktı veren doğrusal olmayan bir fonksiyondur. Bir nöronun her zaman negatif bir değer verebileceği ve bu nedenle ReLU'nun gradyanı 0 olduğu için herhangi bir ilerleme kaydedemeyeceği "ölmekte olan ReLU" (dying ReLU) problemini çözmeyi amaçlamaktadır.

```{.python .input  n=10}
alphas = [0, 0.2, 0.4, .6, .8, 1]
x = np.arange(-2, 1, 0.1)
Y = [nn.LeakyReLU(alpha)(x).asnumpy() for alpha in alphas]
d2l.plot(x.asnumpy(), Y, 'x', 'y', alphas)
```

Ayrımcının temel bloğu, bir evrişim katmanı ve ardından bir toptan normalleştirme katmanı ve bir sızıntılı ReLU etkinleştirmesidir. Evrişim katmanının hiper parametreleri, üretici bloğundaki devrik evrişim katmanına benzer.

```{.python .input  n=11}
class D_block(nn.Block):
    def __init__(self, channels, kernel_size=4, strides=2,
                 padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2D(
            channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = nn.BatchNorm()
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

Varsayılan ayarlara sahip temel bir blok, girdilerin genişliğini ve yüksekliğini, burada gösterdiğimiz gibi :numref:`sec_padding`, yarıya düşürecektir. Örneğin, $k_h = k_w = 4$ çekirdek, $s_h = s_w = 2$ uzun adım, $p_h = p_w = 1$ dolgu ve $n_h = n_w = 16$ girdi şekli verildiğinde, çıktının şekli şöyle olacaktır:

$$
\begin{aligned}
n_h^{'} \times n_w^{'} &= \lfloor(n_h-k_h+2p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+2p_w+s_w)/s_w\rfloor\\
  &= \lfloor(16-4+2\times 1+2)/2\rfloor \times \lfloor(16-4+2\times 1+2)/2\rfloor\\
  &= 8 \times 8 .\\
\end{aligned}
$$

```{.python .input  n=12}
x = np.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk.initialize()
d_blk(x).shape
```

Ayrımcı üreticinin bir yansımasıdır.

```{.python .input  n=13}
n_D = 64
net_D = nn.Sequential()
net_D.add(D_block(n_D),   # Output: (64, 32, 32)
          D_block(n_D*2),  # Output: (64 * 2, 16, 16)
          D_block(n_D*4),  # Output: (64 * 4, 8, 8)
          D_block(n_D*8),  # Output: (64 * 8, 4, 4)
          nn.Conv2D(1, kernel_size=4, use_bias=False))  # Output: (1, 1, 1)
```

Tek bir tahmin değeri elde etmek için son katmanda çıktı kanalı $1$ olan bir evrişim katmanı kullanır.

```{.python .input  n=15}
x = np.zeros((1, 3, 64, 64))
net_D.initialize()
net_D(x).shape
```

## Eğitim

Temel GAN, :numref:`sec_basic_gan`, ile karşılaştırıldığında, birbirlerine benzer olduklarından hem üretici hem de ayrımcı için aynı öğrenme oranını kullanıyoruz. Ek olarak, Adam'daki (:numref:`sec_adam`) $\beta_1$'yı $0.9$'dan $0.5$'e değiştiriyoruz. Üretici ve ayrımcı birbiriyle çekiştiği için, hızlı değişen gradyanlarla ilgilenmek için momentumun, ki geçmiş gradyanların üssel ağırlıklı hareketli ortalamasıdır, düzgünlüğünü azaltır. Ayrıca, rastgele üretilen `Z` gürültüsü bir 4B tensördür ve bu nedenle hesaplamayı hızlandırmak için GPU kullanırız.

```{.python .input  n=20}
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          ctx=d2l.try_gpu()):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True, ctx=ctx)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True, ctx=ctx)
    trainer_hp = {'learning_rate': lr, 'beta1': 0.5}
    trainer_D = gluon.Trainer(net_D.collect_params(), 'adam', trainer_hp)
    trainer_G = gluon.Trainer(net_G.collect_params(), 'adam', trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.as_in_ctx(ctx), Z.as_in_ctx(ctx),
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Show generated examples
        Z = np.random.normal(0, 1, size=(21, latent_dim, 1, 1), ctx=ctx)
        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).transpose(0, 2, 3, 1) / 2 + 0.5
        imgs = np.concatenate(
            [np.concatenate([fake_x[i * 7 + j] for j in range(7)], axis=1)
             for i in range(len(fake_x)//7)], axis=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs.asnumpy())
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(ctx)}')
```

Modeli sadece gösterim amaçlı az sayıda dönemle eğitiyoruz. Daha iyi performans için, `num_epochs` değişkeni daha büyük bir sayıya ayarlayabiliriz.

```{.python .input  n=21}
latent_dim, lr, num_epochs = 100, 0.005, 20
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
```

## Özet

* DCGAN mimarisi, ayrımcı için dört evrişimli katmana ve üretici için dört "kesirli-uzun adımlı" evrişimli katmana sahiptir.
* Ayrımcı, toptan normalleştirme (girdi katmanı hariç) ve sızıntılı ReLU etkinleştirmeleri olan 4 katman uzun adımlı evrişimlerdir.
* Sızıntılı ReLU, negatif bir girdi için sıfır olmayan bir çıktı veren doğrusal olmayan bir fonksiyondur. "Ölmekte olan ReLU" sorununu çözmeyi amaçlar ve gradyanların mimari boyunca daha kolay akmasına yardımcı olur.

## Alıştırmalar

1. Sızıntılı ReLU yerine standart ReLU etkinleştirmesi kullanırsak ne olur?
1. DCGAN'ı Fashion-MNIST'e uygulayın ve hangi kategorinin işe yarayıp hangilerinin yaramadığını görün.


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/409)
:end_tab:
