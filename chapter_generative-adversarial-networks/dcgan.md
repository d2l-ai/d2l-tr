# Derin Evrişimli Çekişmeli Üretici Ağlar
:label:`sec_dcgan`

:numref:`sec_basic_gan` içinde, GAN'ların nasıl çalıştığına dair temel fikirleri tanıttık. Tekdüze veya normal dağılım gibi basit, örneklenmesi kolay bir dağılımdan örnekler alabileceklerini ve bunları bazı veri kümelerinin dağılımıyla eşleşiyor gibi görünen örneklere dönüştürebileceklerini gösterdik. Burada bir 2B Gauss dağılımını eşleştirme örneğimiz amaca uygunken, özellikle heyecan verici bir örnek değil.

Bu bölümde, foto-gerçekçi imgeler oluşturmak için GAN'ları nasıl kullanabileceğinizi göstereceğiz. Modellerimizi derin evrişimli GAN'lara (DCGAN) dayandıracağız :cite:`Radford.Metz.Chintala.2015`. Ayrımcı bilgisayarla görme problemleri için çok başarılı olduğu kanıtlanmış evrişimli mimariyi ödünç alacağız ve GAN'lar aracılığıyla foto-gerçekçi imgeler oluşturmak için nasıl kullanılabileceklerini göstereceğiz.

```{.python .input}
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import warnings
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
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

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
pokemon = torchvision.datasets.ImageFolder(data_dir)
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
batch_size = 256
pokemon = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, batch_size=batch_size, image_size=(64, 64))
```

Her bir imgeyi $64 \times 64$ şekilde yeniden boyutlandırıyoruz. `ToTensor` dönüşümü piksel değerini $[0, 1]$ aralığına izdüşürürken bizim üreticimiz $[- 1, 1]$ aralığından çıktılar elde etmek için tanh işlevini kullanacak. Bu nedenle, verileri değer aralığına uyması için $0.5$ ortalama ve $0.5$ standart sapma ile normalize ediyoruz.

```{.python .input}
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

```{.python .input}
#@tab pytorch
batch_size = 256
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])
pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(
    pokemon, batch_size=batch_size,
    shuffle=True, num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
def transform_func(X):
    X = X / 255.
    X = (X - 0.5) / (0.5)
    return X

# For TF>=2.4 use `num_parallel_calls = tf.data.AUTOTUNE`
data_iter = pokemon.map(lambda x, y: (transform_func(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
data_iter = data_iter.cache().shuffle(buffer_size=1000).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
```

İlk 20 imgeyi görselleştirelim.

```{.python .input}
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[0:20,:,:,:].transpose(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break
```

```{.python .input}
#@tab pytorch
warnings.filterwarnings('ignore')
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[0:20,:,:,:].permute(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize(figsize=(4, 4))
for X, y in data_iter.take(1):
    imgs = X[:20, :, :, :] / 2 + 0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
```

## Üretici

Üreticinin, $d$-uzunluklu vektör olan $\mathbf z \in \mathbb R^d$ gürültü değişkenini, genişliği ve yüksekliği $64 \times 64$ olan bir RGB imgesine eşlemesi gerekir. :numref:`sec_fcn` bölümünde, girdi boyutunu büyütmek için devrik evrişim katmanı (bkz. :numref:`sec_transposed_conv`) kullanan tam evrişimli ağı tanıttık. Üreticinin temel bloğu, devrik bir evrişim katmanı, ardından toptan normalleştirme ve ReLU etkinleştirmesi içerir.

```{.python .input}
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

```{.python .input}
#@tab pytorch
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

```{.python .input}
#@tab tensorflow
class G_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=4, strides=2, padding="same",
                 **kwargs):
        super().__init__(**kwargs)
        self.conv2d_trans = tf.keras.layers.Conv2DTranspose(
            out_channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        
    def call(self, X):
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

```{.python .input}
x = np.zeros((2, 3, 16, 16))
g_blk = G_block(20)
g_blk.initialize()
g_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 16, 16))
g_blk = G_block(20)
g_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 16, 16, 3))  # Channel last convention
g_blk = G_block(20)
g_blk(x).shape
```

Devrik evrişim katmanını $4 \times 4$ çekirdek, $1 \times 1$ uzun adımları ve sıfır dolgu olarak değiştirelim. $1 \times 1$ girdi boyutuyla çıktının genişliği ve yüksekliği sırasıyla 3 artar.

```{.python .input}
x = np.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk.initialize()
g_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 1, 1, 3))
# `padding="valid"` dolgu olmamasına karşılık gelir
g_blk = G_block(20, strides=1, padding="valid")
g_blk(x).shape
```

Üretici, girdinin hem genişliğini hem de yüksekliğini 1'den 32'ye çıkaran dört temel bloktan oluşur. Aynı zamanda, önce saklı değişkeni $64 \times 8$ kanala izdüşürür ve ardından her seferinde kanalları yarıya indirir. Sonunda, çıktının üretilmesi için bir devrik evrişim katmanı kullanılır. Genişliği ve yüksekliği istenen $64 \times 64$ şekline uyacak şekilde ikiye katlar ve kanal sayısını $3$'e düşürür. Çıktı değerlerini $(- 1, 1)$ aralığına yansıtmak için tanh etkinleştirme işlevi uygulanır.

```{.python .input}
n_G = 64
net_G = nn.Sequential()
net_G.add(G_block(n_G*8, strides=1, padding=0),  # Çıktı: (64 * 8, 4, 4)
          G_block(n_G*4),  # Çıktı: (64 * 4, 8, 8)
          G_block(n_G*2),  # Çıktı: (64 * 2, 16, 16)
          G_block(n_G),    # Çıktı: (64, 32, 32)
          nn.Conv2DTranspose(
              3, kernel_size=4, strides=2, padding=1, use_bias=False,
              activation='tanh'))  # Çıktı: (3, 64, 64)
```

```{.python .input}
#@tab pytorch
n_G = 64
net_G = nn.Sequential(
    G_block(in_channels=100, out_channels=n_G*8,
            strides=1, padding=0),                  # Çıktı: (64 * 8, 4, 4)
    G_block(in_channels=n_G*8, out_channels=n_G*4), # Çıktı: (64 * 4, 8, 8)
    G_block(in_channels=n_G*4, out_channels=n_G*2), # Çıktı: (64 * 2, 16, 16)
    G_block(in_channels=n_G*2, out_channels=n_G),   # Çıktı: (64, 32, 32)
    nn.ConvTranspose2d(in_channels=n_G, out_channels=3, 
                       kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh())  # Çıktı: (3, 64, 64)
```

```{.python .input}
#@tab tensorflow
n_G = 64
net_G = tf.keras.Sequential([
    # Çıktı: (4, 4, 64 * 8)
    G_block(out_channels=n_G*8, strides=1, padding="valid"),
    G_block(out_channels=n_G*4), # Çıktı: (8, 8, 64 * 4)
    G_block(out_channels=n_G*2), # Çıktı: (16, 16, 64 * 2)
    G_block(out_channels=n_G), # Çıktı: (32, 32, 64)
    # Çıktı: (64, 64, 3)
    tf.keras.layers.Conv2DTranspose(
        3, kernel_size=4, strides=2, padding="same", use_bias=False,
        activation="tanh") 
])
```

Üreticinin çıktı şeklini doğrulamak için 100 boyutlu bir saklı değişken oluşturalım.

```{.python .input}
x = np.zeros((1, 100, 1, 1))
net_G.initialize()
net_G(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((1, 100, 1, 1))
net_G(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((1, 1, 1, 100))
net_G(x).shape
```

## Ayrımcı

Ayrımcı, etkinleştirme işlevi olarak sızıntılı (leaky) ReLU kullanması dışında normal bir evrişimli ağdır. $\alpha \in[0, 1]$ verildiğinde, tanım şöyledir:

$$\textrm{sızıntılı ReLU}(x) = \begin{cases}x & \text{eğer}\ x > 0\\ \alpha x &\text{diğer türlü}\end{cases}.$$

Görüldüğü gibi, $\alpha = 0$ ise normal ReLU, $\alpha = 1$ ise bir birim fonksiyondur. $\alpha \in (0, 1)$ için, sızıntılı ReLU, negatif bir girdi için sıfır olmayan bir çıktı veren doğrusal olmayan bir fonksiyondur. Bir nöronun her zaman negatif bir değer verebileceği ve bu nedenle ReLU'nun gradyanı 0 olduğu için herhangi bir ilerleme kaydedemeyeceği "ölmekte olan ReLU" (dying ReLU) problemini çözmeyi amaçlamaktadır.

```{.python .input}
#@tab mxnet,pytorch
alphas = [0, .2, .4, .6, .8, 1]
x = d2l.arange(-2, 1, 0.1)
Y = [d2l.numpy(nn.LeakyReLU(alpha)(x)) for alpha in alphas]
d2l.plot(d2l.numpy(x), Y, 'x', 'y', alphas)
```

```{.python .input}
#@tab tensorflow
alphas = [0, .2, .4, .6, .8, 1]
x = tf.range(-2, 1, 0.1)
Y = [tf.keras.layers.LeakyReLU(alpha)(x).numpy() for alpha in alphas]
d2l.plot(x.numpy(), Y, 'x', 'y', alphas)
```

Ayrımcının temel bloğu, bir evrişim katmanı ve ardından bir toptan normalleştirme katmanı ve bir sızıntılı ReLU etkinleştirmesidir. Evrişim katmanının hiper parametreleri, üretici bloğundaki devrik evrişim katmanına benzer.

```{.python .input}
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

```{.python .input}
#@tab pytorch
class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

```{.python .input}
#@tab tensorflow
class D_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=4, strides=2, padding="same",
                 alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(out_channels, kernel_size,
                                             strides, padding, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha)
        
    def call(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

Varsayılan ayarlara sahip temel bir blok, girdilerin genişliğini ve yüksekliğini, :numref:`sec_padding` içinde gösterdiğimiz gibi, yarıya düşürecektir. Örneğin, $k_h = k_w = 4$ çekirdek, $s_h = s_w = 2$ uzun adım, $p_h = p_w = 1$ dolgu ve $n_h = n_w = 16$ girdi şekli verildiğinde, çıktının şekli şöyle olacaktır:

$$
\begin{aligned}
n_h^{'} \times n_w^{'} &= \lfloor(n_h-k_h+2p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+2p_w+s_w)/s_w\rfloor\\
  &= \lfloor(16-4+2\times 1+2)/2\rfloor \times \lfloor(16-4+2\times 1+2)/2\rfloor\\
  &= 8 \times 8 .\\
\end{aligned}
$$

```{.python .input}
x = np.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk.initialize()
d_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 16, 16, 3))
d_blk = D_block(20)
d_blk(x).shape
```

Ayrımcı üreticinin bir yansımasıdır.

```{.python .input}
n_D = 64
net_D = nn.Sequential()
net_D.add(D_block(n_D),   # Çıktı: (64, 32, 32)
          D_block(n_D*2),  # Çıktı: (64 * 2, 16, 16)
          D_block(n_D*4),  # Çıktı: (64 * 4, 8, 8)
          D_block(n_D*8),  # Çıktı: (64 * 8, 4, 4)
          nn.Conv2D(1, kernel_size=4, use_bias=False))  # Çıktı: (1, 1, 1)
```

```{.python .input}
#@tab pytorch
n_D = 64
net_D = nn.Sequential(
    D_block(n_D),  # Çıktı: (64, 32, 32)
    D_block(in_channels=n_D, out_channels=n_D*2),  # Çıktı: (64 * 2, 16, 16)
    D_block(in_channels=n_D*2, out_channels=n_D*4),  # Çıktı: (64 * 4, 8, 8)
    D_block(in_channels=n_D*4, out_channels=n_D*8),  # Çıktı: (64 * 8, 4, 4)
    nn.Conv2d(in_channels=n_D*8, out_channels=1,
              kernel_size=4, bias=False))  # Çıktı: (1, 1, 1)
```

```{.python .input}
#@tab tensorflow
n_D = 64
net_D = tf.keras.Sequential([
    D_block(n_D), # Çıktı: (32, 32, 64)
    D_block(out_channels=n_D*2), # Çıktı: (16, 16, 64 * 2)
    D_block(out_channels=n_D*4), # Çıktı: (8, 8, 64 * 4)
    D_block(out_channels=n_D*8), # Çıktı: (4, 4, 64 * 64)
    # Çıktı: (1, 1, 1)
    tf.keras.layers.Conv2D(1, kernel_size=4, use_bias=False)
])
```

Tek bir tahmin değeri elde etmek için son katmanda çıktı kanalı $1$ olan bir evrişim katmanı kullanır.

```{.python .input}
x = np.zeros((1, 3, 64, 64))
net_D.initialize()
net_D(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((1, 3, 64, 64))
net_D(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((1, 64, 64, 3))
net_D(x).shape
```

## Eğitim

Temel GAN, :numref:`sec_basic_gan`, ile karşılaştırıldığında, birbirlerine benzer olduklarından hem üretici hem de ayrımcı için aynı öğrenme oranını kullanıyoruz. Ek olarak, Adam'daki (:numref:`sec_adam`) $\beta_1$'yı $0.9$'dan $0.5$'e değiştiriyoruz. Üretici ve ayrımcı birbiriyle çekiştiği için, hızlı değişen gradyanlarla ilgilenmek için momentumun, ki geçmiş gradyanların üstel ağırlıklı hareketli ortalamasıdır, düzgünlüğünü azaltır. Ayrıca, rastgele üretilen `Z` gürültüsü bir 4B tensördür ve bu nedenle hesaplamayı hızlandırmak için GPU kullanırız.

```{.python .input}
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True, ctx=device)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True, ctx=device)
    trainer_hp = {'learning_rate': lr, 'beta1': 0.5}
    trainer_D = gluon.Trainer(net_D.collect_params(), 'adam', trainer_hp)
    trainer_G = gluon.Trainer(net_G.collect_params(), 'adam', trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # Bir dönem eğit
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.as_in_ctx(device), Z.as_in_ctx(device),
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Üretilen örnekleri göster
        Z = np.random.normal(0, 1, size=(21, latent_dim, 1, 1), ctx=device)
        # Yapay verileri N(0, 1) olarak normalleştirin
        fake_x = net_G(Z).transpose(0, 2, 3, 1) / 2 + 0.5
        imgs = np.concatenate(
            [np.concatenate([fake_x[i * 7 + j] for j in range(7)], axis=1)
             for i in range(len(fake_x)//7)], axis=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs.asnumpy())
        # Kayıpları göster
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # Bir dönem eğit
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Üretilen örnekleri göster
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # Yapay verileri N(0, 1) olarak normalleştirin
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
             for i in range(len(fake_x)//7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # Kayıpları göster
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    
    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    
    optimizer_hp = {"lr": lr, "beta_1": 0.5, "beta_2": 0.999}
    optimizer_D = tf.keras.optimizers.Adam(**optimizer_hp)
    optimizer_G = tf.keras.optimizers.Adam(**optimizer_hp)
    
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    
    for epoch in range(1, num_epochs + 1):
        # Bir dönem eğit
        timer = d2l.Timer()
        metric = d2l.Accumulator(3) # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(mean=0, stddev=1,
                                 shape=(batch_size, 1, 1, latent_dim))
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       d2l.update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)
            
        # Üretilen örnekleri göster
        Z = tf.random.normal(mean=0, stddev=1, shape=(21, 1, 1, latent_dim))
        # Yapay verileri N(0, 1) olarak normalleştirin
        fake_x = net_G(Z) / 2 + 0.5
        imgs = tf.concat([tf.concat([fake_x[i * 7 + j] for j in range(7)],
                                    axis=1) 
                          for i in range(len(fake_x) // 7)], axis=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # Kayıpları göster
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
```

Modeli sadece gösterim amaçlı az sayıda dönemle eğitiyoruz. Daha iyi performans için, `num_epochs` değişkeni daha büyük bir sayıya ayarlayabiliriz.

```{.python .input}
#@tab mxnet, pytorch
latent_dim, lr, num_epochs = 100, 0.005, 20
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
```

```{.python .input}
#@tab tensorflow
latent_dim, lr, num_epochs = 100, 0.0005, 40
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
```

## Özet

* DCGAN mimarisi, ayrımcı için dört evrişimli katmana ve üretici için dört "kesirli uzun adımlı" evrişimli katmana sahiptir.
* Ayrımcı, toptan normalleştirme (girdi katmanı hariç) ve sızıntılı ReLU etkinleştirmeleri olan 4 katman uzun adımlı evrişimlerdir.
* Sızıntılı ReLU, negatif bir girdi için sıfır olmayan bir çıktı veren doğrusal olmayan bir fonksiyondur. "Ölen ReLU" sorununu çözmeyi amaçlar ve gradyanların mimari boyunca daha kolay akmasına yardımcı olur.

## Alıştırmalar

1. Sızıntılı ReLU yerine standart ReLU etkinleştirmesi kullanırsak ne olur?
1. DCGAN'ı Fashion-MNIST'e uygulayın ve hangi kategorinin işe yarayıp hangilerinin yaramadığını görün.


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/409)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1083)
:end_tab:
