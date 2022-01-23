# Transpoze Konvolüsyon
:label:`sec_transposed_conv`

Bugüne kadar gördüğümüz CNN katmanları, evrimsel katmanlar (:numref:`sec_conv_layer`) ve havuzlama katmanları (:numref:`sec_pooling`) gibi, genellikle girdinin uzamsal boyutlarını (yükseklik ve genişlik) azaltır (azaltma) veya değişmeden tutar. Piksel düzeyinde sınıflandırılan semantik segmentasyonda, girdi ve çıktının mekansal boyutları aynı ise uygun olacaktır. Örneğin, bir çıkış pikselindeki kanal boyutu, girdi pikselinin sınıflandırma sonuçlarını aynı uzamsal konumda tutabilir. 

Bunu başarmak için, özellikle uzamsal boyutlar CNN katmanları tarafından azaltıldıktan sonra, ara özellik haritalarının mekansal boyutlarını artırabilecek (upsample) yapabilen başka bir CNN katmanlarını kullanabiliriz. Bu bölümde, tanıtacağız 
*de* kesirli kıvrımlı kıvrım* olarak da adlandırılan dönüşüm* :cite:`Dumoulin.Visin.2016`, 
evrişim ile aşağı örnekleme işlemlerini tersine çevirmek için.

```{.python .input}
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## Temel İşlem

Şimdilik kanalları görmezden gelerek, 1 adım ve dolgu olmadan temel transpoze edilmiş evrişim işlemiyle başlayalım. Bir $n_h \times n_w$ giriş tensör ve bir $k_h \times k_w$ çekirdeği verildiğini varsayalım. Çekirdek penceresini her satırda $n_w$ kez ve her sütundaki $n_h$ kez 1 adımıyla kaydırılması toplam $n_h n_w$ ara sonuç verir. Her ara sonuç sıfır olarak başlatılan bir $(n_h + k_h - 1) \times (n_w + k_w - 1)$ tensördür. Her ara tensörün hesaplanması için, giriş tensöründe bulunan her eleman çekirdek ile çarpılır, böylece sonuçta ortaya çıkan $k_h \times k_w$ tensör her ara tensörde bir bölümün yerini alır. Her ara tensördeki değiştirilen bölümün konumunun, hesaplama için kullanılan giriş tensöründe elemanın konumuna karşılık geldiğini unutmayın. Sonunda, çıktı üretmek için tüm ara sonuçlar toplanır. 

Örnek olarak, :numref:`fig_trans_conv`, $2\times 2$ giriş tensör için $2\times 2$ çekirdeği ile dönüştürülmüş evrimin nasıl hesaplandığını göstermektedir. 

![Transposed convolution with a $2\times 2$ kernel. The shaded portions are a portion of an intermediate tensor as well as the input and kernel tensor elements used for the  computation.](../img/trans_conv.svg)
:label:`fig_trans_conv`

Biz (**Bu temel transposed evrişim operasyonu uygulamak**) `trans_conv` Bir giriş matrisi için `X` ve bir çekirdek matrisi `K`.

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

Düzenli konvolüsyonun aksine (:numref:`sec_conv_layer`'te) bu *reduces* giriş elemanları çekirdek üzerinden, transpoze edilmiş konvolüsyon
*yayınlar* giriş elemanları 
çekirdek aracılığıyla, böylece girişten daha büyük bir çıktı üretir. :numref:`fig_trans_conv`'ten :numref:`fig_trans_conv`'ten `X` giriş tensörünü ve çekirdek tensörünü :numref:`fig_trans_conv`'ten temel iki boyutlu transpoze edilmiş evrişim işleminin [**doğrulamak**] için inşa edebiliriz.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```

Alternatif olarak, `X` ve çekirdek `K` girişi dört boyutlu tensör olduğunda, [**aynı sonuçlar elde etmek için yüksek seviyeli API'leri kullanabiliriz**].

```{.python .input}
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## [**Dolgu, Adım ve Çoklu Kanal**]

Dolgunun girişe uygulandığı düzenli konvolüsiyondan farklı olarak, transpoze edilmiş konvolüsyon içindeki çıkışa uygulanır. Örneğin, yükseklik ve genişliğin her iki tarafındaki dolgu numarasını 1 olarak belirtirken, ilk ve son satırlar ve sütunlar aktarılan konvolüsyon çıkışından kaldırılır.

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

Transpoze edilen konvolüsyonda, giriş için değil, ara sonuçlar için adımlar belirtilir (böylece çıkış). :numref:`fig_trans_conv`'ten itibaren aynı giriş ve çekirdek tensörlerinin kullanılması, adımın 1'den 2'ye değiştirilmesi, ara tensörlerin hem yüksekliğini hem de ağırlığını arttırır, dolayısıyla :numref:`fig_trans_conv_stride2`'teki çıkış tensörünü artırır. 

![Transposed convolution with a $2\times 2$ kernel with stride of 2. The shaded portions are a portion of an intermediate tensor as well as the input and kernel tensor elements used for the  computation.](../img/trans_conv_stride2.svg)
:label:`fig_trans_conv_stride2`

Aşağıdaki kod parçacığı, :numref:`fig_trans_conv_stride2`'te 2 adım için dönüştürülmüş evrişim çıktısını doğrulayabilir.

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

Birden fazla giriş ve çıkış kanalı için, dönüştürülmüş evrişim normal konvolüsyon ile aynı şekilde çalışır. Girdinin $c_i$ kanallara sahip olduğunu ve dönüştürülmüş evrimin her giriş kanalına bir $k_h\times k_w$ çekirdek tensör atadığını varsayalım. Birden fazla çıkış kanalı belirtildiğinde, her çıkış kanalı için bir $c_i\times k_h\times k_w$ çekirdeğine sahip olacağız. 

Tüm olarak, $\mathsf{X}$'yi $\mathsf{X}$'yi $f$ çıkış yapmak için $f$ çıktısını $f$ olarak beslersek ve $f$ ile aynı hiperparametrelere sahip transpoze edilmiş bir evrimsel katman oluşturursak, $\mathsf{X}$'deki kanal sayısı hariç, $\mathsf{X}$'deki kanal sayısı hariç, $g(Y)$ ile aynı şekle sahip olacaktır $\mathsf{X}$. Bu, aşağıdaki örnekte gösterilebilir.

```{.python .input}
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## [**Matris Transposition'a Bağlantı**]
:label:`subsec-connection-to-mat-transposition`

Transpoze edilen evrişim, matris transpozisyonundan sonra adlandırılır. Açıklamak için, önce matris çarpımlarını kullanarak kıvrımların nasıl uygulanacağını görelim. Aşağıdaki örnekte, $3\times 3$ giriş `X` ve $2\times 2$ evrişim çekirdeği `K` tanımlıyoruz ve `Y` evrişim çıktısını hesaplamak için `corr2d` işlevini kullanıyoruz.

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

Daha sonra, `K`'i, çok sayıda sıfır içeren seyrek ağırlık matrisi `W` olarak konvolüsyon çekirdeğini yeniden yazıyoruz. Ağırlık matrisinin şekli ($4$, $9$) olup, sıfır olmayan elemanların evrişim çekirdeği `K`'den geldiği yerdir.

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

9 uzunluğunda bir vektör elde etmek için giriş `X` satırını satır birleştirin. Daha sonra `W`'in matris çarpımı ve vektörleştirilmiş `X`, 4 uzunluğunda bir vektör verir. Yeniden şekillendirdikten sonra, yukarıdaki orijinal evrişim işleminden aynı sonucu `Y`'ü elde edebiliriz: matris çarpımlarını kullanarak kıvrımları uyguladık.

```{.python .input}
#@tab all
Y == d2l.matmul(W, d2l.reshape(X, -1)).reshape(2, 2)
```

Aynı şekilde, matris çarpımlarını kullanarak transpoze edilmiş kıvrımları da uygulayabiliriz. Aşağıdaki örnekte, $2 \times 2$ çıkış `Y`'ü yukarıdaki düzenli konvolüsyondan aktarılan evrime giriş olarak alıyoruz. Bu işlemi matrisleri çarparak uygulamak için, `W` ağırlık matrisini yeni şekil $(9, 4)$ ile aktarmamız yeterlidir.

```{.python .input}
#@tab all
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, -1)).reshape(3, 3)
```

Matrisleri çarparak evrimi uygulamayı düşünün. Bir giriş vektörü $\mathbf{x}$ ve bir ağırlık matrisi $\mathbf{W}$ göz önüne alındığında, evrimin ileri yayılma fonksiyonu, giriş ağırlık matrisi ile çarpılarak ve bir vektör $\mathbf{y}=\mathbf{W}\mathbf{x}$ çıktısıyla uygulanabilir. Geri yayılma zincir kuralını ve $\nabla_{\mathbf{x}}\mathbf{y}=\mathbf{W}^\top$'i izlediğinden, konvolüsiyonun geri yayılma fonksiyonu, girişini transpoze ağırlık matrisi $\mathbf{W}^\top$ ile çarpılarak uygulanabilir. Bu nedenle, transpoze edilmiş konvolüsyonel tabaka sadece ileri yayılma fonksiyonunu ve dönme tabakasının geri yayılma fonksiyonunu değiştirebilir: ileri yayılma ve geri yayılma fonksiyonları sırasıyla $\mathbf{W}^\top$ ve $\mathbf{W}$ ile giriş vektörünü çarpar. 

## Özet

* Çekirdek üzerinden giriş elemanlarını azaltan düzenli evrimin aksine, dönüştürülmüş evrişim çekirdek üzerinden giriş öğelerini yayınlar ve böylece girişten daha büyük bir çıktı üretir.
* $\mathsf{X}$'yi $\mathsf{X}$ çıkış yapmak için $f$ çıktısını $f$ ile evrimsel bir katmana beslersek ve $f$ ile aynı hiperparametrelere sahip bir transpoze evrimsel katman oluşturursak, $\mathsf{X}$'deki kanal sayısı olan çıkış kanallarının sayısı haricinde $\mathsf{X}$, $\mathsf{X}$ ile aynı şekle sahip olacaktır.
* Matris çarpımlarını kullanarak kıvrımları gerçekleştirebiliriz. Transpoze edilmiş evrimsel tabaka sadece ileri yayılma fonksiyonunu ve kıvrımlı tabakanın geri yayılma işlevini değiştirebilir.

## Egzersizler

1. :numref:`subsec-connection-to-mat-transposition`'te, `X` konvolüsyon girişi ve transpoze edilmiş konvolüsyon çıkışı `Z` aynı şekle sahiptir. Aynı değere sahipler mi? Neden?
1. Kıvrımları uygulamak için matris çarpımlarını kullanmak etkili midir? Neden?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1450)
:end_tab:
