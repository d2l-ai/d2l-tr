# Sayısal Kararlılık ve İlkleme
:label:`sec_numerical_stability`


Buraya kadar, uyguladığımız her model, parametrelerini önceden belirlenmiş bazı dağılımlara göre ilklememizi gerektirdi. Şimdiye kadar, bu seçimlerin nasıl yapıldığının ayrıntılarını gözden ardı ederek ilkleme düzenini doğal kabul ettik. Bu seçimlerin özellikle önemli olmadığı izlenimini bile almış olabilirsiniz. Aksine, ilkleme düzeninin seçimi sinir ağı öğrenmesinde önemli bir rol oynar ve sayısal kararlılığı korumak için çok önemli olabilir. Dahası, bu seçimler doğrusal olmayan etkinleştirme fonksiyonunun seçimi ile ilginç şekillerde bağlanabilir. Hangi işlevi seçtiğimiz ve parametreleri nasıl ilklettiğimiz, optimizasyon algoritmamızın ne kadar hızlı yakınsadığını belirleyebilir. Buradaki kötü seçimler, eğitim sırasında patlayan veya kaybolan gradyanlarla karşılaşmamıza neden olabilir. Bu bölümde, bu konuları daha ayrıntılı olarak inceliyoruz ve derin öğrenmedeki kariyeriniz boyunca yararlı bulacağınız bazı yararlı buluşsal yöntemleri tartışıyoruz.

## Kaybolan ve Patlayan Gradyanlar

$L$ katmanlı $\mathbf{x}$ girdili ve $\mathbf{o}$ çıktılı derin bir ağ düşünün. Bir $f_l$ dönüşümü ile tanımlanan $\mathbf{W}^{(l)}$ ağırlıklarıyla parametreleştirilen her bir katman $l$ ile, ki gizli değişkenleri $\mathbf{h}^{(l)}$'dir ($\mathbf{h}^{(0)} = \mathbf{x}$), ağımız şu şekilde ifade edilebilir:
$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ ve böylece } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$


Tüm gizli değişkenler ve girdiler vektör ise, $\mathbf{o}$ gradyanını herhangi bir $\mathbf{W}^{(l)}$ parametre kümesine göre aşağıdaki gibi yazabiliriz:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$


Başka bir deyişle, bu gradyan $L-l$ matrisleri 
$\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$ ve $\mathbf{v}^{(l)}$ gradyan vektörünün çarpımıdır. Bu nedenle, çok fazla olasılığı bir araya getirirken sıklıkla ortaya çıkan aynı sayısal küçümenlik sorunlarına duyarlıyız. Olasılıklarla uğraşırken, yaygın bir hile, logaritma-uzayına geçmektir, yani basıncı mantisten sayısal temsilin üssüne kaydırmaktır. Ne yazık ki, yukarıdaki sorunumuz daha ciddidir: Başlangıçta $\mathbf{M}^{(l)}$ matrisleri çok çeşitli özdeğerlere sahip olabilir. Küçük veya büyük olabilirler ve çarpımları *çok büyük* veya *çok küçük* olabilir.

Kararsız gradyanların yarattığı riskler sayısal temsilin ötesine geçer. Tahmin edilemeyen büyüklükteki gradyanlar, optimizasyon algoritmalarımızın kararlılığını da tehdit eder: Ya (i) aşırı büyük olan, modelimizi yok eden (*patlayan gradyan* problemi); veya (ii) aşırı derecede küçük (*kaybolan gradyan* problemi), parametreler neredeyse hiç hareket etmediği için öğrenmeyi imkansız kılan güncellemeler.

### (**Kaybolan Gradyanlar**)

Kaybolan gradyan sorununa neden olan sık karşılaşılan bir zor durum, her katmanın doğrusal işlemlerinin ardından eklenen $\sigma$ etkinleştirme işlevinin seçimidir. Tarihsel olarak, sigmoid işlevi $1/(1 + \exp(-x))$ (bkz. :numref:`sec_mlp`), bir eşikleme işlevine benzediği için popülerdi. Öncül yapay sinir ağları biyolojik sinir ağlarından ilham aldığından, ya *tamamen* ateşleyen ya da *hiç* ateşlemeyen (biyolojik nöronlar gibi) nöronlar fikri çekici görünüyordu. Neden yok olan gradyanlara neden olabileceğini görmek için sigmoide daha yakından bakalım.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Gördüğünüz gibi (**sigmoidin gradyanı, girdileri hem büyük hem de küçük olduklarında kaybolur**). Dahası, birçok katmanda geri yayma yaparken, birçok sigmoidin girdilerinin sıfıra yakın olduğu altın bölgede (Goldilocks) olmadıkça, tüm çarpımın gradyanları kaybolabilir. Ağımız birçok katmana sahip olduğunda, dikkatli olmadıkça, gradyan muhtemelen herhangi bir katmanda kesilecektir. Gerçekten de, bu problem derin ağ eğitiminda başa bela oluyordu. Sonuç olarak, daha kararlı (ancak sinirsel olarak daha az makul olan) ReLU'lar, uygulayıcıların varsayılan seçimi olarak öne çıktı.


### [**Patlayan Gradyanlar**]

Tersi problem de, gradyanlar patladığında, benzer şekilde can sıkıcı olabilir. Bunu biraz daha iyi açıklamak için, 100 tane Gauss'luk rasgele matrisler çekip bunları bir başlangıç matrisiyle çarpıyoruz. Seçtiğimiz ölçek için ($\sigma^2=1$ varyans seçimi ile), matris çarpımı patlar. Bu, derin bir ağın ilklenmesi nedeniyle gerçekleştiğinde, yakınsama için bir gradyan iniş optimize ediciyi alma şansımız yoktur.

```{.python .input}
M = np.random.normal(size=(4, 4))
print('Tek bir matris', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('100 matrisi carptiktan sonra', M)
```

```{.python .input}
#@tab pytorch
M = torch.normal(0, 1, size=(4,4))
print('Tek bir matris \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('100 matrisi carptiktan sonra\n',M)
```

```{.python .input}
#@tab tensorflow
M = tf.random.normal((4, 4))
print('Tek bir matris \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('100 matrisi carptiktan sonra\n', M.numpy())
```

### Bakışımı (Simetriyi) Kırma

Sinir ağı tasarımındaki bir başka sorun, parametrelendirilmesinde bulunan simetridir. Bir gizli katman ve iki birim içeren basit bir MLP'miz olduğunu varsayalım. Bu durumda, ilk katmanın $\mathbf{W}^{(1)}$ ağırlıklarını ve aynı şekilde aynı işlevi elde etmek için çıktı katmanının ağırlıklarını devrişebiliriz (permütasyon). İlk gizli birimi veya ikinci gizli birimi türev alma arasında bir fark yoktur. Başka bir deyişle, her katmanın gizli birimleri arasında devrişim bakışımımız var.

Bu teorik bir can sıkıntısından daha fazlasıdır. Yukarıda bahsedilen, iki gizli birimi olan tek-katmanlı MLP'yi düşünün. Örnek olarak, çıktı katmanının iki gizli birimi yalnızca bir çıktı birimine dönüştürdüğünü varsayalım. Bir sabit $c$ için gizli katmanın tüm parametrelerini $\mathbf{W}^{(1)} = c$ olarak ilkletirsek ne olacağını hayal edin. Bu durumda, ileriye doğru yayma sırasında gizli birim aynı girdileri ve parametreleri alarak aynı aktivasyonu üretir, ki o da çıktı birimine beslenir. Geri yayma sırasında, çıktı biriminin $\mathbf{W}^{(1)}$ parametrelerine göre türevini almak, öğelerinin tümü aynı değeri alan bir gradyan verir. Bu nedenle, gradyan tabanlı yinelemeden sonra (örneğin, minigrup rasgele gradyan inişi), $\mathbf{W}^{(1)}$ öğesinin tüm öğeleri hala aynı değeri alır.

Bu yinelemeler, *bakışımı asla kendi başına bozmaz* ve ağın ifade gücünü asla gerçekleyemeyebiliriz. Gizli katman, yalnızca tek bir birimi varmış gibi davranacaktır. Minigrup rasgele gradyan inişi bu bakışımı bozmasa da, hattan düşürme düzenlileştirmesinin bozacağını unutmayın!


## Parametre İlkleme

Yukarıda belirtilen sorunları ele almanın---ya da en azından hafifletmenin---bir yolu dikkatli ilkletmektir. Optimizasyon sırasında ek özen ve uygun düzenlileştirme, kararlılığı daha da artırabilir.

### Varsayılan İlkletme

Önceki bölümlerde, örneğin :numref:`sec_linear_concise` içindeki gibi, ağırlıklarımızın değerlerini ilkletmek için normal dağılım kullandık. İlkleme yöntemini belirtmezsek, çerçeve, pratikte genellikle orta düzey problem boyutları için iyi çalışan varsayılan bir rastgele ilkleme yöntemi kullanacaktır.




### Xavier İlkletmesi
:label:`subsec_xavier`

Bir çıktının (örneğin, bir gizli değişken) $o_{i}$ tam bağlı bir katman *doğrusal olmayanlar* için ölçek dağılımına bakalım.
Bu katman için $n_\mathrm{in}$ tane $x_j$ girdisi ve bunlarla ilişkili ağırlıkları $w_{ij}$ ile,  çıktı şu şekilde verilir:

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

$w_{ij}$ ağırlıklarının tümü aynı dağılımdan bağımsız olarak çekilir. Dahası, bu dağılımın sıfır ortalamaya ve $\sigma^2$ varyansına sahip olduğunu varsayalım. Bu, dağılımın Gaussian olması gerektiği anlamına gelmez, sadece ortalama ve varyansın var olması gerektiği anlamına gelir. Şimdilik, $x_j$ katmanındaki girdilerin de sıfır ortalamaya ve $\gamma^2$ varyansına sahip olduğunu ve $w_{ij}$'den ve birbirinden bağımsız olduklarını varsayalım. Bu durumda, $o_i$'nin ortalamasını ve varyansını şu şekilde hesaplayabiliriz:

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Varyansı sabit tutmanın bir yolu, $n_\mathrm{in} \sigma^2 = 1$ diye ayarlamaktır. Şimdi geri yaymayı düşünün. Orada, çıktıya yakın katmanlardan yayılmakta olan gradyanlarla da olsa benzer bir sorunla karşı karşıyayız. İleri yaymayla aynı mantığı kullanarak, gradyanların varyansının $n_\mathrm{out} \sigma^2 = 1$, ki $n_\mathrm{out}$ burada bu katmanın çıktı sayısıdır, olmadıkça patlayabileceğini görüyoruz. Bu bizi bir ikilemde bırakıyor: Her iki koşulu aynı anda karşılayamayız. Bunun yerine, sadece şunlara uymaya çalışırız:

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ veya eşdeğeri }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

Bu, yaratıcılarının ilk yazarının adını taşıyan, artık standart ve pratik olarak faydalı *Xavier ilkletmesi*nin altında yatan mantıktır :cite:`Glorot.Bengio.2010`. Tipik olarak, Xavier ilkletmesi sıfır ortalama ve $\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$ varyansına sahip bir Gauss dağılımından ağırlıkları örnekler. Aynı zamanda, tekdüze bir dağılımdan ağırlıkları örneklerken, Xavier'in sezgisini varyansı seçecek şekilde uyarlayabiliriz. $U(-a, a)$ tekdüze dağılımının $\frac{a^2}{3}$ varyansına sahip olduğuna dikkat edin. $\frac{a^2}{3}$'ı $\sigma^2$'daki koşulumuza eklemek, şu ilkletme önerisini verir:

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

Yukarıdaki matematiksel akıl yürütmede doğrusal olmayanların var olmadığı varsayımı, sinir ağlarında kolayca ihlal edilebilse de, Xavier ilkletme yönteminin pratikte iyi çalıştığı ortaya çıkıyor.


### Daha Fazlası

Yukarıdaki mantık, parametre ilklendirmesine yönelik modern yaklaşımların yüzeyine ışık tutar. Bir derin öğrenme çerçevesi genellikle bir düzineden fazla farklı buluşsal yöntem uygular. Dahası, parametre ilkletme, derin öğrenmede temel araştırma için sıcak bir alan olmaya devam ediyor. Bunlar arasında bağlı (paylaşılan) parametreler, süper çözünürlük, dizi modelleri ve diğer durumlar için özelleştirilmiş buluşsal yöntemler bulunur. Örneğin, Xiao ve ark. dikkatle tasarlanmış bir ilkletme yöntemi kullanarak mimari hileler olmadan 10000 katmanlı sinir ağlarını eğitme olasılığını gösterdi :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`.

Konu ilginizi çekiyorsa, bu modülün önerdiklerine derinlemesine dalmanızı, her buluşsal yöntemi öneren ve analiz eden çalışmaları okumanızı ve ardından konuyla ilgili en son yayınlarda gezinmenizi öneririz. Belki de zekice bir fikre rastlarsınız, hatta icat edersiniz ve derin öğrenme çerçevelerinde bir uygulamaya katkıda bulunursunuz.

## Özet

* Kaybolan ve patlayan gradyanlar, derin ağlarda yaygın sorunlardır. Gradyanların ve parametrelerin iyi kontrol altında kalmasını sağlamak için parametre ilklemeye büyük özen gösterilmesi gerekir.
* İlk gradyanların ne çok büyük ne de çok küçük olmasını sağlamak için ilkleme buluşsal yöntemlerine ihtiyaç vardır.
* ReLU etkinleştirme fonksiyonları, kaybolan gradyan problemini azaltır. Bu yakınsamayı hızlandırabilir.
* Rastgele ilkleme, optimizasyondan önce bakışımın bozulmasını sağlamak için bir araçtır.
* Xavier ilkletme, her katman için herhangi bir çıktının varyansının girdi sayısından etkilenmediğini ve herhangi bir gradyanın varyansının çıktı sayısından etkilenmediğini öne sürer.

## Alıştırmalar

1. Bir sinir ağının, bir MLP'nin, katmanlarında yer değiştirme bakışımının yanısıra kırılma gerektiren bakışım sergileyebileceği başka durumlar tasarlayabilir misiniz?
1. Doğrusal regresyonda veya softmaks regresyonunda tüm ağırlık parametrelerini aynı değere ilkleyebilir miyiz?
1. İki matrisin çarpımının özdeğerlerindeki analitik sınırlara bakınız. Bu, gradyanların iyi şartlandırılmasının sağlanması konusunda size ne anlatıyor?
1. Bazı terimlerin ıraksadığını biliyorsak, bunu sonradan düzeltebilir miyiz? İlham almak için katmanlı uyarlanabilir oran ölçekleme makalesine bakınız :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/235)
:end_tab:
