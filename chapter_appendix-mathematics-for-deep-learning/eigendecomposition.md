# Özayrışmalar
:label:`sec_eigendecompositions`

Özdeğerler genellikle doğrusal cebiri incelerken karşılaşacağımız en yararlı kavramlardan biridir, ancak yeni başlayanlar olarak önemlerini gözden kaçırmak kolaydır.
Aşağıda, özayrışmayı tanıtıyoruz ve neden bu kadar önemli olduğuna dair bir fikir aktarmaya çalışıyoruz.

Aşağıdaki girdileri içeren bir $A$ matrisimiz olduğunu varsayalım:

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

Herhangi bir $\mathbf{v} = [x, y]^\top$ vektörüne $A$ uygularsak, bir $\mathbf{A}\mathbf{v} = [2x, -y]^\top$ vektörü elde ederiz.
Bunun sezgisel bir yorumu var: Vektörü $x$ yönünde iki kat geniş olacak şekilde uzatın ve ardından $y$ yönünde ters çevirin.

Ancak, bir şeyin değişmeden kaldığı *bazı* vektörler vardır.
Yani $[1, 0]^\top$, $[2, 0]^\top$'e ve $[0, 1]^\top$, $[0, -1]^\top$'e gönderilir.
Bu vektörler hala aynı doğrudadır ve tek değişiklik, matrisin onları sırasıyla $2$ ve $-1$ çarpanı ile genişletmesidir.
Bu tür vektörlere *özvektörler* diyoruz ve bunların uzatıldıkları çarpan da *özdeğerler*dir.

Genel olarak, bir $\lambda$ sayısı ve şöyle bir $\mathbf{v}$ vektörü bulabilirsek

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
$$

$\mathbf{v}$ $A$ için bir özvektör ve $\lambda$ bir özdeğerdir deriz.

## Özdeğerleri Bulma
Onları nasıl bulacağımızı anlayalım.
Her iki taraftan $\lambda \mathbf{v}$ çıkararak ve ardından vektörü dışarıda bırakarak, yukarıdakinin şuna eşdeğer olduğunu görürüz:

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

:eqref:`eq_eigvalue_der` denkleminin gerçekleşmesi için, $(\mathbf{A} - \lambda \mathbf{I})$'nın bir yönü sıfıra kadar sıkıştırması gerektiğini görüyoruz, bu nedenle tersinir değildir ve bu nedenle determinant sıfırdır.
Böylece, *özdeğerleri*, $\lambda$ değerinin ne zaman $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$ olduğunu bularak bulabiliriz.
Özdeğerleri bulduktan sonra, ilişkili *özvektör(leri)* bulmak için $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$'yı çözebiliriz.

### Bir örnek
Bunu daha zorlu bir matrisle görelim

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}.
$$

$\det(\mathbf{A}-\lambda \mathbf{I}) = 0$ olarak düşünürsek, bunun $0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$ polinom denklemine eşdeğer olduğunu görürüz.
Böylece, iki özdeğer $4$ ve $1$'dir.
İlişkili vektörleri bulmak için bunu çözmemiz gerekir:

$$
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \text{ve} \;
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

Bunu sırasıyla $[1, -1]^\top$ ve $[1, 2]^\top$ vektörleriyle çözebiliriz.

Bunu yerleşik `numpy.linalg.eig` rutinini kullanarak kodda kontrol edebiliriz.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64),
          eigenvectors=True)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

`numpy` kütüphanesinin özvektörleri bir uzunluğunda normalleştirdiğini, oysa bizimkileri keyfi uzunlukta kabul ettiğimizi unutmayın.
Ek olarak, işaret seçimi keyfidir.
Bununla birlikte, hesaplanan vektörler, aynı özdeğerlerle elle bulduklarımıza paraleldir.

## Matris Ayrıştırma
Önceki örnek ile bir adım daha devam edelim. 

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

Sütunları $\mathbf{A}$ matrisinin özvektörleri olduğu matris olsun.

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

Köşegen üzerinde ilişkili özdeğerleri olan matris olsun.
Özdeğerlerin ve özvektörlerin tanımı bize şunu söyler:

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

$W$ matrisi ters çevrilebilir, bu yüzden her iki tarafı da sağdan $W^{-1}$ ile çarpabiliriz ve görürüz ki:

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

Bir sonraki bölümde bunun bazı güzel sonuçlarını göreceğiz, ancak şimdilik sadece, doğrusal olarak bağımsız özvektörlerin tam bir topluluğunu bulabildiğimiz sürece böyle bir ayrışmanın var olacağını bilmemiz gerekiyor (böylece $W$ tersinirdir).

## Özayrışmalar Üzerinde İşlemler
Özayrışmalar, :eqref:`eq_eig_decomp`, ilgili güzel bir şey, genellikle karşılaştığımız birçok işlemi özayrışmalar açısından temiz bir şekilde yazabilmemizdir.
İlk örnek olarak şunları düşünün:

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\text{$n$ kere}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\text{$n$ kere}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\text{$n$ kere}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

Bu bize, bir matrisin herhangi bir pozitif kuvveti için, özayrışmasının özdeğerlerin sadece aynı kuvvete yükseltilmesiyle elde edildiğini söyler.
Aynısı negatif kuvvetler için de gösterilebilir, bu nedenle bir matrisi tersine çevirmek istiyorsak, yapmamız gereken yalnızca

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

veya başka bir deyişle, her bir özdeğeri ters çevirin.
Bu, her özdeğer sıfır olmadığı sürece işe yarayacaktır, bu nedenle tersinebilirin sıfır özdeğer olmamasıyla aynı olduğunu görürüz.

Aslında, ek çalışma şunu gösterebilir: Eğer $\lambda_1, \ldots, \lambda_n$  bir matrisin özdeğerleriyse, o zaman o matrisin determinantı

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

veya tüm özdeğerlerin çarpımıdır.
Bu sezgisel olarak mantıklıdır, çünkü $\mathbf{W}$ ne kadar esnetme yaparsa, $W^{-1}$ bunu geri alır, dolayısıyla sonunda gerçekleşen tek uzatma köşegen matris $\boldsymbol{\Sigma}$ ile çarpma yoluyla olur, ki o da köşegen elemanların çarpımına göre hacimleri uzatır.

Son olarak, kertenin matrisinizin en büyük doğrusal olarak bağımsız sütunlarının sayısı olduğunu hatırlayın.
Özayrışmayı yakından inceleyerek, kertenin $\mathbf{A}$'nın sıfır olmayan özdeğerlerinin sayısıyla aynı olduğunu görebiliriz.

Örnekler devam edebilirdi, ancak umarız ki mesele açıktır: Özayrışmalar, birçok doğrusal-cebirsel hesaplamayı basitleştirebilir ve birçok sayısal (numerik) algoritmanın ve doğrusal cebirde yaptığımız analizlerin çoğunun altında yatan temel bir işlemdir.

## Simetrik Matrislerin Özayrışmaları
Yukarıdaki işlemin çalışması için yeterli doğrusal olarak bağımsız özvektör bulmak her zaman mümkün değildir. Örneğin matris

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

tek bir özvektöre, $(1, 0)^\top$, sahiptir.
Bu tür matrisleri işlemek için, ele alabileceğimizden daha gelişmiş tekniklere ihtiyacımız var (Jordan Normal Formu veya Tekil Değer Ayrıştırması gibi).
Dikkatimizi sık sık tam bir özvektörler kümesinin varlığını garanti edebileceğimiz matrislere sınırlamamız gerekecek.

En sık karşılaşılan aile, $\mathbf{A} = \mathbf{A}^\top$ olan *simetrik matrislerdir*.
Bu durumda, $W$'i bir *dikgen (ortogonal) matris* olarak alabiliriz — sütunlarının tümü birbirine dik açıda birim uzunluklu vektörler olan bir matris, burada $\mathbf{W}^\top = \mathbf{W}^{-1}$'dir - ve tüm özdeğerler gerçel olacaktır.
Böylece, bu özel durumda :eqref:`eq_eig_decomp` denklemini şöyle yazabiliriz 

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Gershgorin Çember Teoremi
Özdeğerlerle sezgisel olarak akıl yürütmek genellikle zordur.
Rasgele bir matris sunulursa, özdeğerleri hesaplamadan ne oldukları hakkında söylenebilecek çok az şey vardır.
Bununla birlikte, en büyük değerler köşegen üzerindeyse, iyi bir tahmin yapmayı kolaylaştıran bir teorem vardır.

$\mathbf{A} = (a_{ij})$ herhangi bir ($n\times n$) kare matris olsun.
Şöyle tanımlayalım: $r_i = \sum_{j \neq i} |a_{ij}|$.
$\mathcal{D}_i$ karmaşık düzlemde $a_{ii}$ merkezli $r_i$ yarıçaplı diski temsil etsin.
Daha sonra, $\mathbf{A}$'nın her özdeğeri $\mathcal{D}_i$'den birinin içinde bulunur.

Bunu açmak biraz zaman alabilir, o yüzden bir örneğe bakalım.
Şu matrisi düşünün:

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

Elimizde $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ ve $r_4 = 0.9$ var.
Matris simetriktir, bu nedenle tüm özdeğerler gerçeldir.
Bu, tüm özdeğerlerimizin aşağıdaki aralıklardan birinde olacağı anlamına gelir.

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$

Sayısal hesaplamanın gerçekleştirilmesi, özdeğerlerin yaklaşık $0.99$, $2.97$, $4.95$, $9.08$ olduğunu ve rahatlıkla sağlanan aralıklar içinde olduğunu gösterir.

```{.python .input}
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

Bu şekilde, özdeğerler yaklaşık olarak tahmin edilebilir ve köşegenin diğer tüm öğelerden önemli ölçüde daha büyük olması durumunda yaklaşımlar oldukça doğru olacaktır.

Bu küçük bir şey, ancak özayrışma gibi karmaşık ve incelikli bir konuda, yapabileceğimiz herhangi bir sezgisel kavrayışa sahip olmak iyidir.

## Yararlı Bir Uygulama: Yinelenen Eşlemelerin Gelişimi

Artık özvektörlerin prensipte ne olduğunu anladığımıza göre, bunların sinir ağı davranışının merkezinde olan bir problemin derinlemesine anlaşılmasını sağlamak için nasıl kullanılabileceklerini görelim: Uygun ağırlık ilklenmesi.

### Uzun Vadeli Davranış Olarak Özvektörler

Derin sinir ağlarının ilklenmesinin tam matematiksel araştırması, metnin kapsamı dışındadır, ancak özdeğerlerin bu modellerin nasıl çalıştığını görmemize nasıl yardımcı olabileceğini anlamak için burada bir basit örnek sürümünü görebiliriz.
Bildiğimiz gibi, sinir ağları, doğrusal olmayan işlemlerle doğrusal dönüşüm katmanlarını serpiştirerek çalışır.
Burada basitleştirmek için, doğrusal olmayanlığın olmadığını ve dönüşümün tek bir tekrarlanan matris işlemi $A$ olduğunu varsayacağız, böylece modelimizin çıktısı şu şekildedir:

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

Bu modeller ilklendirildiğinde, $A$ Gauss girdileri olan rastgele bir matris olarak alınır, bu yüzden onlardan birini yapalım.
Somut olmak gerekirse, ortalama sıfır, değişinti (varyans) bir olan Gauss dağılımından gelen bir $5\times 5$ matrisi ile başlıyoruz.

```{.python .input}
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### Rastgele Verilerde Davranış
Oyuncak modelimizde basitlik sağlamak için, $\mathbf{v}_{in}$ ile beslediğimiz veri vektörünün rastgele beş boyutlu bir Gauss vektörü olduğunu varsayacağız.
Ne olmasını istediğimizi düşünelim.
Bağlam için, bir imge gibi girdi verilerini, imgenin bir kedi resmi olma olasılığı gibi bir tahmine dönüştürmeye çalıştığımız genel bir makine öğrenmesi problemini düşünelim.
Tekrarlanan $\mathbf{A}$ uygulaması rastgele bir vektörü çok uzun olacak şekilde esnetirse, o zaman girdideki küçük değişiklikler çıktıdaki büyük değişikliklere yükseltilir --- girdi imgesindeki küçük değişiklikler çok farklı tahminlere yol açar.
Bu doğru görünmüyor!

Diğer taraftan, $\mathbf{A}$ rasgele vektörleri daha kısa olacak şekilde küçültürse, o zaman birçok katmandan geçtikten sonra, vektör esasen hiçbir şeye (sıfıra yakın) küçülür ve çıktı girdiye bağlı olmaz. Bu da açıkça doğru değil!

Çıktımızın girdimize bağlı olarak değiştiğinden emin olmak için büyüme ve bozulma arasındaki dar çizgide yürümemiz gerekir, ancak çok fazla değil!

$\mathbf{A}$ matrisimizi rastgele bir girdi vektörüyle tekrar tekrar çarptığımızda ne olacağını görelim ve normunu takip edelim.

```{.python .input}
# `A`'yı tekrar tekrar uyguladıktan sonra normların dizisini hesapla
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# `A`'yı tekrar tekrar uyguladıktan sonra normların dizisini hesapla
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# `A`'yı tekrar tekrar uyguladıktan sonra normların dizisini hesapla
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Norm kontrolsüz bir şekilde büyüyor!
Nitekim bölüm listesini alırsak, bir desen göreceğiz.

```{.python .input}
# Normların ölçeklendirme çarpanını hesapla
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Normların ölçeklendirme çarpanını hesapla
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Normların ölçeklendirme çarpanını hesapla
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

Yukarıdaki hesaplamanın son kısmına bakarsak, rastgele vektörün `1.974459321485[...]` çarpanı ile esnediğini görürüz, burada son kısım biraz kayar, ancak esneme çarpanı sabittir.

### Özvektörlerle İlişkilendirme

Özvektörlerin ve özdeğerlerin, bir şeyin gerildiği (esnetildiği) miktara karşılık geldiğini gördük, ancak bu, belirli vektörler ve belirli gerilmeler içindi.
$\mathbf{A}$ için ne olduklarına bir göz atalım.
Burada bir uyarı: Hepsini görmek için karmaşık sayılara gitmemiz gerekeceği ortaya çıkıyor.
Bunları esnemeler ve dönüşler olarak düşünebilirsiniz.
Karmaşık sayının normunu (gerçek ve sanal kısımların karelerinin toplamının karekökü) alarak, bu esneme çarpanını ölçebiliriz. Bunları da sıralayabiliriz.

```{.python .input}
# Özdeğerleri hesapla
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Özdeğerleri hesapla
eigs = torch.eig(A)[0][:,0].tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Özdeğerleri hesapla
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### Bir Gözlem

Burada biraz beklenmedik bir şey görüyoruz: Daha önce rasgele bir vektöre uzun vadeli gerilmesi için $\mathbf{A}$ matrisimizi uygularken tanımladığımız sayı *tam olarak* (on üç ondalık basamağa kadar doğru!) $\mathbf{A}$'nın en büyük özdeğeridir.
Bu açıkça bir tesadüf değil!

Ama şimdi geometrik olarak ne olduğunu düşünürsek, bu mantıklı gelmeye başlar. Rastgele bir vektör düşünün.
Bu rasgele vektör her yönü biraz işaret ediyor, bu nedenle özellikle en büyük özdeğerle ilişkili $\mathbf{A}$ özvektörüyle çok az olsa bile bir miktar aynı yönü gösteriyor.
Bu o kadar önemlidir ki *ana (esas) özdeğer* ve *ana (esas) özvektör* olarak adlandırılır.
$\mathbf{A}$'yı uyguladıktan sonra, rastgele vektörümüz her olası özvektörle ilişkili olduğu gibi mümkün olan her yönde gerilir, ancak en çok bu özvektörle ilişkili yönde esnetilir.
Bunun anlamı, $A$'da uygulandıktan sonra, rasgele vektörümüzün daha uzun olması ve ana özvektör ile hizalanmaya daha yakın bir yönü göstermesidir.
Matrisi birçok kez uyguladıktan sonra, ana özvektörle hizalama gittikçe daha yakın hale gelir, ta ki tüm pratik amaçlar için rastgele vektörümüz temel özvektöre dönüştürülene kadar!
Aslında bu algoritma, bir matrisin en büyük özdeğerini ve özvektörünü bulmak için *kuvvet yinelemesi* olarak bilinen şeyin temelidir. Ayrıntılar için, örneğin, bkz. :cite:`Van-Loan.Golub.1983`.

### Normalleştirmeyi (Düzgelemeyi) Düzeltme

Şimdi, yukarıdaki tartışmalardan, rastgele bir vektörün esnetilmesini veya ezilmesini istemediğimiz sonucuna vardık, rastgele vektörlerin tüm süreç boyunca yaklaşık aynı boyutta kalmasını istiyoruz.
Bunu yapmak için, şimdi matrisimizi bu ana özdeğere göre yeniden ölçeklendiriyoruz, böylece en büyük özdeğer şimdi eskinin yerine sadece bir olur.
Bakalım bu durumda ne olacak.

```{.python .input}
# `A` matrisini yeniden ölçeklendir
A /= norm_eigs[-1]

# Aynı deneyi tekrar yapın
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# `A` matrisini yeniden ölçeklendir
A /= norm_eigs[-1]

# Aynı deneyi tekrar yapın
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# `A` matrisini yeniden ölçeklendir
A /= norm_eigs[-1]

# Aynı deneyi tekrar yapın
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Aynı zamanda ardışık normlar arasındaki oranı daha önce olduğu gibi çizebiliriz ve gerçekten dengelendiğini görebiliriz.

```{.python .input}
# Oranı da çiz
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Oranı da çiz
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Oranı da çiz
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## Sonuçlar

Şimdi tam olarak ne umduysak görüyoruz!
Matrisleri ana özdeğere göre normalleştirdikten sonra, rastgele verilerin eskisi gibi patlamadığını görüyoruz, bunun yerine nihai belirli bir değerde dengelenirler.
Bunları ilk ana özdeğerlerden yapabilmek güzel olurdu ve matematiğine derinlemesine bakarsak, bağımsız, ortalaması sıfır varyansı bir olan Gauss dağılımlı girdileri olan büyük bir rastgele matrisin en büyük özdeğerinin, ortalamada yaklaşık $\sqrt{n}$ veya bizim durumumuzda $\sqrt{5}\approx  2.2$ civarında olduğunu görebiliriz; bu *dairesel yasa* olarak bilinen büyüleyici bir gerçekten kaynaklanır :cite:`Ginibre.1965`.
Rastgele matrislerin özdeğerlerinin (ve tekil değerler olarak adlandırılan ilgili bir konunun) arasındaki ilişkinin, şu adreste :cite:`Pennington.Schoenholz.Ganguli.2017` ve sonraki çalışmalarda tartışıldığı gibi sinir ağlarının uygun şekilde ilklendirilmesiyle derin bağlantıları olduğu gösterilmiştir.

## Özet
* Özvektörler, yön değiştirmeden bir matris tarafından esnetilen vektörlerdir.
* Özdeğerler, özvektörlerin matris uygulamasıyla gerildikleri miktardır.
* Bir matrisin özayrışımı, birçok işlemin özdeğerler üzerindeki işlemlere indirgenmesine izin verebilir.
* Gershgorin Çember Teoremi, bir matrisin özdeğerleri için yaklaşık değerler sağlayabilir.
* Yinelenen matris kuvvetlerinin davranışı, öncelikle en büyük özdeğerin boyutuna bağlıdır. Bu anlayış, sinir ağı ilkleme teorisinde birçok uygulamaya sahiptir.

## Alıştırmalar
1. Aşağıdaki matrisin özdeğerleri ve özvektörleri nedir?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
$$
1.  Aşağıdaki matrisin özdeğerleri ve özvektörleri nelerdir ve bu örnekte bir öncekine kıyasla garip olan nedir?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}
$$
1. Özdeğerleri hesaplamadan, aşağıdaki matrisin en küçük özdeğerinin $0.5$'den az olması mümkün müdür? *Not*: Bu problemi kafanızda yapılabilirsiniz.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1087)
:end_tab:
