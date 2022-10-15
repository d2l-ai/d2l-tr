# Geometri ve Doğrusal Cebirsel İşlemler
:label:`sec_geometry-linear-algebraic-ops`

:numref:`sec_linear-algebra` içinde, doğrusal cebirin temelleriyle karşılaştık ve verilerimizi dönüştürürken genel işlemleri ifade etmek için nasıl kullanılabileceğini gördük. Doğrusal cebir, derin öğrenmede ve daha geniş anlamda makine öğrenmesinde yaptığımız işlerin çoğunun altında yatan temel matematiksel sütunlardan biridir. :numref:`sec_linear-algebra`, modern derin öğrenme modellerinin mekaniğini iletmek için yeterli mekanizmayı içerirken, konuyla ilgili daha çok şey var.
Bu bölümde daha derine ineceğiz, doğrusal cebir işlemlerinin bazı geometrik yorumlarını vurgulayacağız ve özdeğerler (eigenvalues) ve özvektörler (eigenvectors) dahil birkaç temel kavramı tanıtacağız.

## Vektörlerin Geometrisi
İlk olarak, uzaydaki noktalar veya yönler olarak vektörlerin iki ortak geometrik yorumunu tartışmamız gerekiyor.
Temel olarak bir vektör, aşağıdaki Python listesi gibi bir sayı listesidir.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

Matematikçiler bunu genellikle bir *sütun* veya *satır* vektörü olarak yazarlar;

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

veya

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$ 

Bunlar genellikle veri örneklerinin sütun vektörleri ve ağırlıklı toplamları oluşturmada kullanılan ağırlıkların satır vektörleri olduğu farklı yorumlara sahiptir.
Ancak esnek olmak faydalı olabilir.
 :numref:`sec_linear-algebra` bölümünde açıkladığımız gibi, tek bir vektörün varsayılan yönelimi bir sütun vektörü olsa da, tablo halindeki bir veri kümesini temsil eden herhangi bir matris için, her bir veri örneğini matriste bir satır vektörü olarak ele almak daha gelenekseldir.

Bir vektör verildiğinde, ona vermemiz gereken ilk yorum uzayda bir nokta olduğudur.
İki veya üç boyutta, bu noktaları, *köken (orijin)* adı verilen sabit bir referansa kıyasla uzaydaki konumlarını belirtmek için vektör bileşenlerini kullanarak görselleştirebiliriz. Bu, şurada görülebilir :numref:`fig_grid`.

![Vektörleri düzlemdeki noktalar olarak görselleştirmenin bir örneği. Vektörün ilk bileşeni $x$ koordinatını verir, ikinci bileşen $y$ koordinatını verir. Görselleştirilmesi çok daha zor olsa da, daha yüksek boyutlar da benzerdir.](../img/grid-points.svg)
:label:`fig_grid`

Bu geometrik bakış açısı, sorunu daha soyut bir düzeyde ele almamızı sağlar.
Artık resimleri kedi veya köpek olarak sınıflandırmak gibi başa çıkılmaz görünen bir problemle karşılaşmadığımızdan, görevleri soyut olarak uzaydaki nokta toplulukları olarak değerlendirmeye ve görevi iki farklı nokta kümesini nasıl ayıracağımızı keşfetmek olarak resmetmeye başlayabiliriz.

Buna paralel olarak, insanların genellikle vektörleri aldıkları ikinci bir bakış açısı vardır: Uzayda yönler olarak.
$\mathbf{v} = [3,2]^\top$ vektörünü başlangıç noktasından $3$ birim sağda ve $2$ birim yukarıda bir konum olarak düşünmekle kalmayabiliriz, aynı zamanda onu sağa doğru $3$ adım ve yukarı doğru $2$ adım şekilde yönün kendisi olarak da düşünebiliriz.
Bu şekilde, şekildeki tüm vektörleri aynı kabul ederiz :numref:`fig_arrow`.

![Herhangi bir vektör, düzlemde bir ok olarak görselleştirilebilir. Bu durumda, çizilen her vektör $(3,2)$ vektörünün bir temsilidir.](../img/par-vec.svg)
:label:`fig_arrow`

Bu değişik gösterimin faydalarından biri, vektör toplama işlemini görsel olarak anlamlandırabilmemizdir.
Özellikle, bir vektör tarafından verilen yönleri izliyoruz ve şekil :numref:`fig_add-vec` içinde görüldüğü gibi, sonra diğerinin verdiği yönleri takip ediyoruz.

![Önce bir vektörü, sonra diğerini takip ederek vektör toplamayı görselleştirebiliriz.](../img/vec-add.svg)
:label:`fig_add-vec`

Vektör çıkarma işleminin benzer bir yorumu vardır.
$\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$ özdeşliğini göz önünde bulundurursak, $\mathbf{u}-\mathbf{v}$ vektörü, bizi $\mathbf{v}$ noktasından $\mathbf{u}$ noktasına götüren yöndür.


## Nokta (İç) Çarpımları ve Açılar
:numref:`sec_linear-algebra` içinde gördüğümüz gibi, $\mathbf{u}$ ve $\mathbf{v}$ gibi iki sütun vektörü alırsak, bunların nokta çarpımını aşağıdaki işlemi hesaplayarak oluşturabiliriz:

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

:eqref:`eq_dot_def` simetrik olduğundan, klasik çarpmanın gösterimini kopyalayacağız ve şöyle yazacağız:

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

Böylece vektörlerin sırasını değiştirmenin aynı cevabı vereceği gerçeğini vurgulamış olacağız.

İç çarpım :eqref:`eq_dot_def` ayrıca geometrik bir yorumu da kabul eder: O da iki vektör arasındaki açı ile yakından ilgilidir. :numref:`fig_angle` içinde gösterilen açıyı düşünün.

![Düzlemdeki herhangi iki vektör arasında iyi tanımlanmış bir $\theta$ açısı vardır. Bu açının iç çarpıma yakından bağlı olduğunu göreceğiz.](../img/vec-angle.svg)
:label:`fig_angle`

Başlamak için iki belli vektörü ele alalım:

$$
\mathbf{v} = (r,0) \; \text{ve} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

$\mathbf{v}$ vektörü $r$ uzunluğundadır ve $x$ eksenine paralel uzanır, $\mathbf{w}$ vektörü $s$ uzunluğundadır ve $x$ ekseni ile arasında $\theta$ açısı vardır.
Bu iki vektörün iç çarpımını hesaplarsak, şunu görürüz:

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

Bazı basit cebirsel işlemlerle, terimleri yeniden düzenleyebiliriz.

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

Kısacası, bu iki belli vektör için, normlarla birleştirilmiş iç çarpım bize iki vektör arasındaki açıyı söyler. Aynı gerçek genel olarak doğrudur. Burada ifadeyi türetmeyeceğiz, ancak $\|\mathbf{v} - \mathbf{w}\|^2$'yi iki şekilde yazmayı düşünürsek, biri nokta çarpımı ile, diğeri geometrik olarak kosinüsler yasasının kullanımı ile, tam ilişkiyi elde edebiliriz.
Gerçekten de, herhangi iki vektör, $\mathbf{v}$ ve $\mathbf{w}$ için, aralarındaki açı:

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

Hesaplamadaki hiçbir şey iki boyutluluğu referans almadığı için bu güzel bir sonuçtur.
Aslında bunu üç veya üç milyon boyutta da sorunsuz olarak kullanabiliriz.

Basit bir örnek olarak, bir çift vektör arasındaki açıyı nasıl hesaplayacağımızı görelim:

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

Şu anda kullanmayacağız, ancak açılarının $\pi/2$ (veya eşdeğer olarak $90^{\circ}$) olduğu vektörleri *dik* olarak isimlendireceğimizi bilmekte fayda var.
Yukarıdaki denklemi inceleyerek, bunun $\theta = \pi/2$ olduğunda gerçekleştiğini görürüz, bu $\cos(\theta) = 0$ ile aynı şeydir.
Bunun gerçekleşmesinin tek yolu, nokta çarpımın kendisinin sıfır olmasıdır ve ancak ve ancak $\mathbf{v}\cdot\mathbf{w} = 0$ ise iki vektör dik olur.
Bu, nesneleri geometrik olarak anlarken faydalı bir formül olacaktır.  

Şu soruyu sormak mantıklıdır: Açıyı hesaplamak neden yararlıdır?
Cevap, verinin sahip olmasını beklediğimiz türden değişmezlikten gelir.
Bir imge ve her piksel değerinin aynı, ancak parlaklığın $\% 10$ olduğu kopya bir imge düşünün.
Tek tek piksellerin değerleri genel olarak asıl değerlerden uzaktır.
Bu nedenle, hakiki imge ile daha karanlık olan arasındaki mesafe hesaplanırsa, mesafe büyük olabilir.
Gene de, çoğu makine öğrenmesi uygulaması için *içerik* aynıdır---kedi/köpek sınıflandırıcısı söz konusu olduğunda yine de bir kedinin imgesidir.
Ancak, açıyı düşünürsek, herhangi bir $\mathbf{v}$ vektörü için $\mathbf{v}$ ve $0.1\cdot\mathbf{v}$ arasındaki açının sıfır olduğunu görmek zor değildir.
Bu, ölçeklemenin vektörlerin yönlerini koruduğu ve sadece uzunluğu değiştirdiği gerçeğine karşılık gelir.
Açı, koyu imgeyi aynı kabul edecektir.

Buna benzer örnekler her yerdedir.
Metinde, aynı şeyleri söyleyen iki kat daha uzun bir belge yazarsak tartışılan konunun değişmemesini isteyebiliriz.
Bazı kodlamalar için (herhangi sözcük dağarcığındaki kelimelerin kaç kere geçtiğinin sayılması gibi), bu, belgeyi kodlayan vektörün ikiye çarpılmasına karşılık gelir ki, böylece yine açıyı kullanabiliriz.

### Kosinüs Benzerliği
Açının iki vektörün yakınlığını ölçmek için kullanıldığı makine öğrenmesi bağlamlarında, uygulayıcılar benzerlik miktarını ifade etmek için *kosinüs benzerliği* terimini kullanırlar.
$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

İki vektör aynı yönü gösterdiğinde kosinüs maksimum $1$, zıt yönleri gösterdiklerinde minimum  $-1$ ve birbirlerine dik iseler $0$ değerini alır.
Yüksek boyutlu vektörlerin bileşenleri ortalama $0$ ile rastgele örneklenirse, kosinüslerinin neredeyse her zaman $0$'a yakın olacağını unutmayın.


## Hiperdüzlemler

Vektörlerle çalışmaya ek olarak, doğrusal cebirde ileri gitmek için anlamanız gereken bir diğer önemli nesne olan *hiperdüzlem*, bir doğrunun (iki boyut) veya bir düzlemin (üç boyut) daha yüksek boyutlarına genellenmesidir.
$d$ boyutlu bir vektör uzayında, bir hiperdüzlemin $d-1$ boyutu vardır ve uzayı iki yarı-uzaya böler.

Bir örnekle başlayalım.
Sütun vektörümüzün $\mathbf{w}=[2,1]^\top$ olduğunu varsayalım. "$\mathbf{w}\cdot\mathbf{v} = 1$ olan $\mathbf{v}$ noktaları nedir?", bilmek istiyoruz. 
Yukarıda :eqref:`eq_angle_forumla` içindeki nokta çarpımları ile açılar arasındaki bağlantıyı hatırlayarak, bunun aşağıdaki denkleme eşdeğer olduğunu görebiliriz:
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

![Trigonometriyi anımsarsak, $\|\mathbf{v}\|\cos(\theta)$ formülünün $\mathbf{v}$ vektörünün $\mathbf{w}$ yönüne izdüşümünün uzunluğu olduğunu görürüz.](../img/proj-vec.svg)
:label:`fig_vector-project`

Bu ifadenin geometrik anlamını düşünürsek, :numref:`fig_vector-project` içinde gösterildiği gibi, bunun $\mathbf{v}$'nin $\mathbf{w}$ yönündeki izdüşümünün uzunluğunun tam olarak $1/\|\mathbf{w}\|$ olduğunu söylemeye eşdeğer olduğunu görürüz.
Bunun doğru olduğu tüm noktalar kümesi, $\mathbf{w}$ vektörüne dik açıda olan bir doğrudur.
İstersek, bu doğrunun denklemini bulabilir ve bunun $2x + y = 1$ veya eşdeğer olarak $y = 1 - 2x$ olduğunu görebiliriz.

Şimdi $\mathbf{w}\cdot\mathbf{v} > 1$ veya $\mathbf{w}\cdot\mathbf{v} < 1$ ile nokta kümesini sorduğumuzda ne olduğuna bakarsak, bunların sırasıyla $1/\|\mathbf{w}\|$'den daha uzun veya daha kısa izdüşümlerin (projeksiyon) olduğu durumlar olduğunu görebiliriz.
Dolayısıyla, bu iki eşitsizlik çizginin her iki tarafını da tanımlar.
Bu şekilde, :numref:`fig_space-division` içinde gördüğümüz gibi, uzayımızı iki yarıya bölmenin bir yolunu bulduk, öyleki bir taraftaki tüm noktaların nokta çarpımı bir eşiğin altında ve diğer tarafında ise yukarıdadır. 

![Şimdi ifadenin eşitsizlik versiyonunu ele alırsak, hiperdüzlemimizin (bu durumda: sadece bir çizgi) uzayı iki yarıma ayırdığını görürüz.](../img/space-division.svg)
:label:`fig_space-division`

Daha yüksek boyuttaki hikaye hemen hemen aynıdır.
Şimdi $\mathbf{w} = [1,2,3]^\top$ alırsak ve $\mathbf{w}\cdot\mathbf{v} = 1$ ile üç boyuttaki noktaları sorarsak, verilen $\mathbf{w}$ vektörüne dik açıda bir düzlem elde ederiz.
İki eşitsizlik yine düzlemin iki tarafını şu şekilde, :numref:`fig_higher-division`, gösterildiği gibi tanımlar .

![Herhangi bir boyuttaki hiperdüzlemler, uzayı ikiye böler.](../img/space-division-3d.svg)
:label:`fig_higher-division`

Bu noktada görselleştirme yeteneğimiz tükenirken, bizi bunu onlarca, yüzlerce veya milyarlarca boyutta yapmaktan hiçbir şey alıkoyamaz.
Bu genellikle makinenin öğrendiği modeller hakkında düşünürken ortaya çıkar.
Örneğin :numref:`sec_softmax` içindeki gibi doğrusal sınıflandırma modellerini farklı hedef sınıfları ayıran hiperdüzlemleri bulma yöntemleri olarak anlayabiliriz.
Bu bağlamda, bu tür hiperdüzlemlere genellikle *karar düzlemleri* adı verilir.
Derin eğitilmiş sınıflandırma modellerinin çoğu, bir eşiksiz en büyük işleve (softmak) beslenen doğrusal bir katmanla sona erer, bu nedenle derin sinir ağının rolü, hedef sınıfların hiperdüzlemler tarafından temiz bir şekilde ayrılabileceği şekilde doğrusal olmayan bir gömme bulmak olarak yorumlanabilir.

El yapımı bir örnek vermek gerekirse, Fashion MNIST veri kümesinden (:numref:`sec_fashion_mnist`) küçük tişört ve pantolon resimlerini sınıflandırmak için sadece ortalamalarını birleştiren bir vektör alıp karar düzlemini ve göz kararı kaba bir eşiği tanımlayarak makul bir model oluşturabileceğimize dikkat edin. İlk önce verileri yükleyeceğiz ve ortalamaları hesaplayacağız.

```{.python .input}
# Veri kümesine yükle
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Ortalamaları hesapla
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Veri kümesine yükle
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Ortalamaları hesapla
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# Veri kümesine yükle
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# Ortalamaları hesapla
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

Bu ortalamaları ayrıntılı olarak incelemek bilgilendirici olabilir, bu yüzden neye benzediklerini çizelim. Bu durumda, ortalamanın gerçekten de bir tişörtün bulanık görüntüsüne benzediğini görüyoruz.

```{.python .input}
#@tab mxnet, pytorch
# Ortalama tişörtü çizdir
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Ortalama tişörtü çizdir
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

İkinci durumda, yine ortalamanın bulanık bir pantolon görüntüsüne benzediğini görüyoruz.

```{.python .input}
#@tab mxnet, pytorch
# Ortalama pantolonu çizdir
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Ortalama pantolonu çizdir
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

Tamamen makine öğrenmeli bir çözümde, eşiği veri kümesinden öğrenecektik. Bu durumda, el ile eğitim verilerinde iyi görünen bir eşiği göz kararı aldık.

```{.python .input}
# Göz elde edilen eşiği ile test kümesi doğruluğunu yazdırın
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Doğruluk
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# Göz elde edilen eşiği ile test kümesi doğruluğunu yazdırın
w = (ave_1 - ave_0).T
# '@' is Matrix Multiplication operator in pytorch.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Doğruluk
torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# Göz elde edilen eşiği ile test kümesi doğruluğunu yazdırın
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# Doğruluk
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## Doğrusal Dönüşümlerin Geometrisi

:numref:`sec_linear-algebra` ve yukarıdaki tartışmalar sayesinde, vektörlerin geometrisine, uzunluklara ve açılara dair sağlam bir anlayışa sahibiz.
Bununla birlikte, tartışmayı atladığımız önemli bir nesne var ve bu, matrislerle temsil edilen doğrusal dönüşümlerin geometrik bir şekilde anlaşılmasıdır. Potansiyel olarak farklı yüksek boyutlu iki uzay arasında verileri dönüştürken matrislerin neler yapabileceğini tam olarak içselleştirmek, önemli bir uygulama gerektirir ve bu ek bölümün kapsamı dışındadır.
Bununla birlikte, sezgimizi iki boyutta oluşturmaya başlayabiliriz.

Bir matrisimiz olduğunu varsayalım:

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

Bunu rastgele bir $\mathbf{v} = [x, y]^\top$ vektörüne uygulamak istersek, çarpar ve görürüz ki

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

Bu, net bir şeyin bir şekilde anlaşılmaz hale geldiği garip bir hesaplama gibi görünebilir.
Bununla birlikte, bize bir matrisin *herhangi* bir vektörü *iki belirli vektöre* göre nasıl dönüştürdüğünü yazabileceğimizi söyler: $[1,0]^\top$ and $[0,1]^\top$.
Bu bir an için düşünmeye değer.
Esasen sonsuz bir problemi (herhangi bir gerçel sayı çiftine olanı) sonlu bir probleme (bu belirli vektörlere ne olduğuna) indirgedik.
Bu vektörler, uzayımızdaki herhangi bir vektörü bu *taban vektörlerin* ağırlıklı toplamı olarak yazabileceğimiz örnek bir *taban*dır.

Belli bir matrisi kullandığımızda ne olacağını çizelim

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

Belirli $\mathbf{v} = [2, -1]^\top$ vektörüne bakarsak, bunun $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$ olduğunu görürüz, dolayısıyla $A$ matrisinin bunu $2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$'e göndereceğini biliyoruz.
Bu mantığı dikkatlice takip edersek, diyelim ki tüm tamsayı nokta çiftlerinin ızgarasını (grid) göz önünde bulundurarak, matris çarpımının ızgarayı eğriltebileceğini, döndürebileceğini ve ölçekleyebileceğini görürüz, ancak ızgara yapısı numref:`fig_grid-transform` içinde gördüğünüz gibi kalmalıdır.

![Verilen temel vektörlere göre hareket eden $\mathbf {A}$ matrisi. Tüm ızgaranın onunla birlikte nasıl taşındığına dikkat edin.](../img/grid-transform.svg)
:label:`fig_grid-transform`

Bu, matrisler tarafından temsil edilen doğrusal dönüşümler hakkında içselleştirilmesi gereken en önemli sezgisel noktadır.
Matrisler, uzayın bazı bölümlerini diğerlerinden farklı şekilde bozma yeteneğine sahip değildir.
Tüm yapabilecekleri, uzayımızdaki hakiki koordinatları almak ve onları eğriltmek, döndürmek ve ölçeklendirmektir.

Bazı çarpıklıklar şiddetli olabilir. Örneğin matris

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

iki boyutlu düzlemin tamamını tek bir çizgiye sıkıştırır.
Bu tür dönüşümleri tanımlamak ve bunlarla çalışmak daha sonraki bir bölümün konusudur, ancak geometrik olarak bunun yukarıda gördüğümüz dönüşüm türlerinden temelde farklı olduğunu görebiliriz.
Örneğin, $\mathbf{A}$ matrisinden gelen sonuç, orijinal ızgaraya "geri eğilebilir". $\mathbf{B}$ matrisinden gelen sonuçlar olamaz çünkü $[1,2]^\top$ vektörünün nereden geldiğini asla bilemeyiz --- bu $[1,1]^\top$ veya $[0,-1]^\top$ mıydı?

Bu resim $2\times 2$ matrisi için olsa da, hiçbir şey öğrenilen dersleri daha yüksek boyutlara taşımamızı engellemiyor.
$[1,0, \ldots,0]$ gibi benzer taban vektörleri alırsak ve matrisimizin onları nereye gönderdiğini görürsek, matris çarpımının, uğraştığımız boyut uzayında tüm uzayı nasıl bozduğu hakkında bir fikir edinmeye başlayabiliriz.

## Doğrusal Bağımlılık

Matrisi tekrar düşünün

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

Bu, tüm düzlemi $y = 2x$ tek doğruda yaşaması için sıkıştırır. Şimdi şu soru ortaya çıkıyor: Bunu sadece matrise bakarak tespit etmemizin bir yolu var mı?
Cevap, gerçekten edebiliriz.
$\mathbf{b}_1 = [2,4]^\top$ ve $\mathbf{b}_2 = [-1,-2]^\top$, $\mathbf {B}$'nin iki sütunu olsun.
$\mathbf{B}$ matrisi tarafından dönüştürülen her şeyi, matrisin sütunlarının ağırlıklı toplamı olarak yazabileceğimizi unutmayın: $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$ gibi.
Buna *doğrusal birleşim (kombinasyon)* diyoruz.
$\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ olması, bu iki sütunun herhangi bir doğrusal kombinasyonunu tamamen, mesela, $\mathbf{b}_2$ cinsinden yazabileceğimiz anlamına gelir, çünkü

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

Bu, sütunlardan birinin uzayda tek bir yön tanımlamadığından bir bakıma gereksiz olduğu anlamına gelir.
Bu matrisin tüm düzlemi tek bir çizgiye indirdiğini gördüğümüz için bu bizi çok şaşırtmamalı.
Dahası, $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ doğrusal bağımlılığının bunu yakaladığını görüyoruz.
Bunu iki vektör arasında daha simetrik hale getirmek için şöyle yazacağız:

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

Genel olarak, eğer aşağıdaki denklem için *hepsi sıfıra eşit olmayan* $a_1, \ldots, a_k$ katsayıları varsa, bir $\mathbf{v}_1, \ldots, \mathbf{v}_k$ vektörler topluluğunun *doğrusal olarak bağımlı* olduğunu söyleyeceğiz:

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

Bu durumda, vektörlerden birini diğerlerinin birtakım birleşimi olarak çözebilir ve onu etkili bir şekilde gereksiz hale getirebiliriz.
Bu nedenle, bir matrisin sütunlarındaki doğrusal bir bağımlılık, matrisimizin uzayı daha düşük bir boyuta sıkıştırdığına bir kanıttır.
Doğrusal bağımlılık yoksa, vektörlerin *doğrusal olarak bağımsız* olduğunu söyleriz.
Bir matrisin sütunları doğrusal olarak bağımsızsa, sıkıştırma gerçekleşmez ve işlem geri alınabilir.

## Kerte (Rank)

Genel bir $n\times m$ matrisimiz varsa, matrisin hangi boyut uzayına eşlendiğini sormak mantıklıdır.
Cevabımız *kerte* olarak bilinen bir kavram olacaktır.
Önceki bölümde, doğrusal bir bağımlılığın uzayın daha düşük bir boyuta sıkıştırılmasına tanıklık ettiğini ve bu nedenle bunu kerte kavramını tanımlamak için kullanabileceğimizi ifade ettik.
Özellikle, bir $\mathbf{A}$ matrisinin kertesi, sütunların tüm alt kümeleri arasındaki doğrusal bağımsız en büyük sütun sayısıdır. Örneğin, matris

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

$\mathrm{kerte}(B) = 1$'e sahiptir, çünkü iki sütunu doğrusal olarak bağımlıdır, ancak iki sütun da kendi başına doğrusal olarak bağımlı değildir.
Daha zorlu bir örnek için,

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

ve $\mathbf{C}$'nin kertesinin iki olduğunu gösterebiliriz, çünkü örneğin ilk iki sütun doğrusal olarak bağımsızdır, ancak herhangi bir dört sütunlu toplulukta üç sütun bağımlıdır.

Bu prosedür, açıklandığı gibi, çok verimsizdir.
Verdiğimiz matrisin sütunlarının her alt kümesine bakmayı gerektirir ve bu nedenle sütun sayısına bağlı, potansiyel olarak üsteldir.
Daha sonra bir matrisin kertesini hesaplamanın hesaplama açısından daha verimli bir yolunu göreceğiz, ancak şimdilik, kavramın iyi tanımlandığını görmek ve anlamı anlamak yeterlidir.

## Tersinirlik (Invertibility)

Yukarıda doğrusal olarak bağımlı sütunları olan bir matris ile çarpmanın geri alınamayacağını gördük, yani girdiyi her zaman kurtarabilecek ters işlem yoktur. Bununla birlikte, tam kerteli bir matrisle çarpma durumunda (yani, $\mathbf{A}$, $n\times n$'lik $n$ kerteli matristir), bunu her zaman geri alabilmeliyiz. Şu matrisi düşünün:

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1 
\end{bmatrix}.
$$

Bu, köşegen boyunca birlerin ve başka yerlerde sıfırların bulunduğu matristir.
Buna *birim* matris diyoruz.
Uygulandığında verilerimizi değiştirmeden bırakan matristir.
$\mathbf{A}$ matrisimizin yaptıklarını geri alan bir matris bulmak için, şu şekilde bir $\mathbf{A}^{-1}$ matrisi bulmak istiyoruz:

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

Buna bir sistem olarak bakarsak, $n\times n$ bilinmeyenli ($\mathbf{A}^{-1}$'nin girdileri) ve $n\times n$ denklemimiz var ($\mathbf{A}^{-1}\mathbf{A}$ çarpımının her girdisi ve $\mathbf{I}$'nin her girdisi arasında uyulması gereken eşitlik), bu nedenle genel olarak bir çözümün var olmasını beklemeliyiz.
Nitekim bir sonraki bölümde sıfır olmadığı sürece çözüm bulabileceğimizi gösterme özelliğine sahip olan *determinant* adlı bir miktar göreceğiz. Böyle bir $\mathbf{A}^{-1}$ matrise, *ters* matris diyoruz.
Örnek olarak, eğer $\mathbf{A}$ genel bir $2\times 2$ matris ise

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d 
\end{bmatrix},
$$

o zaman tersinin şöyle olduğunu görebiliriz:

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a 
\end{bmatrix}.
$$

Yukarıdaki formülün verdiği ters ile çarpmanın pratikte işe yaradığını görmek için test edebiliriz.

```{.python .input}
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### Sayısal (Numerik) Sorunlar
Bir matrisin tersi teoride yararlı olsa da, pratikte bir problemi çözmek için çoğu zaman matris tersini *kullanmak* istemediğimizi söylemeliyiz.
Genel olarak,

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

gibi doğrusal denklemleri çözmek için sayısal olarak çok daha kararlı algoritmalar vardır, aşağıdaki gibi tersini hesaplamaktan ve çarpmaktan daha çok tercih edebileceğimiz yöntemlerdir.

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

Küçük bir sayıya bölünmenin sayısal kararsızlığa yol açması gibi, düşük kerteye olmaya yakın bir matrisin ters çevrilmesi de kararsızlığa neden olabilir.

Dahası, $\mathbf{A}$ matrisinin *seyrek* olması yaygındır, yani sadece az sayıda sıfır olmayan değer içerir.
Örnekleri araştıracak olsaydık, bunun tersin de seyrek olduğu anlamına gelmediğini görürdük.
$\mathbf{A}$, yalnızca $5$ milyon tanesi sıfır olmayan girdileri olan $1$ milyona $1$ milyonluk bir matris olsa bile (ve bu nedenle yalnızca bu $5$ milyon girdiyi saklamamız gerekir), tersi genellikle hemen hemen hepsi eksi değer olmayan tüm girdilere sahip olacaktır ki tüm $1\text{M}^2$ girdiyi saklamamızı gerektirir --- bu da $1$ trilyon girdidir!

Doğrusal cebir ile çalışırken sıkça karşılaşılan çetrefilli sayısal sorunlara tam olarak dalacak vaktimiz olmasa da, ne zaman dikkatli bir şekilde ilerlemeniz gerektiği konusunda size biraz önsezi sağlamak istiyoruz ve pratikte genellikle ters çevirmekten kaçınmak yararlı bir kuraldır.

## Determinant
Doğrusal cebirin geometrik görünümü, *determinant* olarak bilinen temel bir miktarı yorumlamanın sezgisel bir yolunu sunar.
Önceki ızgara görüntüsünü, ama şimdi vurgulanmış bölgeyle (:numref:`fig_grid-filled`) düşünün.

![$\mathbf{A}$ matrisi yine ızgarayı bozuyor. Bu sefer, vurgulanan kareye ne olduğuna özellikle dikkat çekmek istiyoruz.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

Vurgulanan kareye bakın. Bu, kenarları $(0,1)$ ve $(1,0)$ ile verilen bir karedir ve dolayısıyla bir birim alana sahiptir. $\mathbf{A}$ bu kareyi dönüştürdükten sonra, bunun bir paralelkenar olduğunu görürüz.
Bu paralelkenarın başladığımızdaki aynı alana sahip olması için hiçbir neden yok ve aslında burada gösterilen özel durumda aşağıdaki matristir.

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

Bu paralelkenarın alanını hesaplamak ve alanın $5$ olduğunu elde etmek koordinat geometrisinde bir alıştırmadır.

Genel olarak, bir matrisimiz varsa,

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

biraz hesaplamayla elde edilen paralelkenarın alanının $ad-bc$ olduğunu görebiliriz.
Bu alan, *determinant* olarak adlandırılır.

Bunu bazı örnek kodlarla hızlıca kontrol edelim.

```{.python .input}
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

Aramızdaki kartal gözlüler, bu ifadenin sıfır, hatta negatif olabileceğini fark edecek.
Negatif terim için, bu genel olarak matematikte ele alınan bir ifade meselesidir: Eğer matris şekli ters çevirirse, alanın aksine çevrildiğini söyleriz.
Şimdi determinant sıfır olduğunda daha fazlasını öğreneceğimizi görelim.

Bir düşünelim.

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

Bu matrisin determinantını hesaplarsak, $2\cdot(-2) - 4\cdot(-1) = 0$ elde ederiz.
Yukarıdaki anlayışımıza göre, bu mantıklı.
$\mathbf{B}$, orijinal görüntüdeki kareyi sıfır alana sahip bir çizgi parçasına sıkıştırır.
Ve aslında, dönüşümden sonra sıfır alana sahip olmanın tek yolu, daha düşük boyutlu bir alana sıkıştırılmaktır.
Böylece, aşağıdaki sonucun doğru olduğunu görüyoruz: Bir $A$ matrisinin, ancak ve ancak determinantı sıfıra eşit değilse tersi hesaplanabilir.

Son bir yorum olarak, düzlemde herhangi bir figürün çizildiğini hayal edin.
Bilgisayar bilimcileri gibi düşünürsek, bu şekli küçük kareler toplamına ayırabiliriz, böylece şeklin alanı özünde sadece ayrıştırmadaki karelerin sayısı olur.
Şimdi bu rakamı bir matrisle dönüştürürsek, bu karelerin her birini, determinant tarafından verilen alana sahip olan paralelkenarlara göndeririz.
Herhangi bir şekil için determinantın, bir matrisin herhangi bir şeklin alanını ölçeklendirdiği (işaretli) sayıyı verdiğini görüyoruz.

Daha büyük matrisler için belirleyicilerin hesaplanması zahmetli olabilir, ancak sezgi aynıdır.
Determinant, $n\times n$ matrislerin $n$-boyutlu hacimlerini ölçeklendiren faktör olarak kalır.

## Tensörler ve Genel Doğrusal Cebir İşlemleri

:numref:`sec_linear-algebra`'de tensör kavramı tanıtıldı.
Bu bölümde, tensör daralmalarına (büzülmesine) (matris çarpımının tensör eşdeğeri) daha derinlemesine dalacağız ve bir dizi matris ve vektör işlemi üzerinde nasıl birleşik bir görünüm sağlayabileceğini göreceğiz.

Matrisler ve vektörlerle, verileri dönüştürmek için onları nasıl çarpacağımızı biliyorduk.
Bize yararlı olacaksa, tensörler için de benzer bir tanıma ihtiyacımız var.
Matris çarpımını düşünün:

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

Veya eşdeğer olarak

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

Bu model tensörler için tekrar edebileceğimiz bir modeldir.
Tensörler için, neyin toplanacağına dair evrensel olarak seçilebilecek tek bir durum yoktur, bu yüzden tam olarak hangi indeksleri toplamak istediğimizi belirlememiz gerekir.
Örneğin düşünün,

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$ 

Böyle bir dönüşüme *tensör daralması* denir.
Tek başına matris çarpımının olduğundan çok daha esnek bir dönüşüm ailesini temsil edebilir.

Sık kullanılan bir gösterimsel sadeleştirme olarak, toplamın ifadede birden fazla kez yer alan indislerin üzerinde olduğunu fark edebiliriz, bu nedenle insanlar genellikle, toplamın örtülü olarak tüm tekrarlanan indisler üzerinden alındığı *Einstein gösterimi* ile çalışır.
Bu aşağıdaki kompakt ifadeyi verir:

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### Doğrusal Cebirden Yaygın Örnekler

Daha önce gördüğümüz doğrusal cebirsel tanımların kaçının bu sıkıştırılmış tensör gösteriminde ifade edilebileceğini görelim:

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

Bu şekilde, çok sayıda özel gösterimi kısa tensör ifadeleriyle değiştirebiliriz.

### Kodla İfade Etme
Tensörler de kod içinde esnek bir şekilde çalıştırılabilir.
:numref:`sec_linear-algebra` içinde görüldüğü gibi, aşağıda gösterildiği gibi tensörler oluşturabiliriz.

```{.python .input}
# Tensörleri tanımla
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Şekilleri yazdır
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Tensörleri tanımla
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Şekilleri yazdır
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# Tensörleri tanımla
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Şekilleri yazdır
A.shape, B.shape, v.shape
```

Einstein toplamı doğrudan uygulanır.
Einstein toplamında ortaya çıkan indisler bir dizi olarak aktarılabilir ve ardından işlem yapılan tensörler eklenebilir.
Örneğin, matris çarpımını uygulamak için, yukarıda görülen Einstein toplamını ($\mathbf{A}\mathbf{v} = a_{ij}v_j$) düşünebilir ve uygulamayı (gerçeklemeyi) elde etmek için indisleri söküp atabiliriz:

```{.python .input}
# Matris çarpımını yeniden uygula
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# Matris çarpımını yeniden uygula
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# Matris çarpımını yeniden uygula
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

Bu oldukça esnek bir gösterimdir.
Örneğin, geleneksel olarak şu şekilde yazılanı hesaplamak istiyorsak,

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

Einstein toplamı aracılığıyla şu şekilde uygulanabilir:

```{.python .input}
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

Bu gösterim insanlar için okunabilir ve etkilidir, ancak herhangi bir nedenle programlı olarak bir tensör daralması üretmeniz gerekirse hantaldır.
Bu nedenle `einsum`, her tensör için tamsayı indisleri sağlayarak alternatif bir gösterim sağlar.
Örneğin, aynı tensör daralması şu şekilde de yazılabilir:

```{.python .input}
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch bu tür gösterimi desteklemez.
```

```{.python .input}
#@tab tensorflow
# TensorFlow bu tür gösterimi desteklemez.
```

Her iki gösterim, tensör daralmalarının kodda kısa ve verimli bir şekilde temsiline izin verir.

## Özet
* Vektörler, uzayda geometrik olarak noktalar veya yönler olarak yorumlanabilir.
* Nokta çarpımları, keyfi olarak yüksek boyutlu uzaylar için açı kavramını tanımlar.
* Hiperdüzlemler, doğruların ve düzlemlerin yüksek boyutlu genellemeleridir. Genellikle bir sınıflandırma görevinde son adım olarak kullanılan karar düzlemlerini tanımlamak için kullanılabilirler.
* Matris çarpımı, geometrik olarak, temel koordinatların tekdüze bozulmaları olarak yorumlanabilir. Vektörleri dönüştürmenin çok kısıtlı, ancak matematiksel olarak temiz bir yolunu temsil ederler.
* Doğrusal bağımlılık, bir vektör topluluğunun beklediğimizden daha düşük boyutlu bir uzayda olduğunu anlamanın bir yoludur (diyelim ki $2$ boyutunda bir uzayda yaşayan $3$ vektörünüz var). Bir matrisin kertesi, doğrusal olarak bağımsız olan sütunlarının en büyük alt kümesinin ebadıdır.
* Bir matrisin tersi tanımlandığında, matris tersi bulma, ilkinin eylemini geri alan başka bir matris bulmamızı sağlar. Matris tersi bulma teoride faydalıdır, ancak sayısal kararsızlık nedeniyle pratikte dikkat gerektirir.
* Determinantlar, bir matrisin bir alanı ne kadar genişlettiğini veya daralttığını ölçmemizi sağlar. Sıfır olmayan bir determinant, tersinir (tekil olmayan) bir matris anlamına gelir ve sıfır değerli bir determinant, matrisin tersinemez (tekil) olduğu anlamına gelir.
* Tensör daralmaları ve Einstein toplamı, makine öğrenmesinde görülen hesaplamaların çoğunu ifade etmek için düzgün ve temiz bir gösterim sağlar.

## Alıştırmalar
1. Aralarındaki açı nedir?
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}
$$
2. Doğru veya yanlış: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ ve $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ birbirinin tersi mi?
3. Düzlemde $100\mathrm{m}^2$ alanına sahip bir şekil çizdiğimizi varsayalım. Şeklin aşağıdaki matrise göre dönüştürüldükten sonraki alanı nedir?
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
4. Aşağıdaki vektör kümelerinden hangisi doğrusal olarak bağımsızdır?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
5. Bazı $a , b, c$ ve $d$ değerleri için $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ olarak yazılmış bir matrisiniz olduğunu varsayalım . Doğru mu yanlış mı: Böyle bir matrisin determinantı her zaman $0$'dır?
6. $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$  ve $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ vektörleri diktir. $Ae_1$ ve $Ae_2$'nin dik olması için $A$ matrisindeki koşul nedir?
7. Rasgele bir $A$ matrisi için Einstein gösterimi ile $\mathrm{tr}(\mathbf{A}^4)$'u nasıl yazabilirsiniz?


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1085)
:end_tab:
