# Ağırlık Sönümü
:label:`sec_weight_decay`

Artık aşırı öğrenme sorununu tanımladığımıza göre, modelleri düzenlileştirmek için bazı standart tekniklerle tanışabiliriz. Dışarı çıkıp daha fazla eğitim verisi toplayarak aşırı öğrenmeyi her zaman azaltabileceğimizi hatırlayın. Bu maliyetli, zaman alıcı veya tamamen kontrolümüz dışında olabilir ve kısa vadede koşturmayı imkansız hale getirebilir. Şimdilik, kaynaklarımızın izin verdiği kadar yüksek kaliteli veriye sahip olduğumuzu ve düzenlileştirmek tekniklerine odaklandığımızı varsayabiliriz.

Polinom bağlanım örneğimizde (:numref:`sec_model_selection`), modelimizin kapasitesini sadece yerleştirilmiş polinomun derecesini ayarlayarak sınırlayabileceğimizi hatırlayın. Gerçekten de, özniteliklerin sayısını sınırlamak, aşırı öğrenme ile mücadele için popüler bir tekniktir. Bununla birlikte, öznitelikleri bir kenara atmak, iş için çok patavatsız bir araç olabilir. Polinom bağlanım örneğine bağlı kalarak, yüksek boyutlu girdilerde neler olabileceğini düşünün. Polinomların çok değişkenli verilere doğal uzantıları, basitçe değişkenlerin güçlerinin çarpımı, *tek terimli* olarak adlandırılır. Bir tek terimliğin derecesi, güçlerin toplamıdır. Örneğin, $x_1^2 x_2$ ve $x_3 x_5^2$'nin her ikisi de 3. dereceden tek terimlidir.

$d$ derecesine sahip terimlerin sayısının, $d$ büyüdükçe hızla arttığını unutmayın. $k$ tane değişken verildiğinde, $d$ derecesindeki tek terimli sayıların sayısı (örn., $d$'den çok seçmeli $k$) ${k - 1 + d} \choose {k - 1}$ olur. Mesela $2$'den $3$'e kadar olan küçük derece değişiklikler bile modelimizin karmaşıklığını önemli ölçüde artırır. Bu nedenle, işlev karmaşıklığını ayarlamak için genellikle daha ince ayarlı bir araca ihtiyaç duyarız.

## Normlar ve Ağırlık Sönümü

Hem $L_2$ hem de $L_1$ normunu daha önce :numref:`subsec_lin-algebra-norms` ünitesinde tanıtmıştık, ikisi de genel $L_p$ normunun özel durumlarıdır.
(***Ağırlık sönümü* (genellikle $L_2$ düzenlileştirme olarak adlandırılır), parametrik makine öğrenmesi modellerini düzenlemek için en yaygın kullanılan teknik olabilir.**) Teknik, tüm $f$ işlevleri arasında, $f = 0$ işlevinin (tüm girdilere $0$ değerini atayarak) bir anlamda *en basit* olduğu ve sıfırdan uzaklığına göre bir fonksiyonun karmaşıklığını ölçebileceğimiz temel sezgisiyle motive edilir. Fakat bir fonksiyon ile sıfır arasındaki mesafeyi ne kadar kesinlikle ölçmeliyiz? Tek bir doğru cevap yok. Aslında, fonksiyonel analiz bölümleri ve Banach uzayları teorisi de dahil olmak üzere matematiğin tüm dalları, bu sorunu yanıtlamaya adanmıştır.

Basit bir yorumlama, bir $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ doğrusal fonksiyonunun karmaşıklığını, ağırlık vektörünün bir normu ile ölçmek olabilir, örneğin, $\| \mathbf{w} \|^2$ gibi. Küçük bir ağırlık vektörü sağlamanın en yaygın yöntemi, onun normunu, kaybın en aza indirilmesi problemine bir ceza terimi olarak eklemektir. Böylece orijinal amaç fonksiyonumuzu, *eğitim etiketlerindeki tahmin kaybını en aza indirir*, yeni bir amaç fonksiyonu ile değiştiriyor, *tahmin kaybı ile ceza teriminin toplamını en aza indiriyoruz*. Şimdi, ağırlık vektörümüz çok büyürse, öğrenme algoritmamız eğitim hatasını en aza indirmek yerine $\| \mathbf{w} \|^2$ ağırlık normunu en aza indirmeye odaklanabilir. Bu tam olarak istediğimiz şey. Bu şeyleri kodda örneklendirmek için, :numref:`sec_linear_regression` içindeki önceki doğrusal regresyon örneğimizi canlandıralım. Kaybımız şöyle verilir:

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

$\mathbf{x}^{(i)}$'in öznitelikler, $y^{(i)}$'nin her $i$ veri örneği için etiket ve $(\mathbf{w}, b)$ değerlerinin sırasıyla ağırlık ve ek girdi parametreleri olduğunu hatırlayın. Ağırlık vektörünün büyüklüğünü cezalandırmak için, bir şekilde $\| \mathbf{w} \|^2$'yi kayıp fonksiyonuna eklemeliyiz, ancak model bu yeni ilave ceza ile standart kaybı nasıl bir ödünleşmeye sokmalıdır? Pratikte, bu ödünleşme, geçerleme verilerini kullanarak öğrendiğimiz negatif olmayan bir hiper parametre, yani *düzenlileştirme sabiti* $\lambda$ ile karakterize ediyoruz:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

$\lambda = 0$ için, esas kayıp fonksiyonumuzu elde ediyoruz. $\lambda> 0$ için, $\| \mathbf{w} \|$'in boyutunu kısıtlıyoruz. Kural olarak $2$'ye böleriz: İkinci dereceden bir fonksiyonun türevini aldığımızda, $2$ ve $1/2$ birbirini götürür ve güncelleme ifadesinin güzel ve basit görünmesini sağlar. Dikkatli okuyucular, neden standart normla (yani Öklid mesafesi) değil de kare normla çalıştığımızı merak edebilirler. Bunu hesaplama kolaylığı için yapıyoruz. $L_2$ normunun karesini alarak, karekökü kaldırıyoruz ve ağırlık vektörünün her bir bileşeninin karelerinin toplamını bakıyoruz. Bu, cezanın türevini hesaplamayı kolaylaştırır: Türevlerin toplamı, toplamın türevine eşittir.

Dahası, neden ilk olarak L2 normuyla çalıştığımızı ve örneğin L1 normuyla çalışmadığımızı sorabilirsiniz.
Aslında, diğer seçenekler geçerli ve istatistiksel bakımdan popülerdir. $L_2$-regresyonlu doğrusal modeller klasik *sırt regresyon* algoritmasını oluştururken, $L_1$-regresyonlu doğrusal regresyon benzer şekilde istatistikte temel bir modeldir, ki popüler olarak *kement regresyon* diye bilinir.


$L_2$ normuyla çalışmanın bir nedeni, ağırlık vektörünün büyük bileşenlerine daha büyük cezalar verilmesidir. Bu, öğrenme algoritmamızı, ağırlığı daha fazla sayıda özniteliğe eşit olarak dağıtan modellere doğru yönlendirir. Uygulamada, bu onları tek bir değişkendeki ölçüm hatasına karşı daha gürbüz hale getirebilir. Aksine, $L_1$ cezaları, diğer ağırlıkları sıfıra yaklaştırarak temizler ve ağırlıkları küçük bir öznitelik kümesine yoğunlaştıran modellere yol açar. Buna *öznitelik seçme* denir ve başka nedenlerden dolayı arzu edilebilir.

:eqref:`eq_linreg_batch_update` içindeki gösterimi kullanırsak, $L_2$ ile düzenlileştirilmiş regresyon için rasgele gradyan iniş güncellemeleri aşağıdaki gibidir:

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

Daha önce olduğu gibi, $\mathbf{w}$'yi tahminimizin gözlemden farklı olduğu miktara göre güncelliyoruz. Bununla birlikte, $\mathbf{w}$'nin boyutunu sıfıra doğru küçültürüz. Bu nedenle, bu yönteme bazen "ağırlık sönümü" adı verilir: Yalnızca ceza terimi verildiğinde, optimizasyon algoritmamız, eğitimin her adımında ağırlığı *söndürür*. Öznitelik seçiminin tersine, ağırlık sönümü bize bir fonksiyonun karmaşıklığını ayarlamak için süregelen bir mekanizma sunar. Daha küçük $\lambda$ değerleri, daha az kısıtlanmış $\mathbf{w}$'ye karşılık gelirken, daha büyük $\lambda$ değerleri $\mathbf{w}$'yi daha önemli ölçüde kısıtlar. 

Karşılık gelen ek girdi cezasını, $b^2$'yi, dahil edip etmememiz, uygulamalar arasında değişebilir ve hatta bir sinir ağının katmanları arasında değişebilir. Genellikle, bir ağın çıktı katmanının ek girdi terimini düzenli hale getirmeyiz.

## Yüksek Boyutlu Doğrusal Regresyon

Basit bir sentetik örnekle ağırlık sönümlenmesinin faydalarını gösterebiliriz. İlk önce, daha önce olduğu gibi biraz veri oluşturuyoruz:

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

İlk olarak, [**önceki gibi biraz veri üretiyoruz**]

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ öyleki }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

Etiketimizi, sıfır ortalama ve 0.01 standart sapma ile Gauss gürültüsü ile bozulmuş girdilerimizin doğrusal bir fonksiyonu olarak seçiyoruz. Aşırı eğitimin etkilerini belirgin hale getirmek için problemimizin boyutunu $d = 200$'e çıkarabilir ve sadece 20 örnek içeren küçük bir eğitim kümesi ile çalışabiliriz.

```{.python .input}
#@tab all
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## Sıfırdan Uygulama

Aşağıda, orijinal hedef işlevine basitçe kare $L_2$ cezasını ekleyerek, ağırlık sönümünü sıfırdan uygulayacağız.

### [**Model Parametrelerini İlkletme**]

İlk olarak, model parametrelerimizi rastgele ilkletmek için bir fonksiyon tanımlayacağız.

```{.python .input}
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

```{.python .input}
#@tab pytorch
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

```{.python .input}
#@tab tensorflow
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

### (**$L_2$ Norm Cezasının Tanımlanması**)

Belki de bu cezayı uygulamanın en uygun yolu, tüm terimlerin karesini almak ve bunları toplamaktır.

```{.python .input}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
#@tab pytorch
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```{.python .input}
#@tab tensorflow
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

### [**Eğitim Döngülerini Tanımlama**]

Aşağıdaki kod, eğitim kümesine bir model uyarlar ve onu test kümesinde değerlendirir. Doğrusal ağ ve kare kayıp :numref:`chap_linear` bölümünden bu yana değişmedi, bu yüzden onları sadece `d2l.linreg` ve `d2l.squared_loss` yoluyla içe aktaracağız. Buradaki tek değişiklik, kaybımızın artık ceza terimi içermesidir.

```{.python .input}
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # L2 normu ceza terimi eklendi ve yayma,`l2_penalty(w)`'yi 
                # uzunluğu `batch_size` olan bir vektör yapıyor.
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                 d2l.evaluate_loss(net, test_iter, loss)))
    print('w\'nin L2 normu:', np.linalg.norm(w))
```

```{.python .input}
#@tab pytorch
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # L2 normu ceza terimi eklendi ve yayma,`l2_penalty(w)`'yi 
            # uzunluğu `batch_size` olan bir vektör yapıyor.
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w\'nin L2 normu:', torch.norm(w).item())
```

```{.python .input}
#@tab tensorflow
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # L2 normu ceza terimi eklendi ve yayma,`l2_penalty(w)`'yi 
                # uzunluğu `batch_size` olan bir vektör yapıyor.
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                 d2l.evaluate_loss(net, test_iter, loss)))
    print('w\'nin L2 normu:', tf.norm(w).numpy())
```

### [**Düzenleştirmesiz Eğitim**]

Şimdi bu kodu `lambd = 0` ile çalıştırarak ağırlık sönümünü devre dışı bırakıyoruz. Kötü bir şekilde fazla öğrendiğimizi, eğitim hatasını azalttığımızı ancak test hatasını azaltmadığımızı unutmayın---aşırı öğrenmenin bir ders kitabı vakası.

```{.python .input}
#@tab all
train(lambd=0)
```

### [**Ağırlık Sönümünü Kullanma**]

Aşağıda, önemli ölçüde ağırlık sönümü ile çalışıyoruz. Eğitim hatasının arttığını ancak test hatasının azaldığını unutmayın. Düzenlileştirme ile beklediğimiz etki tam da budur.

```{.python .input}
#@tab all
train(lambd=3)
```

## [**Kısa Uygulama**]

Ağırlık sönümü sinir ağı optimizasyonunda her yerde mevcut olduğu için, derin öğrenme çerçevesi, herhangi bir kayıp fonksiyonuyla birlikte kolay kullanım için ağırlık sönümü optimizasyon algoritmasını kendisine kaynaştırarak bunu özellikle kullanışlı hale getirir. Dahası, bu kaynaştırma, herhangi bir ek hesaplama yükü olmaksızın, uygulama marifetlerinin algoritmaya ağırlık sönümü eklemesine izin vererek hesaplama avantajı sağlar. Güncellemenin ağırlık sönümü kısmı yalnızca her bir parametrenin mevcut değerine bağlı olduğundan, optimize edicinin herhalükarda her parametreye bir kez dokunması gerekir.

:begin_tab:`mxnet`
Aşağıdaki kodda, ağırlık sönümü hiper parametresini, `Trainer` (Eğitici) örneğimizi oluştururken doğrudan `wd` aracılığıyla belirtiyoruz. Varsayılan olarak Gluon hem ağırlıkları hem de ek girdileri aynı anda azaltır. Model parametreleri güncellenirken hiper parametre `wd`'nin `wd_mult` ile çarpılacağına dikkat edin. Bu nedenle, `wd_mult`'i sıfır olarak ayarlarsak, ek girdi parametresi $b$ sönmeyecektir.
:end_tab:

:begin_tab:`pytorch`
Aşağıdaki kodda, optimize edicimizi başlatırken ağırlık sönümü hiper parametresini doğrudan `weight_decay` aracılığıyla belirtiyoruz. PyTorch varsayılan olarak hem ağırlıkları hem de ek girdileri aynı anda azaltır. Burada ağırlık için yalnızca `weight_decay`'i ayarlıyoruz, böylece ek girdi parametresi $b$ sönmeyecektir.
:end_tab:

:begin_tab:`tensorflow`
Aşağıdaki kodda, ağırlık sönümü hiper parametresi `wd` ile bir $L_2$ düzenlileştirici oluşturuyoruz ve bunu katmana `kernel_regularizer` argümanı aracılığıyla uyguluyoruz.
:end_tab:

```{.python .input}
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # Ek girdi parametresi sönümlenmedi. Ek girdi adları genellikle "bias" ile biter
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w\'nin L2 normu:', np.linalg.norm(net[0].weight.data()))
```

```{.python .input}
#@tab pytorch
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # Ek girdi parametresi sönümlenmedi
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print(' w\'nin L2 normu:', net[0].weight.norm().item())
```

```{.python .input}
#@tab tensorflow
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # `tf.keras`, özel eğitim döngüsü için katmanlardaki kayıpların 
                # manuel olarak alınmasını ve eklenmesini gerektirir.
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                 d2l.evaluate_loss(net, test_iter, loss)))
    print('w\'nin L2 normu:', tf.norm(net.get_weights()[0]).numpy())
```

[**Grafikler, ağırlık sönümünü sıfırdan uyguladığımızdakilerle aynı görünüyor**]. Bununla birlikte, önemli ölçüde daha hızlı çalışırlar ve uygulanması daha kolaydır, bu fayda daha büyük problemler için daha belirgin hale gelecektir.

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

Şimdiye kadar, basit bir doğrusal işlevi neyin oluşturduğuna dair yalnızca bir fikre değindik. Dahası, basit doğrusal olmayan bir işlevi neyin oluşturduğu daha da karmaşık bir soru olabilir. Örneğin, [çekirdek Hilbert uzayını çoğaltma (Reproducing Kernel Hilbert Spaces - RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space), doğrusal  olmayan bir bağlamda doğrusal fonksiyonlar için tanınmış araçları uygulamaya izin verir. Ne yazık ki, RKHS tabanlı algoritmalar büyük, yüksek boyutlu verilere vasat ölçeklenme eğilimindedir. Bu kitapta, derin bir ağın tüm katmanlarına ağırlık sönümü uygulamanın basit sezgisel yöntemini varsayılan olarak alacağız.


## Özet

* Düzenlileştirme, aşırı öğrenme ile başa çıkmak için yaygın bir yöntemdir. Öğrenilen modelin karmaşıklığını azaltmak için eğitim kümesindeki kayıp işlevine bir ceza terimi ekler.
* Modeli basit tutmak için belirli bir seçenek, $L_2$ ceza kullanarak ağırlık sönümlemektir. Bu, öğrenme algoritmasının güncelleme adımlarında ağırlık sönümüne yol açar.
* Ağırlık sönümü işlevi, derin öğrenme çerçevelerinden optimize edicilerde sağlanır.
* Farklı parametre kümeleri, aynı eğitim döngüsü içinde farklı güncelleme davranışlarına sahip olabilir.


## Alıştırmalar

1. Bu bölümdeki tahmin probleminde $\lambda$ değeri ile deney yapınız. Eğitim ve test doğruluğunu $\lambda$ işlevinin bir işlevi olarak çizin. Ne gözlemliyorsunuz?
1. En uygun $\lambda$ değerini bulmak için bir geçerleme kümesi kullanın. Gerçekten optimal değer bu mudur? Bu önemli mi?
1. $\|\mathbf{w}\|^2$ yerine ceza seçimi olarak $\sum_i |w_i|$ kullansaydık ($L_1$ düzenlileştirme) güncelleme denklemleri nasıl görünürdü?
1. $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$ olduğunu biliyoruz. Matrisler için benzer bir denklem bulabilir misiniz (matematikçiler buna [Frobenius normu](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) diyorlar)?
1. Eğitim hatası ile genelleme hatası arasındaki ilişkiyi gözden geçirin. Ağırlık sönümü, artan eğitim ve uygun karmaşıklıkta bir modelin kullanılmasına ek olarak, aşırı öğrenme ile başa çıkmak için başka hangi yolları düşünebilirsiniz?
1. Bayesçi istatistikte, $P(w \mid x) \propto P(x \mid w) P(w)$ aracılığıyla bir sonsal olasılığın ve önsel olasılığın çarpımını kullanırız. $P(w)$'yi düzenlileştirme ile nasıl tanımlayabilirsiniz?


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/236)
:end_tab:
