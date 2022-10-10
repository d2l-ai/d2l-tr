 # Çok Değişkenli Hesap
:label:`sec_multivariable_calculus`

Artık tek değişkenli bir fonksiyonun türevleri hakkında oldukça güçlü bir anlayışa sahip olduğumuza göre, potansiyel olarak milyarlarca ağırlığa sahip bir kayıp (yitim) fonksiyonunu düşündüğümüz esas sorumuza dönelim.

## Yüksek Boyutlu Türev Alma
:numref:`sec_single_variable_calculus` bize, bu milyarlarca ağırlıktan diğer her birini sabit bırakarak sadece birini değiştirirsek ne olacağını bildiğimizi söyler! Bu, tek değişkenli bir fonksiyondan başka bir şey değildir, bu yüzden şöyle yazabiliriz

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`

Diğer değişkenleri sabit tutarken bir değişkendeki türeve *kısmi türev* diyeceğiz ve :eqref:`eq_part_der` denklemindeki türev için $\frac{\partial}{\partial w_1}$ gösterimini kullanacağız.

Şimdi bunu alalım ve $w_2$'yi biraz $w_2 + \epsilon_2$ olarak değiştirelim:

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$

Bir kez daha, $\epsilon_1 \epsilon_2$'nin daha yüksek bir terim olduğu fikrini kullandık ve önceki bölümde gördüğümüz $\epsilon^{2}$ ile aynı şekilde :eqref:`eq_part_der` denkleminden atabildik. Bu şekilde devam ederek şunu yazabiliriz:

$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$

Bu bir karmaşa gibi görünebilir, ancak sağdaki toplamın tam olarak bir iç çarpıma benzediğini fark ederek bunu daha tanıdık hale getirebiliriz.

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \text{and} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top,
$$

o zaman da:

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`

$\nabla_{\mathbf{w}}L$'yı $L$'nin *gradyanı* olarak adlandıracağız.

Denklem :eqref:`eq_nabla_use` bir an üstünde düşünmeye değerdir. Tam olarak bir boyutta karşılaştığımız formata sahip, sadece her şeyi vektörlere ve nokta çarpımlarına dönüştürdük. Girdiye herhangi ufak bir dürtme verildiğinde $L$ fonksiyonunun nasıl değişeceğini yaklaşık olarak söylememizi sağlar. Bir sonraki bölümde göreceğimiz gibi, bu bize gradyanda bulunan bilgileri kullanarak nasıl öğrenebileceğimizi geometrik olarak anlamamız için önemli bir araç sağlayacaktır.

Ama önce bu yaklaşıklamayı bir örnekle iş başında görelim. Şu fonksiyon ile çalıştığımızı varsayalım:

$$
f(x, y) = \log(e^x + e^y) \text{ ve gradyanı } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$

$(0,\log(2))$ gibi bir noktaya bakarsak görürüz ki

$$
f(x, y) = \log(3) \text{ ve gradyanı } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$

Bu nedenle,  $(\epsilon_1, \log(2) + \epsilon_2)$ konumunda $f$'ye yaklaşmak istiyorsak, şu özel örneğe sahip olmamız gerektiğini görürüz :eqref:`eq_nabla_use`:

$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$

Yaklaşıklamanın ne kadar iyi olduğunu görmek için bunu kodda test edebiliriz.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

## Gradyanların Geometrisi ve Gradyan (Eğim) İnişi
:eqref:`eq_nabla_use` denklemindeki ifadeyi tekrar düşünün: 

$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$

Diyelim ki, bunu $L$ kaybımızı en aza indirmeye yardımcı olmak için kullanmak istiyoruz. İlk önce :numref:`sec_autograd` içinde açıklanan gradyan iniş algoritmasını geometrik olarak anlayalım. Yapacağımız şey şudur:

1. $\mathbf{w}$ başlangıç parametreleri için rastgele bir seçimle başlayın.
2. $L$'nin $\mathbf{w}$ seviyesinde en hızlı azalmasını sağlayan $\mathbf{v}$ yönünü bulun.
3. Bu yönde küçük bir adım atın: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$.
4. Tekrar edin.

Tam olarak nasıl yapılacağını bilmediğimiz tek şey, ikinci adımdaki $\mathbf{v}$ vektörünü hesaplamaktır. Böyle bir yöne *en dik iniş yönü* diyeceğiz. :numref:`sec_geometry-linear-algebraic-ops` konusundan nokta çarpımlarının geometrik anlamını kullanarak, şunu, :eqref:`eq_nabla_use`, yeniden yazabileceğimizi görüyoruz:

$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$

Kolaylık olması açısından yönümüzü birim uzunluğa sahip olacak şekilde aldığımızı ve $\mathbf{v}$ ile $\nabla_{\mathbf{w}} L(\mathbf{w})$ arasındaki açı için $\theta$'yı kullandığımızı unutmayın. $L$'nin olabildiğince hızlı azalan yönünü bulmak istiyorsak, bu ifadeyi olabildiğince negatif olarak ifade etmek isteriz. Seçtiğimiz yönün bu denkleme girmesinin tek yolu $\cos(\theta)$ sayesindedir ve bu yüzden bu kosinüsü olabildiğince negatif yapmak istiyoruz. Şimdi, kosinüs şeklini hatırlarsak, bunu mümkün olduğunca negatif yapmak için $\cos(\theta) = -1$ yapmamız veya eşdeğer olarak gradyan ile seçtiğimiz yön arasındaki açıyı $\pi$ radyan olacak şekilde, diğer anlamda $180$ derece yapmamız gerekecektir. Bunu başarmanın tek yolu, tam ters yöne gitmektir: $\nabla_{\mathbf{w}} L(\mathbf{w})$ yönünün tam tersini gösteren $\mathbf{v}$'yi seçin!

Bu bizi makine öğrenmesindeki en önemli matematiksel kavramlardan birine getiriyor: $-\nabla_{\mathbf{w}}L(\mathbf{w})$ yönündeki en dik yokuş noktaların yönü. Böylece resmi olmayan algoritmamız aşağıdaki gibi yeniden yazılabilir.

1. $\mathbf{w}$ ilk parametreleri için rastgele bir seçimle başlayın.
2. $\nabla_{\mathbf{w}} L(\mathbf{w})$'yi hesaplayın.
3. Ters yönde küçük bir adım atın: $\mathbf{w} \rightarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. Tekrar edin.

Bu temel algoritma, birçok araştırmacı tarafından birçok şekilde değiştirilmiş ve uyarlanmıştır, ancak temel kavram hepsinde aynı kalır. Kaybı olabildiğince hızlı azaltan yönü bulmak için gradyanı kullanın ve bu yönde bir adım atmak için parametreleri güncelleyin.

## Matematiksel Optimizasyon (Eniyileme) Üzerine Bir Not
Bu kitap boyunca, derin öğrenme ortamında karşılaştığımız tüm işlevlerin açıkça en aza indirilemeyecek kadar karmaşık olmasından dolayı pratik nedenlerle doğrudan sayısal eniyileme tekniklerine odaklanıyoruz.

Bununla birlikte, yukarıda elde ettiğimiz geometrik anlayışın bize fonksiyonları doğrudan optimize etme hakkında ne söylediğini düşünmek faydalı bir alıştırmadır.

Bir $L(\mathbf{x})$ işlevini en aza indiren $\mathbf{x}_0$'nin değerini bulmak istediğimizi varsayalım. Diyelim ki birisi bize bir değer veriyor ve bize $L$'yi en aza indiren şeyin bu değer olduğunu söylüyor. Yanıtın makul olup olmadığını kontrol edebileceğimiz bir şey var mı?

Tekrar düşünün :eqref:`eq_nabla_use`:
$$
L(\mathbf{x}_0 + \boldsymbol{\epsilon}) \approx L(\mathbf{x}_0) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{x}} L(\mathbf{x}_0).
$$

Gradyan sıfır değilse, $L$'nin daha küçük değerini bulmak için $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ yönünde bir adım atabileceğimizi biliyoruz. Bu nedenle, gerçekten en düşük değerde isek, böyle bir durum olamaz! $\mathbf{x}_0$ bir minimum ise, $\nabla_{\mathbf{x}} L(\mathbf{x}_{0}) = 0$ olduğu sonucuna varabiliriz. $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$ ifadesi gerçek olan noktaları *kritik nokta* diye çağırıyoruz.

Bu güzel bir bilgi, çünkü bazı nadir durumlarda gradyanın sıfır olduğu tüm noktaları açıkça *bulabiliriz* ve en küçük değere sahip olanı bulabiliriz.

Somut bir örnek için bu işlevi düşünün:
$$
f(x) = 3x^4 - 4x^3 -12x^2.
$$

Bu fonksiyonun türevi aşağıdadır:
$$
\frac{df}{dx} = 12x^3 - 12x^2 -24x = 12x(x-2)(x+1).
$$

Minimumun olası konumları $x = -1, 0, 2$'dir, burada fonksiyon sırasıyla $-5,0, -32$ değerlerini alır ve böylece $x =2$ olduğunda fonksiyonumuzu küçülttüğümüz sonucuna varabiliriz. Hızlı bir çizim bunu doğrular.

```{.python .input}
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

Bu, teorik veya sayısal olarak çalışırken bilinmesi gereken önemli bir gerçeğin altını çiziyor: Bir işlevi en aza indirebileceğimiz (veya en yükseğe çıkarabileceğimiz) olası noktalar sıfıra eşit bir gradyana sahip olacaktır, ancak sıfır gradyanlı her nokta gerçek *küresel (global)* minimum (veya maksimum) değildir.

## Çok Değişkenli Zincir Kuralı
Pek çok terim oluşturarak yapabileceğimiz dört değişkenli ($w, x, y$ ve $z$) bir fonksiyonumuz olduğunu varsayalım:

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`

Sinir ağları ile çalışırken bu tür denklem zincirleri yaygındır, bu nedenle bu tür işlevlerin gradyanlarının nasıl hesaplanacağını anlamaya çalışmak çok önemlidir. Hangi değişkenlerin doğrudan birbiriyle ilişkili olduğuna bakarsak, bu bağlantının görsel ipuçlarını :numref:`fig_chain-1` içinde görmeye başlayabiliriz.

![Düğümlerin değerleri temsil ettiği ve kenarların işlevsel bağımlılığı gösterdiği yukarıda geçen işlev ilişkileri.](../img/chain-net1.svg)
:label:`fig_chain-1`

Hiçbir şey bizi sadece :eqref:`eq_multi_func_def` denkleminden her şeyi birleştirmekten ve bunu yazmaktan alıkoyamaz

$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$

O zaman türevi sadece tek değişkenli türevler kullanarak alabiliriz, ancak bunu yaparsak, kendimizi hızlı bir şekilde terimlere boğulmuş buluruz, çoğu da tekrar eder! Gerçekten de, örneğin şunu görebiliriz:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$ 

Daha sonra $\frac{\partial f}{\partial x}$'i de hesaplamak isteseydik, birçok tekrarlanan terim ve iki türev arasında birçok *paylaşılan* tekrarlanan terimle tekrar benzer bir denklem elde ederdik. Bu, büyük miktarda boşa harcanan işi temsil ediyor ve eğer türevleri bu şekilde hesaplamamız gerekseydi, tüm derin öğrenme devrimi başlamadan önce durmuş olurdu!

Sorunu çözelim. $a$'yı değiştirdiğimizde $f$'nin nasıl değiştiğini anlamaya çalışarak başlayacağız, esasen $w, x, y$ ve $z$ değerlerinin mevcut olmadığını varsayarak yapacağız. Gradyan ile ilk kez çalışırken yaptığımız gibi akıl yürüteceğiz. Bir $a$ alalım ve ona küçük bir miktar $\epsilon$ ekleyelim.

$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$

İlk satır kısmi türev tanımından, ikincisi gradyan tanımından gelir. $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$ ifadesinde olduğu gibi, her türevi tam olarak nerede değerlendirdiğimizi izlemek gösterimsel olarak külfetlidir, bu nedenle sık sık bunu çok daha akılda kalıcı olarak kısaltırız.

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$
 
Sürecin anlamını düşünmekte fayda var. $f(u (a, b), v(a, b))$ biçimindeki bir fonksiyonun $a$'daki bir değişiklikle değerini nasıl değiştirdiğini anlamaya çalışıyoruz. Bunun meydana gelebileceği iki yol vardır: $a \rightarrow u \rightarrow f$ ve $a \rightarrow v \rightarrow f$. Bu katkıların her ikisini de zincir kuralı aracılığıyla hesaplayabiliriz: $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$ ve $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$ hesaplanırlar ve toplanırlar.

Sağdaki işlevlerin, :numref:`fig_chain-2` şeklinde gösterildiği gibi soldakilere bağlı olan işlevlere bağımlı olduğu farklı bir işlev ağımız olduğunu hayal edin.

![Zincir kuralının daha ince bir başka örneği.](../img/chain-net2.svg)
:label:`fig_chain-2`

$\frac{\partial f}{\partial y}$ gibi bir şeyi hesaplamak için, $y$'den $f$'e kadar tüm (bu durumda $3$) yolları toplamamız gerekir.

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

Zincir kuralını bu şekilde anlamak, gradyanların ağlar boyunca nasıl aktığını ve neden LSTM'lerdeki (:numref:`sec_lstm`) veya artık (residual) katmanlardaki (:numref:`sec_resnet`) gibi çeşitli mimari seçimlerin gradyan akışını kontrol etmeye yardımcı olarak öğrenme sürecini şekillendirdiğini anlamaya çalışırken büyük kazançlar sağlayacaktır.

## Geri Yayma Algoritması

Önceki bölümdeki :eqref:`eq_multi_func_def` örneğine dönelim:

$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$

Diyelim ki $\frac{\partial f}{\partial w}$'yi hesaplıyoruz, çok değişkenli zincir kuralını uygularsak şunu görebiliriz:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$

$\frac{\partial f}{\partial w}$ hesaplamak için bu ayrıştırmayı kullanmayı deneyelim. Burada ihtiyacımız olan tek şeyin çeşitli tek adımlık kısmi türevler olduğuna dikkat edin:

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

Bunu kodda yazarsak, bu oldukça yönetilebilir bir ifade olur.

```{.python .input}
#@tab all
# Girdilerden çıktılara fonksiyonun değerini hesapla
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Tek adımlı kısmileri hesapla
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Girdilerden çıktılara nihai sonucu hesapla
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

Ancak, bunun hala $\frac{\partial f}{\partial x}$ gibi bir şeyi hesaplamayı kolaylaştırmadığını unutmayın. Bunun nedeni, zincir kuralını uygulamayı seçtiğimiz *yoldur*. Yukarıda ne yaptığımıza bakarsak, elimizden geldiğince paydada her zaman $\partial w$ tuttuk. Bu şekilde, $w$ değişkenin her değişkeni nasıl değiştirdiğini görerek zincir kuralını uygulamayı seçtik. İstediğimiz buysa, bu iyi bir fikir olabilir. Bununla birlikte, derin öğrenmedeki motivasyonumuza geri dönün: Her parametrenin *kaybı* nasıl değiştirdiğini görmek istiyoruz. Esasında, yapabildiğimiz her yerde $\partial f$'yi payda tutan zincir kuralını uygulamak istiyoruz!

Daha açık olmak gerekirse, şunu yazabileceğimize dikkat edin:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$

Zincir kuralının bu uygulaması bizim açıkça $\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \text{ve} \; \frac{\partial f}{\partial w}$'ları hesaplamamızı gerektirir. Aşağıdaki denklemleri de dahil etmekten bizi hiçbir şey alıkoyamaz:

$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$

Sonra tüm ağdaki *herhangi bir* düğümü değiştirdiğimizde $f$ değerinin nasıl değiştiğini takip edebiliyoruz. Haydi uygulayalım.

```{.python .input}
#@tab all
# Girdilerden çıktılara fonksiyonun değerini hesapla
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Yukarıdaki ayrıştırmayı kullanarak türevi hesapla
# İlk önce tek adımlı kısmileri hesapla
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Şimdi herhangi bir değeri çıktıdan girdiye değiştirdiğimizde f'nin nasıl değiştiğini hesapla
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```

Türevleri, girdilerden çıktılara, ileriye doğru, hesaplamaktansa, $f$'den girdilere doğru hesapladığımız gerçeği (yukarıdaki ilk kod parçacığında yaptığımız gibi), bu algoritmaya adını veren şeydir: *Geri yayma*. İki adım olduğunu unutmayın:
1. Fonksiyonun değerini ve önden arkaya tek adımlık kısmi değerlerini hesaplayın. Yukarıda yapılmasa da, bu tek bir *ileri geçişte* birleştirilebilir.
2. Arkadan öne doğru $f$ gradyanını hesaplayın. Biz buna *geriye doğru geçiş* diyoruz.

Bu, her derin öğrenme algoritmasının, bir geçişte ağdaki her ağırlığa göre kaybın gradyanının hesaplanmasına izin vermek, uyguladığı şeydir. Böyle bir ayrışmaya sahip olmamız şaşırtıcı bir gerçektir.

Bunu nasıl içeri işlediğimizi görmek için bu örneğe hızlıca bir göz atalım.

```{.python .input}
# ndarrays olarak ilklet, ardından gradyanları ekle
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Hesaplamayı her zamanki gibi yap, gradyanları takip et
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Geriye doğru geçişi uygula
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# ndarrays olarak ilklet, ardından gradyanları ekle
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Hesaplamayı her zamanki gibi yap, gradyanları takip et
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Geriye doğru geçişi uygula
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# ndarrays olarak ilklet, ardından gradyanları ekle
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Hesaplamayı her zamanki gibi yap, gradyanları takip et
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Geriye doğru geçişi uygula
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```

Yukarıda yaptığımız şeylerin tümü, `f.backwards()` çağrısıyla otomatik olarak yapılabilir.

## Hessianlar
Tek değişkenli hesapta olduğu gibi, bir işleve tek başına gradyanı kullanmaktan daha iyi bir yaklaşıklama elde edebilmek için daha yüksek dereceli türevleri düşünmek yararlıdır.

Birkaç değişkenli fonksiyonların daha yüksek dereceden türevleriyle çalışırken karşılaşılan anlık bir sorun vardır ve bu da çok sayıda olmasıdır. $f(x_1, \ldots, x_n)$ fonksiyonun $n$ değişkeni varsa, o zaman $n^{2}$ tane ikinci türev alabiliriz, yani herhangi bir $i$ ve $j$ seçeneği için:

$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$

Bu, geleneksel olarak *Hessian* adı verilen bir matris içinde toplanır:

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`

Bu matrisin her girdisi bağımsız değildir. Aslında, her iki *karışık kısmi* (birden fazla değişkene göre kısmi türevler) var ve sürekli olduğu sürece, herhangi bir $i$ ve $j$ için şunu söyleyebiliriz:

$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$

Bunu, önce bir işlevi $x_i$ yönünde ve ardından $x_j$ yönünde dürtmeyi ve ardından bunun sonucunu önce $x_j$ ve sonra $x_i$ yönünde dürtersek ne olacağıyla karşılaştırmak izler; burada her iki sıranın da $f$'nin çıktısında aynı nihai değişikliğe yol açtığı bilgisine ulaşırız.

Tek değişkenlerde olduğu gibi, fonksiyonun bir noktanın yakınında nasıl davrandığına dair daha iyi bir fikir edinmek için bu türevleri kullanabiliriz. Özellikle, tek bir değişkende gördüğümüz gibi, $\mathbf{x}_0$ noktasının yakınında en uygun ikinci derece fonksiyonu bulmak için kullanabiliriz.

Bir örnek görelim. $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$ olduğunu varsayalım. Bu, iki değişkenli bir ikinci dereceden fonksiyon genel formudur. Fonksiyonun değerine, gradyanına ve Hessian :eqref:`eq_hess_def` değerine bakarsak, hepsi sıfır noktasında şöyle gözükür:

$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$

esas polinomumuzu şöyle diyerek geri elde edebiliriz;

$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$

Genel olarak, bu genişletmeyi herhangi bir $\mathbf {x}_0$ noktasında hesaplasaydık, şunu görürdük

$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$

Bu, herhangi bir boyuttaki girdi için çalışır ve bir noktadaki herhangi bir işleve en iyi yaklaşıklayan ikinci derece polinomu sağlar. Bir örnek vermek gerekirse, fonksiyonun grafiğini çizelim.

$$
f(x, y) = xe^{-x^2-y^2}.
$$

Gradyan ve Hessian şöyle hesaplanabilir:
$$
\nabla f(x, y) = e^{-x^2-y^2}\begin{pmatrix}1-2x^2 \\ -2xy\end{pmatrix} \; \text{and} \; \mathbf{H}f(x, y) = e^{-x^2-y^2}\begin{pmatrix} 4x^3 - 6x & 4x^2y - 2y \\ 4x^2y-2y &4xy^2-2x\end{pmatrix}.
$$

Böylece, biraz cebirle, $[-1,0]^\top$'deki yaklaşık ikinci dereceden polinomu görebiliriz:

$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
# Izgarayı oluştur ve işlevi hesapla
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# (1, 0)'da gradyan ve Hessian ile ikinci dereceden yaklaştırmayı hesapla
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# İşlevi çiz
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), w.asnumpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Izgarayı oluştur ve işlevi hesapla
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# (1, 0)'da gradyan ve Hessian ile ikinci dereceden yaklaştırmayı hesapla
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Pİşlevi çiz
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Izgarayı oluştur ve işlevi hesapla
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# (1, 0)'da gradyan ve Hessian ile ikinci dereceden yaklaştırmayı hesapla
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# İşlevi çiz
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

Bu, :numref:`sec_gd` içinde tartışılan Newton algoritmasının temelini oluşturur; sayısal optimizasyonu yinelemeli olarak uygulayarak en uygun ikinci derece polinomu bulur, sonra da tam olarak bu ikinci dereceden polinomu en aza indiririz. 

## Biraz Matris Hesabı
Matrisleri içeren fonksiyonların türevlerinin oldukça iyi olduğu ortaya çıktı. Bu bölüm gösterimsel olarak ağır hale gelebilir, bu nedenle ilk okumada atlanabilir, ancak genel matris işlemlerini içeren fonksiyonların türevlerinin genellikle başlangıçta tahmin edebileceğinden çok daha temiz olduğunu bilmek yararlıdır, özellikle de merkezi matris işlemlerinin derin öğrenme uygulamaları için ne kadar olduğu göz önüne alındığında.

Bir örnekle başlayalım. Bir sabit sütun vektörümüz $\boldsymbol{\beta}$ olduğunu ve $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ çarpım fonksiyonunu almak ve $\mathbf{x}$'i değiştirdiğimizde iç çarpım nasıl değişir anlamak istediğimizi varsayalım.

Makine öğrenmesinde matris türevleriyle çalışırken faydalı olacak bir gösterim parçası, kısmi türevlerimizi türevin paydada bulunduğu vektör, matris veya tensörün şekline dönüştürdüğümüz *payda düzenli matris türevi* olarak adlandırılır. Bu durumda şöyle yazacağız

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$

Burada $\mathbf{x}$ sütun vektörünün şekline eşleştirdik.

İşlevimizi bileşenlere yazarsak:

$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$

Şimdi $x_1$ göre kısmi türevi alırsak, ilk terim hariç her şeyin sıfır olduğuna dikkat edin, sadece $x_1$ ile $\beta_1$ ile çarpılır, böylece bunu elde ederiz:

$$
\frac{df}{dx_1} = \beta_1,
$$

ya da daha genel olarak

$$
\frac{df}{dx_i} = \beta_i.
$$

Şimdi bunu bir matris olarak görmek için yeniden birleştirebiliriz

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$

Bu, matris hesabı ile ilgili olarak bu bölümde sık sık karşılaşacağımız birkaç etkeni göstermektedir:

* İlk olarak, hesaplamalar daha çok işin içine girecek.
* İkinci olarak, nihai sonuçlar ara süreçten çok daha temizdir ve her zaman tek değişkenli duruma benzer görünecektir. Bu durumda, $\frac{d}{dx}(bx) = b$ ve $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$'nin ikisi de benzerdir.
* Üçüncüsü, devrikler genellikle herhangi bir yerden ortaya çıkabilirler. Bunun temel nedeni, paydanın şeklini eşleştirmemizdir, böylece matrisleri çarptığımızda, esas terimin şekline geri dönmek için devrikler almamız gerekecek.

Önsezi oluşturmaya devam etmek için biraz daha zor bir hesaplama deneyelim. Bir sütun vektörümüz $\mathbf{x}$ ve kare matrisimiz $A$ olduğunu ve şunu hesaplamak istediğimizi varsayalım.

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`

Gösterimde daha kolay oynama yaparak ileriye doğru ilerlemek için, bu problemi Einstein gösterimini kullanarak ele alalım. Bu durumda fonksiyonu şu şekilde yazabiliriz:

$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$

Türevimizi hesaplamak için, her $k$ için, değerin ne olduğunu anlamamız gerekir:

$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$

Çarpım kuralını uygulayalım:

$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$

$\frac{dx_i}{dx_k}$ gibi bir terim için, bunun $i = k$ için bir ve aksi halde sıfır olduğunda  görmek zor değildir. Bu, $i$ ve $k$ değerlerinin farklı olduğu her terimin bu toplamdan kaybolduğu anlamına gelir, bu nedenle bu ilk toplamda kalan tek terim, $i = k$ olanlardır. Aynı çıkarsama, $j = k$'ya ihtiyacımız olduğu ikinci terim için de geçerlidir. Böylece:

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$

Şimdi, Einstein gösterimindeki indislerin isimleri keyfidir --- $i$ ve $j$'nin farklı olması bu noktada bu hesaplama için önemsizdir, bu yüzden ikisinin de $i$ kullanması için yeniden indeksleyebiliriz:

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$

Şimdi, burası daha ileri gitmek için biraz pratik yapmaya ihtiyacımız olan yerdir. Bu sonucu matris işlemleri açısından belirlemeye çalışalım. $a_{ki} + a_{ik}$, $\mathbf{A} + \mathbf{A}^\top$'nin $k,i$-inci bileşenidir.

$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$

Benzer şekilde, bu terim artık $\mathbf{A} + \mathbf{A}^\top$ matrisinin $\mathbf{x}$ vektörü ile çarpımıdır, dolayısıyla şunu görüyoruz

$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$

Böylece, :eqref:`eq_mat_goal_1` denkleminden istenen türevin $k.$ girdisinin sağdaki vektörün $k.$ girdisi olduğunu ve dolayısıyla ikisinin aynı olduğunu görüyoruz. Sonunda:

$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$

Bu, önceki sonucumuzdan önemli ölçüde daha fazla çalışma gerektirdi, ancak nihai sonuç küçük. Bundan daha fazlası, geleneksel tek değişkenli türevler için aşağıdaki hesaplamayı düşünün:

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$ 

Eşdeğer olarak $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$'dir. Yine, tek değişkenli sonuca benzeyen, ancak içine devrik atılmış bir sonuç elde ederiz.

Bu noktada, model oldukça şüpheli görünmelidir, bu yüzden nedenini anlamaya çalışalım. Bunun gibi matris türevlerini aldığımızda, öncelikle aldığımız ifadenin başka bir matris ifadesi olacağını varsayalım: Onu matrislerin çarpımları ve toplamları ve bunların devrikleri cinsinden yazabileceğimiz bir ifade. Böyle bir ifade varsa, tüm matrisler için doğru olması gerekecektir. Özellikle, $1 \times 1$ matrisler için doğru olması gerekecektir, ki bu durumda matris çarpımı sadece sayıların çarpımıdır, matris toplamı sadece toplamdır ve devrik hiçbir şey yapmaz! Başka bir deyişle, aldığımız ifade ne olursa olsun tek değişkenli ifadeyle *eşleşmelidir*. Bu, biraz pratikle, yalnızca ilişkili tek değişkenli ifadenin neye benzemesi gerektiğini bilerek matris türevlerini tahmin edebileceğiniz anlamına gelir!

Bunu deneyelim. $\mathbf{X}$'nin bir $n \times m$ matrisi, $\mathbf{U}$'nun $n\times r$ ve $\mathbf{V}$'nin $r\times m$ olduğunu varsayalım. Hesaplamayı deneyelim

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`

Bu hesaplama, matris çarpanlarına ayırma adı verilen bir alanda önemlidir. Ancak bizim için bu, hesaplanması gereken bir türevdir. Bunun $1\times 1$ matrisler için ne olacağını hayal etmeye çalışalım. Bu durumda ifadeyi alırız

$$ 
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$

Burada türev oldukça standarttır. Bunu tekrar bir matris ifadesine dönüştürmeye çalışırsak,

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$

Ancak buna bakarsak pek işe yaramıyor. $\mathbf{X}$'in $n\times m$ olduğunu ve $\mathbf{U}\mathbf{V}$ öyle olduğunu hatırlayın, dolayısıyla $2(\mathbf{X} - \mathbf{U}\mathbf{V})$, $n\times m$'dir. Öte yandan, $\mathbf{U}$, $n\times r$'dir ve boyutlar eşleşmediğinden, $n\times m$ ve a $n\times r$ matrisleri çarpamayız!

$\frac{d}{d\mathbf{V}}$'i elde etmek istiyoruz, bu $\mathbf{V}$ ile aynı şekle sahip, ki bu $r \times m$. Öyleyse bir şekilde bir $n\times m$ matrisi ve bir $n\times r$ matrisi almalıyız, bunları bir $r \times m$ matris elde etmek için birbirleriyle çarpmalıyız (belki bazı devriklerle). Bunu $\mathbf{U}^\top$'yu $(\mathbf{X} - \mathbf{U}\mathbf{V})$ ile çarparak yapabiliriz. Böylelikle, :eqref:`eq_mat_goal_2` için çözümü tahmin edebiliriz:

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$ 

Bunun işe yaradığını göstermek için, ayrıntılı bir hesaplama sağlamazsak ihmal etmiş oluruz. Bu pratik kuralın işe yaradığına zaten inanıyorsanız, bu türetmeyi atlamaktan çekinmeyin. Hesaplayalım:

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$

Her $a$ ve $b$ için bulmalıyız.

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$

$\mathbf{X}$ and $\mathbf{U}$'nin tüm girdilerinin $\frac{d}{dv_{ab}}$ bakımında sabitler olduğunu hatırlayarak, türevi toplamın içine itebiliriz, ve zincir kuralını kareye uygularız:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$

Önceki türetmede olduğu gibi, $\frac{dv_{kj}}{dv_{ab}}$’nın yalnızca $k = a$ ve $j = b$ ise sıfırdan farklı olduğunu görebiliriz. Bu koşullardan herhangi biri geçerli değilse, toplamdaki terim sıfırdır ve onu özgürce atabiliriz. Bunu görüyoruz:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$

Buradaki önemli bir incelik, $k = a$ şartının iç toplamın içinde oluşmamasıdır, çünkü $k$ iç terimin içinde topladığımız yapay bir değişken. Gösterimsel olarak daha temiz bir örnek için nedenini düşünelim:

$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$

Bu noktadan itibaren, toplamın bileşenlerini belirlemeye başlayabiliriz. İlk olarak,

$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$

Yani toplamın içindeki tüm ifade:

$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

Bu, artık türevimizi şu şekilde yazabileceğimiz anlamına gelir:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$

Bunun bir matrisin $a, b$ öğesi gibi görünmesini istiyoruz ki böylece bir matris ifadesine ulaşmak için önceki örnekte olduğu gibi tekniği kullanabilelim, bu da indislerin sırasını $u_{ia}$ üzerinden değiştirmemiz gerektiği anlamına gelir. $u_{ia} = [\mathbf{U}^\top]_{ai}$ olduğunu fark edersek, bunu yazabiliriz:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

Bu bir matris çarpımıdır ve dolayısıyla şu sonuca varabiliriz:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$

Böylece çözümü şu şekilde yazabiliriz :eqref:`eq_mat_goal_2`

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

Bu, yukarıda tahmin ettiğimiz çözüme uyuyor!

Bu noktada şunu sormak mantıklıdır, "Neden öğrendiğim tüm hesap (kalkülüs) kurallarının matris versiyonlarını yazamıyorum? Bunun hala mekanik olduğu açık. Neden bunun üstesinden gelmiyoruz!" Gerçekten de böyle kurallar var ve :cite:`Petersen.Pedersen.ea.2008` mükemmel bir özet sunuyor. Bununla birlikte, matris işlemlerinin tekli değerlere kıyasla birleştirilebileceği çok sayıda yol olması nedeniyle, tek değişkenli olanlara göre çok daha fazla matris türev kuralı vardır. Genellikle en iyisi indislerle çalışmak veya uygun olduğunda bunu otomatik türev almaya bırakmaktır.

## Özet

* Daha yüksek boyutlarda, bir boyuttaki türevlerle aynı amaca hizmet eden gradyanları tanımlayabiliriz. Bunlar, girdilerde rastgele küçük bir değişiklik yaptığımızda çok değişkenli bir fonksiyonun nasıl değiştiğini görmemizi sağlar.
* Geri yayma algoritması, birçok kısmi türevin verimli bir şekilde hesaplanmasına izin vermek için çok değişkenli zincir kuralını düzenlemenin bir yöntemi olarak görülebilir.
* Matris hesabı, matris ifadelerinin türevlerini öz olarak yazmamızı sağlar.

## Alıştırmalar
1. $\boldsymbol{\beta}$ sütun vektörü verildiğinde, hem $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ hem de $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$ türevlerini hesaplayınız. Neden aynı cevabı alıyorsunuz?
2. $\mathbf{v}$ bir $n$ boyutlu vektör olsun. $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$ nedir?
3. $L(x, y) = \log(e^x + e^y)$ olsun. Gradyanını hesaplayınız. Gradyanın bileşenlerinin toplamı nedir?
4. $f(x, y) = x^2y + xy^2$ olsun. Tek kritik noktanın $(0,0)$ olduğunu gösterin. $f(x, x)$'i dikkate alarak, $(0,0)$'ın maksimum mu, minimum mu olduğunu veya hiçbiri olmadığını belirleyin.
5. $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$ işlevini küçülttüğümüzü varsayalım. $\nabla f = 0$ koşulunu $g$ ve $h$ cinsinden geometrik olarak nasıl yorumlayabiliriz?


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1090)
:end_tab:


:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1091)
:end_tab:
