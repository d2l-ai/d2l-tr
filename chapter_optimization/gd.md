# Gradyan İnişi
:label:`sec_gd`

Bu bölümde *gradyan inişi*nin altında yatan temel kavramları tanıtacağız. Derin öğrenmede nadiren kullanılmasına rağmen, gradyan inişi anlamak rasgele gradyan iniş algoritmalarını anlamak için anahtardır. Örneğin, optimizasyon problemi aşırı büyük bir öğrenme oranı nedeniyle ıraksayabilir. Bu olay halihazırda gradyan inişinde görülebilir. Benzer şekilde, ön şartlandırma gradyan inişinde yaygın bir tekniktir ve daha gelişmiş algoritmalara taşır. Basit bir özel durumla başlayalım. 

## Tek Boyutlu Gradyan İnişi

Bir boyuttaki gradyan iniş, gradyan iniş algoritmasının amaç işlevinin değerini neden azaltabileceğini açıklamak için mükemmel bir örnektir. Bir sürekli türevlenebilir gerçek değerli fonksiyon düşünün $f: \mathbb{R} \rightarrow \mathbb{R}$. Bir Taylor açılımı kullanarak şunu elde ederiz:

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Yani, birinci dereceden açılımda $f(x+\epsilon)$, $f(x)$ fonksiyon değeri ve $x$'deki birinci türev $f'(x)$ tarafından verilir. Küçük $\epsilon$ için negatif gradyan yönünde hareket etmenin $f$'nın azalacağını varsaymak mantıksız değildir. İşleri basit tutmak için sabit bir adım boyutu $\eta > 0$ seçip $\epsilon = -\eta f'(x)$'i seçeriz. Bunu yukarıdaki Taylor açılımı içine koyarsak: 

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Türev $f'(x) \neq 0$ kaybolmazsa $\eta f'^2(x)>0$'dan dolayı ilerleme kaydediyoruz. Dahası, daha yüksek dereceden terimlerin alakasız hale gelmesi için $\eta$'yi her zaman yeterince küçük seçebiliriz. Bu nedenle şuraya varırız: 

$$f(x - \eta f'(x)) \lessapprox f(x).$$

Bu demektir ki, bunu kullanırsak 

$$x \leftarrow x - \eta f'(x)$$

$x$ yinelemek için $f(x)$ işlevinin değeri azalabilir. Bu nedenle, gradyan inişinde ilk $x$ değeri ve sabit bir $\eta > 0$ değeri seçiyoruz ve daha sonra onları durdurma koşuluna ulaşılana kadar $x$'i sürekli yinelemek için kullanıyoruz, örneğin $|f'(x)|$'in büyüklüğü yeterince küçük olduğunda veya yineleme sayısı belirli bir değere ulaşınca. 

Basitlik için, gradyan inişinin nasıl uygulanacağını göstermek için $f(x)=x^2$ amaç işlevini seçiyoruz. $x=0$'in $f(x)$'yi en aza indiren çözüm olduğunu bilmemize rağmen, $x$'in nasıl değiştiğini gözlemlemek için gene de bu basit işlevi kullanıyoruz.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Amaç fonksiyonu
    return x ** 2

def f_grad(x):  # Amaç fonksiyonunun gradyanı (türevi)
    return 2 * x
```

Ardından, ilk değer olarak $x=10$'u kullanıyoruz ve $\eta=0.2$'yi varsayıyoruz. $x$'i 10 kez yinelemek için gradyan iniş kullanırsak, sonunda $x$ değerinin en iyi çözüme yaklaştığını görebiliriz.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'donem 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

$x$ üzerinde optimizasyonun ilerlemesi aşağıdaki gibi çizilebilir.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Öğrenme Oranı
:label:`subsec_gd-learningrate`

$\eta$ öğrenme oranı algoritma tasarımcısı tarafından ayarlanabilir. Çok küçük bir öğrenme oranı kullanırsak, $x$'in çok yavaş güncellenmesine neden olur ve daha iyi bir çözüm elde etmek için daha fazla yineleme gerektirir. Böyle bir durumda ne olduğunu göstermek için, $\eta = 0.05$ için aynı optimizasyon problemindeki ilerlemeyi göz önünde bulundurun. Gördüğümüz gibi, 10 adımdan sonra bile en uygun çözümden çok uzaktayız.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

Tersine, aşırı yüksek öğrenim oranı kullanırsak, $\left|\eta f'(x)\right|$ birinci dereceden Taylor genişleme formülü için çok büyük olabilir. Yani, :eqref:`gd-taylor-2` denklemdeki $\mathcal{O}(\eta^2 f'^2(x))$ terimi önemli hale gelebilir. Bu durumda, $x$ yinelemesinin $f(x)$ değerini düşüreceğini garanti edemeyiz. Örneğin, öğrenme oranını $\eta=1.1$ olarak ayarladığımızda, $x$ optimal çözümü $x=0$'i geçersiz kılar ve kademeli olarak ıraksar.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### Yerel Minimum

Dışbükey olmayan fonksiyonlarde ne olduğunu göstermek için, bazı sabit $c$ için $f(x) = x \cdot \cos(cx)$ durumunu düşünün. Bu işlevde sonsuz sayıda yerel minimum vardır. Öğrenme oranının seçimimize bağlı olarak ve sorunun ne kadar iyi şartlandırıldığına bağlı olarak, birçok çözümden birine varabiliriz. Aşağıdaki örnek, (gerçekçi olmayan) yüksek öğrenme oranının nasıl vasat bir yerel minimum seviyesine yol açacağını göstermektedir.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Amaç fonksiyonu
    return x * d2l.cos(c * x)

def f_grad(x):  # Amaç fonksiyonunun gradyanı (türevi)
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Çok Değişkenli Gradyan İnişi

Artık tek değişkenli durumun daha iyi bir sezgisine sahip olduğumuza göre, $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$'nın bulunduğu hali ele alalım. Yani, amaç fonksiyonu $f: \mathbb{R}^d \to \mathbb{R}$ vektörleri skalerlere eşlesin. Buna göre gradyan çok değişkenli. $d$ tane kısmi türevden oluşan bir vektördür: 

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

Gradyandaki her kısmi türev elemanı $\partial f(\mathbf{x})/\partial x_i$, $x_i$ girdisine göre $\mathbf{x}$'deki $f$ değişiminin oranını gösterir. Tek değişkenli durumda daha önce olduğu gibi, ne yapmamız gerektiğine dair bir fikir edinmek için çok değişkenli fonksiyonlarla ilgili Taylor açılımını kullanabiliriz. Özellikle, elimizde şu var: 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Başka bir deyişle, $\boldsymbol{\epsilon}$'te ikinci dereceden terimlere kadar en dik iniş yönü $-\nabla f(\mathbf{x})$ negatif gradyan ile verilir. Uygun bir öğrenme hızı $\eta > 0$ seçilmesi, prototipik gradyan iniş algoritmasını sağlar: 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Algoritmanın pratikte nasıl davrandığını görmek için girdi olarak iki boyutlu vektörü, $\mathbf{x} = [x_1, x_2]^\top$, çıktı olarak bir skaleri ile bir amaç fonksiyonu, $f(\mathbf{x})=x_1^2+2x_2^2$, oluşturalım. Gradyan $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$ ile verilir. $\mathbf{x}$'in yörüngesini $[-5, -2]$ ilk konumundan gradyan inişle gözlemleyeceğiz. 

Başlangıç olarak, iki yardımcı fonksiyona daha ihtiyacımız var. Birincisi bir güncelleme işlevi kullanır ve ilk değere 20 kez uygular. İkinci yardımcı $\mathbf{x}$'in yörüngesini görselleştirir.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Özelleştirilmiş bir eğitici ile 2B amaç işlevini optimize edin."""
    # `s1` ve `s2` daha sonra kullanılacak dahili durum değişkenleridir
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):  #@save
    """Optimizasyon sırasında 2B değişkenlerin izini gösterin."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Daha sonra, $\eta = 0.1$ öğrenme oranı için optimizasyon değişkeninin $\mathbf{x}$ yörüngesini gözlemliyoruz. 20 adımdan sonra $\mathbf{x}$ değerinin minimuma $[0, 0]$'da yaklaştığını görüyoruz. İlerleme oldukça yavaş olsa da oldukça iyi huyludur.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Amaç fonksiyonu
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Amaç fonksiyonun gradyanı
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Uyarlamalı Yöntemler

:numref:`subsec_gd-learningrate` içinde görebildiğimiz gibi, “doğru” $\eta$ öğrenme oranını elde etmek çetrefillidir. Eğer çok küçük seçersek, çok az ilerleme kaydederiz. Eğer çok büyük seçersek, çözüm salınır ve en kötü ihtimalle ıraksayabilir. $\eta$'yi otomatik olarak belirleyebilirsek veya bir öğrenme oranı seçmek zorunda kalmaktan kurtulsak ne olur? Bu durumda sadece amaç fonksiyonun değerine ve gradyanlarına değil, aynı zamanda *eğri*sinin değerine de bakan ikinci dereceden yöntemler yardımcı olabilir. Bu yöntemler, hesaplama maliyeti nedeniyle doğrudan derin öğrenmeye uygulanamamakla birlikte, aşağıda belirtilen algoritmaların arzu edilen özelliklerinin çoğunu taklit eden gelişmiş optimizasyon algoritmalarının nasıl tasarlanacağına dair yararlı önseziler sağlarlar. 

### Newton Yöntemi

 $f: \mathbb{R}^d \rightarrow \mathbb{R}$ fonksiyonun Taylor açılımının gözden geçirirsek, ilk dönemden sonra durmaya gerek yoktur. Aslında, böyle yazabilirsiniz:

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

Hantal notasyondan kaçınmak için $\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2 f(\mathbf{x})$'i $f$'in Hessian'i olarak tanımlıyoruz, ki bu bir $d \times d$ matristir. Küçük $d$ ve basit sorunlar için $\mathbf{H}$'nin hesaplaması kolaydır. Öte yandan derin sinir ağları için $\mathbf{H}$, $\mathcal{O}(d^2)$ girdilerinin depolanması maliyeti nedeniyle yasaklayıcı derecede büyük olabilir. Ayrıca geri yayma yoluyla hesaplamak çok pahalı olabilir. Şimdilik böyle düşünceleri göz ardı edelim ve hangi algoritmayı alacağımıza bakalım. 

Sonuçta, minimum $f$, $\nabla f = 0$'i karşılar. :numref:`subsec_calculus-grad` içindeki kalkülüs kurallarını izleyerek :eqref:`gd-hot-taylor` türevlerini $\boldsymbol{\epsilon}$ ile ilgili olarak alıp ve yüksek dereceden terimleri göz ardı ederek şuna varırız:

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ ve böylece }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Yani, optimizasyon probleminin bir parçası olarak Hessian $\mathbf{H}$'nin tersini almalıyız. 

Basit bir örnek olarak, $f(x) = \frac{1}{2} x^2$ için $\nabla f(x) = x$ ve $\mathbf{H} = 1$'e sahibiz. Bu nedenle herhangi bir $x$ için $\epsilon = -x$ elde ederiz. Başka bir deyişle, bir *tek* adım, herhangi bir ayarlamaya gerek kalmadan mükemmel bir şekilde yakınsama için yeterlidir! Ne yazık ki, burada biraz şanslıyız: Taylor açılımı $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$'ten dolayı kesindir.  

Diğer problemlerde neler olduğunu görelim. Bazı sabit $c$ için dışbükey hiperbolik kosinüs fonksiyonu $f(x) = \cosh(cx)$ göz önüne alındığında, $x=0$'daki küresel minimum seviyeye birkaç yinelemeden sonra ulaşıldığını görebiliriz.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Amaç fonksiyonu
    return d2l.cosh(c * x)

def f_grad(x):  # Amaç fonksiyonun gradyanı
    return c * d2l.sinh(c * x)

def f_hess(x):  # Amaç fonksiyonunun Hessian'i
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('donem 10, x:', x)
    return results

show_trace(newton(), f)
```

Şimdi bir sabit $c$ için $f(x) = x \cos(c x)$ gibi bir *dışbükey olmayan* işlevi düşünelim. Herşeyin sonunda, Newton'un yönteminde Hessian'a böldüğümüze dikkat edin. Bu, ikinci türev *negatif* ise $f$ değeri *artma* yönüne doğru yürüyebileceğimiz anlamına gelir. Bu algoritmanın ölümcül kusurudur. Pratikte neler olduğunu görelim.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Amaç fonksiyonu
    return x * d2l.cos(c * x)

def f_grad(x):  # Amaç fonksiyonun gradyanı
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Amaç fonksiyonun Hessian'i
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

Olağanüstü derecede yanlış gitti. Bunu nasıl düzeltebiliriz? Bunun yerine mutlak değerini alarak Hessian'i “düzeltmek” bir yol olabilir. Başka bir strateji de öğrenme oranını geri getirmektir. Bu amacı yenmek gibi görünüyor, ama tam olarak değil. İkinci dereceden bilgilere sahip olmak, eğrilik büyük olduğunda dikkatli olmamızı ve amaç fonksiyon daha düz olduğunda daha uzun adımlar atmamızı sağlar. Bunun biraz daha küçük bir öğrenme oranıyla nasıl çalıştığını görelim, $\eta = 0.5$ diyelim. Gördüğümüz gibi, oldukça verimli bir algoritmamız var.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### Yakınsaklık Analizi

Biz sadece bazı dışbükey ve üç kez türevlenebilir amaç fonksiyonu $f$ için Newton yönteminin yakınsama oranını analiz ediyoruz, burada ikinci türev sıfırdan farklı, yani $f'' > 0$. Çok değişkenli kanıt, aşağıdaki tek boyutlu argümanın basit bir uzantısıdır ve sezgi açısından bize pek yardımcı olmadığından ihmal edilir. 

$k$'inci yinelemesinde $x$ değerini $x^{(k)}$ ile belirtelim ve $e^{(k)} \stackrel{\mathrm{def}}{=} x^{(k)} - x^*$ yinelemesinde eniyilikten uzaklık olmasına izin verelim. Taylor açılımı ile $f'(x^*) = 0$ durumu aşağıdaki gibi yazılabilir:

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

ki bu bazı $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$ için tutar. Yukarıdaki açılımı $f''(x^{(k)})$ ile bölmek aşağıdaki ifadeye yol açar: 

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

$x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$ güncellemesine sahip olduğumuzu hatırlayın. Bu güncelleme denklemini yerine koyarak ve her iki tarafın mutlak değerini alarak şuna ulaşırız:

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

Sonuç olarak, $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ ile sınırlı olan bir bölgede olduğumuzda, dördüncü dereceden azalan bir hatamız var:  

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

Bir taraftan, optimizasyon araştırmacıları buna *doğrusal* yakınsama derler, oysa $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ gibi bir koşul bir *sabit* yakınsama oranı olarak adlandırılır. Bu analizin bir dizi uyarıyla birlikte geldiğini unutmayın. İlk olarak, hızlı yakınsama bölgesine ulaşacağımıza dair gerçekten çok fazla bir garantimiz yok. Bunun yerine, sadece bir kez ulaştığımızda yakınsamanın çok hızlı olacağını biliyoruz. İkincisi, bu analiz $f$'in daha yüksek mertebeden türevlere kadar iyi-huylu olmasını gerektirir. $f$'ın değerlerini nasıl değiştirebileceği konusunda "şaşırtıcı" özelliklere sahip olmamasını sağlamaya geliyor.

### Ön Şartlandırma

Oldukça şaşırtıcı bir şekilde hesaplamak ve tam Hessian'ı saklamak çok pahalıdır. Alternatifler bulmak bu nedenle arzu edilir. Meseleleri iyileştirmenin bir yolu da *ön şartlandırma*dır. Hessian'ın bütünüyle hesaplanmasını önler, sadece *köşesel* girdilerini hesaplar. Bu, aşağıdaki biçimdeki algoritmaların güncellenmesine yol açar:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

Bu tam Newton'un yöntemi kadar iyi olmasa da, onu kullanmamaktan çok daha iyidir. Bunun neden iyi bir fikir olabileceğini görmek için bir değişkenin milimetre cinsinden yüksekliği ve diğeri kilometre cinsinden yüksekliği belirten bir durum göz önünde bulundurun. Her ikisi için doğal ölçeğin de metre olarak olduğunu varsayarsak, parametrelemelerde korkunç bir uyumsuzluğumuz var. Neyse ki, ön şartlandırma kullanmak bunu ortadan kaldırır. Gradyan inişi ile etkin bir şekilde ön koşullandırma, her değişken için farklı bir öğrenme oranı seçilmesi anlamına gelir ($\mathbf{x}$ vektörünün koordinatı). Daha sonra göreceğimiz gibi ön şartlandırma, rasgele gradyan iniş optimizasyon algoritmalarındaki bazı yaratıcılığı yönlendirir.  

### Doğru Üzerinde Arama ile Gradyan İnişi

Gradyan inişindeki en önemli sorunlardan biri, hedefi aşmamız veya yetersiz ilerleme kaydetmemizdir. Sorun için basit bir düzeltme, gradyan iniş ile birlikte doğru üzerinde arama kullanmaktır. Yani, $\nabla f(\mathbf{x})$ tarafından verilen yönü kullanırız ve ardından $\eta$ öğrenme oranının $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$'yi en aza indirmek için ikili arama yaparız.

Bu algoritma hızla yakınsar (analiz ve kanıt için bkz., :cite:`Boyd.Vandenberghe.2004`). Ancak, derin öğrenme amacıyla bu o kadar da mümkün değildir, çünkü doğru üzerinde aramanın her adımı, tüm veri kümesindeki amaç fonksiyonunu değerlendirmemizi gerektirir. Bunu tamamlamak çok maliyetlidir.

## Özet

* Öğrenme oranları önemlidir. Çok büyükse ıraksarız, çok küçükse ilerleme kaydetmeyiz.
* Gradyan inişi yerel minimumda takılabilir.
* Yüksek boyutlarda öğrenme oranının ayarlanması karmaşıktır.
* Ön şartlandırma ayarların ölçeklendirilmesine yardımcı olabilir.
* Newton'un yöntemi dışbükey problemlerde düzgün çalışmaya başladıktan sonra çok daha hızlıdır.
* Dışbükey olmayan problemler için herhangi bir ayarlama yapmadan Newton'un yöntemini kullanmaktan sakının.

## Alıştırmalar

1. Gradyan inişi için farklı öğrenme oranları ve amaç fonksiyonları ile deney yapın.
1. $[a, b]$ aralığında dışbükey işlevini en aza indirmek için doğru üzerinde arama gerçekleştirin.
    1. İkili arama için türevlere ihtiyacınız var mı, mesela $[a, (a+b)/2]$ veya $[(a+b)/2, b]$'yi seçip seçmeyeceğinize karar vermek için?
    1. Algoritma için yakınsama oranı ne kadar hızlıdır?
    1. Algoritmayı uygulayın ve $\log (\exp(x) + \exp(-2x -3))$'ü en aza indirmek için uygulayın.
1. Gradyan inişinin son derece yavaş olduğu $\mathbb{R}^2$'te tanımlanan bir amaç fonksiyonu tasarlayın. İpucu: Farklı koordinatları farklı şekilde ölçeklendirin.
1. Aşağıdaki ön şartlandırmaları kullanarak Newton yönteminin hafifsiklet versiyonunu uygulayın:
    1. Ön koşul olarak köşegen Hessian kullanın.
    1. Gerçek (muhtemelen işaretli) değerler yerine bunun mutlak değerlerini kullanın.
    1. Bunu yukarıdaki probleme uygulayın.
1. Yukarıdaki algoritmayı bir takım amaç fonksiyonuna uygulayın (dışbükey olan veya olmayan). Koordinatları $45$ derece döndürürseniz ne olur?

[Tartışmalar](https://discuss.d2l.ai/t/351)
