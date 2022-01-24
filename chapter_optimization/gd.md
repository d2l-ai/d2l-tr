# Degrade İniş
:label:`sec_gd`

Bu bölümde temel kavramları tanıtacağız *degrade descent*. Derin öğrenmede nadiren kullanılmasına rağmen, degrade iniş anlayışı stokastik degrade iniş algoritmalarını anlamak için anahtardır. Örneğin, optimizasyon sorunu aşırı büyük bir öğrenme oranı nedeniyle farklılaşabilir. Bu fenomen zaten degrade inişinde görülebilir. Benzer şekilde, ön şartlandırma degrade inişinde yaygın bir tekniktir ve daha gelişmiş algoritmalara taşır. Basit bir özel durumla başlayalım. 

## Tek Boyutlu Degrade İniş

Bir boyuttaki degrade iniş, degrade iniş algoritmasının nesnel işlevin değerini neden azaltabileceğini açıklamak için mükemmel bir örnektir. Bazı sürekli diferansiyellenebilir gerçek değerli fonksiyon düşünün $f: \mathbb{R} \rightarrow \mathbb{R}$. Elde ettiğimiz bir Taylor genişleme kullanarak 

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Yani, birinci dereceden yaklaşımda $f(x+\epsilon)$ fonksiyon değeri $f(x)$ ve $x$'de $x$'de birinci türev $f'(x)$ ile verilir. Küçük $\epsilon$ için negatif degrade yönünde hareket etmenin $f$'nın azalacağını varsaymak mantıksız değildir. İşleri basit tutmak için sabit bir adım boyutu $\eta > 0$ seçip $\epsilon = -\eta f'(x)$'i seçiyoruz. Bunu yukarıdaki Taylor genişleme içine takarak 

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Türev $f'(x) \neq 0$ kaybolmazsa $\eta f'^2(x)>0$'ten beri ilerleme kaydediyoruz. Dahası, her zaman daha yüksek dereceden terimlerin alakasız hale gelmesi için $\eta$'ü yeterince küçük seçebiliriz. Bu nedenle biz varmak 

$$f(x - \eta f'(x)) \lessapprox f(x).$$

Bu demektir ki, biz kullanırsanız 

$$x \leftarrow x - \eta f'(x)$$

$x$ yinelemek için $f(x)$ işlevinin değeri azalabilir. Bu nedenle, degrade inişinde ilk olarak $x$ ve sabit bir $\eta > 0$ başlangıç değeri seçiyoruz ve daha sonra durdurma koşuluna ulaşılana kadar $x$'i sürekli yinelemek için kullanıyoruz, örneğin $|f'(x)|$'ün büyüklüğü yeterince küçük olduğunda veya yineleme sayısı belirli bir değeri. 

Basitlik için, degrade inişinin nasıl uygulanacağını göstermek için $f(x)=x^2$ nesnel işlevini seçiyoruz. $x=0$'ün $f(x)$'yi en aza indirmek için çözüm olduğunu bilmemize rağmen, $x$'nin nasıl değiştiğini gözlemlemek için hala bu basit işlevi kullanıyoruz.

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
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

Ardından, başlangıç değeri olarak $x=10$'i kullanıyoruz ve $\eta=0.2$'ü varsayıyoruz. $x$'yı 10 kez yinelemek için degrade iniş kullanarak, sonunda $x$ değerinin en uygun çözüme yaklaştığını görebiliriz.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

$x$ üzerinde optimizasyon ilerlemesi aşağıdaki gibi çizilebilir.

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

### Öğrenme Hızı
:label:`subsec_gd-learningrate`

$\eta$ öğrenme hızı algoritma tasarımcısı tarafından ayarlanabilir. Çok küçük bir öğrenme hızı kullanırsak, $x$'ün çok yavaş güncellenmesine neden olur ve daha iyi bir çözüm elde etmek için daha fazla yineleme gerektirir. Böyle bir durumda ne olduğunu göstermek için, $\eta = 0.05$ için aynı optimizasyon sorunundaki ilerlemeyi göz önünde bulundurun. Gördüğümüz gibi, 10 adımdan sonra bile en uygun çözümden çok uzaktayız.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

Tersine, aşırı yüksek öğrenim oranı kullanırsak, $\left|\eta f'(x)\right|$ birinci dereceden Taylor genişleme formülü için çok büyük olabilir. Yani, :eqref:`gd-taylor-2`'teki $\mathcal{O}(\eta^2 f'^2(x))$ terimi önemli hale gelebilir. Bu durumda, $x$ yinelemesinin $f(x)$ değerini düşüreceğini garanti edemeyiz. Örneğin, öğrenme oranını $\eta=1.1$ olarak ayarladığımızda, $x$ optimal çözümü $x=0$'i geçersiz kılar ve kademeli olarak ayrılır.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### Yerel Minima

Dışbükey olmayan işlevler için ne olduğunu göstermek için $f(x) = x \cdot \cos(cx)$ bazı sabit $c$ için durumu göz önünde bulundurun. Bu işlev sonsuz sayıda yerel minima vardır. Öğrenme oranının seçimimize bağlı olarak ve sorunun ne kadar iyi şartlandırıldığına bağlı olarak, birçok çözümden biriyle sonuçlanabiliriz. Aşağıdaki örnek, (gerçekçi olmayan) yüksek öğrenme oranının yerel asgari seviyeye nasıl yol açacağını göstermektedir.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Çok Değişkenli Degrade İniş

Artık tek değişkenli davanın daha iyi bir sezgiye sahip olduğumuza göre, $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$'nın bulunduğu durumu ele alalım. Yani, objektif fonksiyon $f: \mathbb{R}^d \to \mathbb{R}$ vektörleri skalerlere haritalar. Buna göre degrade çok değişkenli. $d$ kısmi türevlerinden oluşan bir vektördür: 

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

Degradedeki her kısmi türev eleman $\partial f(\mathbf{x})/\partial x_i$ $x_i$ girdisine göre $\mathbf{x}$'da $f$ değişiminin oranını gösterir. Tek değişkenli durumda daha önce olduğu gibi, ne yapmamız gerektiğine dair bir fikir edinmek için çok değişkenli fonksiyonlar için ilgili Taylor yaklaşımını kullanabiliriz. Özellikle, bu var 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Başka bir deyişle, $\boldsymbol{\epsilon}$'te ikinci dereceden terimlere kadar en dik iniş yönü $-\nabla f(\mathbf{x})$ negatif degrade ile verilir. Uygun bir öğrenme hızı $\eta > 0$ seçilmesi, prototipik degrade iniş algoritmasını sağlar: 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Algoritmanın pratikte nasıl davrandığını görmek için $f(\mathbf{x})=x_1^2+2x_2^2$ giriş olarak iki boyutlu vektör $\mathbf{x} = [x_1, x_2]^\top$ ve çıktı olarak bir skaler ile objektif bir fonksiyon oluşturalım. Degrade $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$ tarafından verilir. $\mathbf{x}$ yörüngesini ilk konumdan degrade iniş ile $[-5, -2]$ yörüngesini gözlemleyeceğiz.  

Başlangıç olarak, iki yardımcı fonksiyona daha ihtiyacımız var. Birincisi bir güncelleme işlevi kullanır ve ilk değere 20 kez uygular. İkinci yardımcı $\mathbf{x}$'ün yörüngesini görselleştirir.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used later
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
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Daha sonra, $\eta = 0.1$ öğrenme hızı için optimizasyon değişkeninin $\mathbf{x}$ yörüngesini gözlemliyoruz. 20 adımdan sonra $\mathbf{x}$ değerinin minimuma $[0, 0]$'te yaklaştığını görüyoruz. İlerleme oldukça yavaş olsa da oldukça iyi davranılır.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Adaptif Yöntemler

:numref:`subsec_gd-learningrate`'te görebildiğimiz gibi, $\eta$ “doğru” öğrenme oranını elde etmek zor. Eğer çok küçük seçersek, çok az ilerleme kaydederiz. Eğer çok büyük seçersek, çözelti salınır ve en kötü ihtimalle ayrışabilir. $\eta$'i otomatik olarak belirleyebilirsek veya bir öğrenme oranı seçmek zorunda kalmaktan kurtulsak ne olur? Bu durumda sadece objektif fonksiyonun değerine ve degradelerine değil, aynı zamanda *eğrisi* değerine de bakan ikinci dereceden yöntemler yardımcı olabilir. Bu yöntemler, hesaplama maliyeti nedeniyle doğrudan derin öğrenmeye uygulanamamakla birlikte, aşağıda belirtilen algoritmaların arzu edilen özelliklerinin çoğunu taklit eden gelişmiş optimizasyon algoritmalarının nasıl tasarlanacağına dair yararlı önseziler sağlarlar. 

### Newton Yöntemi

Bazı fonksiyonların Taylor genişlemesinin gözden geçirilmesi $f: \mathbb{R}^d \rightarrow \mathbb{R}$ ilk dönemden sonra durmaya gerek yoktur. Aslında, biz olarak yazabilirsiniz 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

Hantal notasyondan kaçınmak için $\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2 f(\mathbf{x})$'u $f$ numaralı Hessian olarak tanımlıyoruz, ki bu bir $d \times d$ matrisi. Küçük $d$ ve basit sorunlar için $\mathbf{H}$ hesaplaması kolaydır. Öte yandan derin sinir ağları için $\mathbf{H}$, $\mathcal{O}(d^2)$ girişlerinin depolanması maliyeti nedeniyle yasaklayıcı derecede büyük olabilir. Ayrıca geri yayılma yoluyla hesaplamak çok pahalı olabilir. Şimdilik böyle düşüncelere göz ardı edelim ve hangi algoritmayı alacağımıza bakalım. 

Sonuçta, minimum $f$ $\nabla f = 0$'i karşılar. :numref:`subsec_calculus-grad`'teki hesap kurallarını izleyerek :eqref:`gd-hot-taylor` türevlerini $\boldsymbol{\epsilon}$ ile ilgili olarak alarak ve vardığımız yüksek dereceden terimleri göz ardı ederek 

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Yani, optimizasyon probleminin bir parçası olarak Hessian $\mathbf{H}$'ü ters çevirmeliyiz. 

Basit bir örnek olarak, $f(x) = \frac{1}{2} x^2$ için $\nabla f(x) = x$ ve $\mathbf{H} = 1$'ya sahibiz. Bu nedenle herhangi bir $x$ elde $\epsilon = -x$. Başka bir deyişle, bir *tek* adım, herhangi bir ayarlamaya gerek kalmadan mükemmel bir şekilde yakınsama için yeterlidir! Ne yazık ki, burada biraz şanslıyız: Taylor genişlemesi $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$'ten beri kesinti.  

Diğer sorunlarda neler olduğunu görelim. Bazı sabit $c$ için dışbükey hiperbolik kosinüs fonksiyonu $f(x) = \cosh(cx)$ göz önüne alındığında, $x=0$'teki küresel asgari seviyeye birkaç yinelemeden sonra ulaşıldığını görebiliriz.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

Şimdi $c$ sabit $c$ için $f(x) = x \cos(c x)$ gibi bir *dışbükeyici* işlevini düşünelim. Sonuçta, Newton'un yönteminde Hessian tarafından bölündüğümüze dikkat edin. Bu, ikinci türev ise *negatif* ise $f$ değeri*artırılmış* yönüne doğru yürüyebileceğimiz anlamına gelir. Bu algoritmanın ölümcül bir kusuru. Pratikte neler olduğunu görelim.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

Olağanüstü derecede yanlış gitti. Bunu nasıl düzeltebiliriz? Bunun yerine mutlak değerini alarak Hessian'ı “düzeltmek” için bir yol olacaktır. Başka bir strateji de öğrenme oranını geri getirmektir. Bu amacı yenmek gibi görünüyor, ama tam olarak değil. İkinci dereceden bilgilere sahip olmak, eğrilik büyük olduğunda dikkatli olmamızı ve objektif fonksiyon daha düz olduğunda daha uzun adımlar atmamızı sağlar. Bunun biraz daha küçük bir öğrenme hızıyla nasıl çalıştığını görelim, $\eta = 0.5$ diyelim. Gördüğümüz gibi, oldukça verimli bir algoritmamız var.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### Yakınsaklık Analizi

Biz sadece bazı dışbükey ve üç kez diferansiyellenebilir objektif fonksiyon $f$ için Newton yönteminin yakınsama oranını analiz ediyoruz, burada ikinci türev sıfır, yani $f'' > 0$. Çok değişkenli kanıt, aşağıdaki tek boyutlu argümanın basit bir uzantısıdır ve sezgi açısından bize pek yardımcı olmadığından ihmal edilir. 

$k^\mathrm{th}$ yinelemesinde $x$ değerini $x^{(k)}$ ile belirtin ve $e^{(k)} \stackrel{\mathrm{def}}{=} x^{(k)} - x^*$ yinelemesinde optimaliteden uzaklık olmasına izin verin. Taylor genişleme ile biz durum $f'(x^*) = 0$ olarak yazılabilir 

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

hangi bazı $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$ için tutar. Yukarıdaki genişleme $f''(x^{(k)})$ verimleri ile bölünmesi 

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

Güncellemeye sahip olduğumuzu hatırlayın $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$. Bu güncelleme denklemini takarak ve her iki tarafın mutlak değerini alarak 

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

Sonuç olarak, sınırlı $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ olan bir bölgede olduğumuzda, dört yönden azalan bir hatamız var  

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

Bir kenara olarak, optimizasyon araştırmacıları bu *doğrusal* yakınsama diyoruz, oysa $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ gibi bir koşul bir *sabit* yakınsama oranı olarak adlandırılır. Bu analizin bir dizi uyarıyla birlikte geldiğini unutmayın. İlk olarak, hızlı yakınsama bölgesine ulaşacağımız zaman gerçekten çok fazla bir garantimiz yok. Bunun yerine, sadece bir kez ulaştığımızda yakınsamanın çok hızlı olacağını biliyoruz. İkincisi, bu analiz $f$'ün daha yüksek mertebeden türevlere kadar iyi davranılmasını gerektirir. Bu $f$'ün değerlerini nasıl değiştirebileceği açısından herhangi bir “şaşırtıcı” özelliklere sahip olmamasını sağlamak için gelir. 

### Ön şartlandırma

Oldukça şaşırtıcı bir şekilde hesaplamak ve tam Hessian'ı saklamak çok pahalıdır. Alternatifler bulmak için bu nedenle arzu edilir. Mesleleri iyileştirmenin bir yolu da *ön koşullama*. Hessian'ın bütünüyle hesaplanmasını önler ancak sadece*diyagonal* girişlerini hesaplar. Bu, formun algoritmalarının güncellenmesine yol açar 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

Bu tam Newton'un yöntemi kadar iyi olmasa da, onu kullanmamaktan çok daha iyidir. Bunun neden iyi bir fikir olabileceğini görmek için bir değişkenin milimetre cinsinden yüksekliği ve diğeri kilometre cinsinden yüksekliği belirten bir durum göz önünde bulundurun. Her iki doğal ölçeğin de metre olarak olduğunu varsayarsak, parametrelemelerde korkunç bir uyumsuzluğumuz var. Neyse ki, ön şartlandırma kullanmak bunu ortadan kaldırır. Her değişken için farklı bir öğrenme hızı (vektör $\mathbf{x}$ koordinatı) seçmek için degrade iniş miktarlarıyla etkili bir şekilde ön koşullandırma. Daha sonra göreceğimiz gibi ön şartlandırma, stokastik degrade iniş optimizasyon algoritmalarındaki bazı yeniliği yönlendiriyor.  

### Satır Araması ile Degrade İniş

Degrade inişindeki en önemli sorunlardan biri, hedefi aşabilir veya yetersiz ilerleme kaydedebilmemizdir. Sorun için basit bir düzeltme, degrade iniş ile birlikte satır aramasını kullanmaktır. Yani, $\nabla f(\mathbf{x})$ tarafından verilen yönü kullanıyoruz ve daha sonra $\eta$'nin $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$'i en aza indirdiği öğrenme oranının en aza indirildiği için ikili arama gerçekleştiriyoruz. 

Bu algoritma hızla yakınlaşır (analiz ve kanıt için bkz., :cite:`Boyd.Vandenberghe.2004`). Bununla birlikte, derin öğrenme amacıyla bu oldukça mümkün değildir, çünkü satır aramasının her adımı, tüm veri kümesinde objektif işlevi değerlendirmemizi gerektirecektir. Bunu başarmak için çok pahalıya mal oluyor. 

## Özet

* Öğrenme oranları önemlidir. Çok büyük ve biz sapmak, çok küçük ve ilerleme kaydetmiyoruz.
* Degrade iniş yerel minima sıkışmış alabilirsiniz.
* Yüksek boyutlarda öğrenme oranının ayarlanması karmaşıktır.
* Ön şartlandırma terazi ayarlamasına yardımcı olabilir.
* Newton'un yöntemi dışbükey problemlerde düzgün çalışmaya başladıktan sonra çok daha hızlıdır.
* Dışbükey olmayan sorunlar için herhangi bir ayarlama yapmadan Newton'un yöntemini kullanmaktan sakının.

## Egzersizler

1. Degrade iniş için farklı öğrenme oranları ve objektif fonksiyonlar ile deney yapın.
1. $[a, b]$ aralığında dışbükey işlevi en aza indirmek için satır araması gerçekleştirin.
    1. İkili arama için türevlere ihtiyacınız var mı, yani $[a, (a+b)/2]$ veya $[(a+b)/2, b]$'ü seçip seçmeyeceğinize karar vermek için.
    1. Algoritma için yakınsama oranı ne kadar hızlıdır?
    1. Algoritmayı uygulayın ve $\log (\exp(x) + \exp(-2x -3))$'ü en aza indirmek için uygulayın.
1. Degrade inişinin son derece yavaş olduğu $\mathbb{R}^2$'te tanımlanan objektif bir fonksiyon tasarlayın. İpucu: Farklı koordinatları farklı şekilde ölçeklendirin.
1. Ön şartlandırma kullanarak Newton yönteminin hafif versiyonunu uygulamak:
    1. Ön koşul olarak diyagonal Hessian kullanın.
    1. Gerçek (muhtemelen imzalanmış) değerler yerine bunun mutlak değerlerini kullanın.
    1. Bunu yukarıdaki soruna uygulayın.
1. Yukarıdaki algoritmayı bir dizi objektif fonksiyona uygulayın (dışbükey veya dışbükey). Koordinatları $45$ derece döndürürseniz ne olur?

[Discussions](https://discuss.d2l.ai/t/351)
