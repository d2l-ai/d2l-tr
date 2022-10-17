# Matrisi Çarpanlarına Ayırma 

Matris Çarpanlarına Ayırma (Ayrıştırma) :cite:`Koren.Bell.Volinsky.2009`, tavsiye sistemleri yazınında köklü bir algoritmadır. Matris ayrıştırma modelinin ilk sürümü Simon Funk tarafından, etkileşim matrisini ayrıştırma fikrini tarif ettiği ünlü bir [blog yazısı](https://sifter.org/~simon/journal/20061211.html) içinde önerilmiştir. Daha sonra 2006'da düzenlenen Netflix yarışması nedeniyle yaygın olarak tanındı. O dönemde medya akışı ve video kiralama şirketi Netflix, öneri sistemi başarımını artırmak için bir yarışma duyurdu. Netflix referans değerinden, yani Cinematch'ten, yüzde 10 oranında iyileştirebilen en iyi takım bir milyon ABD doları ödülü kazanacaktı. Bu nedenle, bu yarışma tavsiye sistemi araştırması alanına çok dikkat çekti. Daha sonra büyük ödül BellKor'un Pragmatik Kaos ekibi, BellKor, Pragmatik Teori ve BigChaos'un birleşmiş bir ekibi tarafından kazanıldı (şimdilik bu algoritmalar hakkında endişelenmenize gerek yok). Nihai sonuç bir topluluk çözümünün (yani birçok algoritmanın birleşimi) sonucu olmasına rağmen, matris ayrıştırma algoritması son karışımda kritik bir rol oynadı. Netflix Büyük Ödül çözümünün :cite:`Toscher.Jahrer.Bell.2009` teknik raporu, benimsenen modele ayrıntılı bir giriş sağlar. Bu bölümde, matris ayrıştırma modelinin ayrıntılarına ve uygulanmasına dalacağız. 

## Matrisi Çarpanlara Ayırma Modeli

Matris çarpanlarına ayırma, işbirlikçi filtreleme modellerinin bir sınıfıdır. Model özellikle, kullanıcı-öğe etkileşim matrisini (örn. derecelendirme matrisi) iki düşük kerteli matrisin çarpımına ayrıştırıp kullanıcı-öğe etkileşimlerinin düşük kerteli yapısını yakalar. 

$\mathbf{R} \in \mathbb{R}^{m \times n}$, $m$ kullanıcıları ve $n$ öğe ile etkileşim matrisi göstersin ve $\mathbf{R}$ değerleri açık derecelendirmeleri temsil eder. Kullanıcı-öğe etkileşimi bir kullanıcı gizli matrisi $\mathbf{P} \in \mathbb{R}^{m \times k}$ ve bir öğe gizli matrisi $\mathbf{Q} \in \mathbb{R}^{n \times k}$ şeklinde çarpanlara ayrılacaktır, burada $k \ll m, n$, gizli faktör boyutudur. $\mathbf{p}_u$ $\mathbf{P}$'nin $u.$  satırını ve $\mathbf{q}_i$ $\mathbf{Q}$'nin $i.$ satırını göstersin. Belirli bir $i$ öğesi için, $\mathbf{q}_i$ öğesinin elemanları, öğenin bir filmin türleri ve dilleri gibi özniteliklere ne ölçüde sahip olduğunu ölçer. Belirli bir kullanıcı $u$ için, $\mathbf{p}_u$ öğesinin elemanları, kullanıcının öğelere karşılık gelen özniteliklere ne kadar ilgi duyduğunu ölçer. Bu gizli faktörler, bu örneklerde belirtildiği gibi belirgin boyutları ölçebilir veya tamamen yorumlanamaz olabilir. Tahmini derecelendirme aşağıdaki gibi hesaplanabilir

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

burada $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$, $\mathbf{R}$ ile aynı şekle sahip tahmin edilen derecelendirme matrisidir. Bu tahmin kuralının önemli bir sorunu kullanıcıların/öğelerin yanlılıklarının modellenemez olmasıdır. Örneğin, bazı kullanıcılar daha yüksek derecelendirme verme eğilimindedir veya bazı öğeler düşük kalitesinden dolayı her zaman daha düşük derecelendirme alır. Bu yanlılıklar gerçek dünyadaki uygulamalarda yaygındır. Bu önyargıları yakalamak için, kullanıcıya özgü ve öğeye özgü yanlıklık terimleri tanıtıldı. Özellikle, $u$ kullanıcısının $i$ öğesine verdiği tahmini derecelendirme aşağıdaki gibi hesaplanır: 

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

Ardından, tahmin edilen derecelendirme puanları ile gerçek derecelendirme puanları arasındaki ortalama kare hatayı en aza indirerek matris çarpanlarına ayırma modelini eğitiyoruz. Amaç işlevi aşağıdaki gibi tanımlanır: 

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

burada $\lambda$ düzenlileştirme oranını gösterir. Düzenlileştirme terimi $\lambda (\| \mathbf{P} \|^2_F +\| \mathbf{Q}\|^2_F + b_u^2 + b_i^2 )$ parametrelerin büyüklüğünü cezalandırarak aşırı öğrenmeden kaçınmakta kullanılır. $\mathbf{R}_{ui}$'nin bilindiği $(u, i)$ çiftleri $\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \text{ biliniyor}\}$ kümesinde saklanır. Model parametreleri, rasgele gradyan inişi ve Adam gibi bir eniyileme algoritması ile öğrenilebilir. 

Matrisi çarpanlara ayırma modelinin sezgisel bir gösterimi aşağıda gösterilmiştir. 

![Matrisi çarpanlara ayırma modelinin resimlendirilmesi](../img/rec-mf.svg)

Bu bölümün geri kalanında, matris ayrıştırma uygulanmasını açıklayacağız ve modeli MovieLens veri kümesinde eğiteceğiz.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## Model Uygulaması

İlk olarak, yukarıda açıklanan matris ayrıştırma modelini uyguluyoruz. Kullanıcı ve öğe gizli etkenleri `nn.Embedding` ile oluşturulabilir. `input_dim` öğe/kullanıcı sayısıdır ve (`output_dim`) ise gizli etkenlerin boyutudur ($k$). `nn.Embedding`'yı `output_dim`'i bir olarak kurup kullanıcı/öğe yanlılıklarını oluşturmak için de kullanabiliriz. `forward` işlevinde, gömmeleri aramak için kullanıcı ve öğe kimlikleri kullanılır.

```{.python .input  n=4}
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## Değerlendirme Ölçümleri

Daha sonra, modelin tahmin ettiği derecelendirme puanları ile gerçekte gözlemlenen derecelendirme puanları (gerçek referans değer) :cite:`Gunawardana.Shani.2015` arasındaki farkları ölçmek için yaygın olarak kullanılan RMSE (ortalama kare hatasının kökü) ölçümünü uygularız. RMSE şu şekilde tanımlanır: 

$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

burada $\mathcal{T}$, değerlendirmek istediğiniz kullanıcıların ve öğelerin çiftlerinden oluşan kümedir. $|\mathcal{T}|$ bu kümenin boyutudur. `mx.metric` tarafından sağlanan RMSE işlevini kullanabiliriz.

```{.python .input  n=3}
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## Model Eğitimi ve Değerlendirilmesi

Eğitim fonksiyonunda, ağırlık sönümü ile $L_2$ kaybını benimsedik. Ağırlık sönümü mekanizması, $L_2$ düzenlileştirmesi ile aynı etkiye sahiptir.

```{.python .input  n=4}
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[0:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Son olarak, her şeyi bir araya koyalım ve modeli eğitelim. Burada, gizli çarpan boyutunu 30'a ayarlıyoruz.

```{.python .input  n=5}
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

Aşağıda, bir kullanıcının (ID 20) bir öğeye (ID 30) verebileceği derecelendirmeyi tahmin etmek için eğitimli modeli kullanıyoruz .

```{.python .input  n=6}
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## Özet

* Matrisi çarpanlara ayırma modeli, tavsiye sistemlerinde yaygın olarak kullanılmaktadır. Kullanıcının bir öğeye verebileceği derecelendirmeleri tahmin etmek için kullanılabilir.
* Tavsiye sistemleri için matris ayrıştırma uygulayabilir ve eğitebiliriz.

## Alıştırmalar

* Gizli çarpanların boyutunu değiştirin. Gizli çarpanların boyutu model performansını nasıl etkiler?
* Farklı eniyileyicilerini, öğrenme oranlarını ve ağırlık sönümü oranlarını deneyin.
* Belirli bir film için diğer kullanıcıların tahmini derecelendirme puanlarını kontrol edin.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/400)
:end_tab:
