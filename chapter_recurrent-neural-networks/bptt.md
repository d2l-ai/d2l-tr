# Zamanda Geri Yayma
:label:`sec_bptt`

Şimdiye kadar defalarca *patlayan gradyanlar*, *kaybolan gradyanlar*, ve RNN'ler için *gradyan ayırma* ihtiyacı gibi şeyler ima ettik. Örneğin, :numref:`sec_rnn_scratch` içinde dizi üzerinde `detach` işlevini çağırdık. Hızlı bir model inşa edebilmek ve nasıl çalıştığını görmek amacıyla, bunların hiçbiri gerçekten tam olarak açıklanmadı. Bu bölümde, dizi modelleri için geri yaymanın ayrıntılarını ve matematiğin neden (ve nasıl) çalıştığını biraz daha derinlemesine inceleyeceğiz.

RNN'leri ilk uyguladığımızda gradyan patlamasının bazı etkileriyle karşılaştık (:numref:`sec_rnn_scratch`). Özellikle, alıştırmaları çözdüyseniz, doğru yakınsamayı sağlamak için gradyan kırpmanın hayati önem taşıdığını görürsünüz. Bu sorunun daha iyi anlaşılmasını sağlamak için, bu bölümde gradyanların dizi modellerinde nasıl hesaplandığını incelenecektir. Nasıl çalıştığına dair kavramsal olarak yeni bir şey olmadığını unutmayın. Sonuçta, biz hala sadece gradyanları hesaplamak için zincir kuralını uyguluyoruz. Bununla birlikte, geri yayma (:numref:`sec_backprop`) tekrar gözden geçirmeye değerdir.

:numref:`sec_backprop` içinde MLP'lerde ileri ve geri yaymayı ve hesaplama çizgelerini tanımladık. Bir RNN'de ileriye yayma nispeten basittir. *Zamanda geri yayma* aslında RNN'lerde geri yaymanın belirli bir uygulamasıdır :cite:`Werbos.1990`. Model değişkenleri ve parametreleri arasındaki bağımlılıkları elde etmek için bir RNN'nin hesaplama çizgesini bir kerede bir adım genişletmemizi gerektirir. Ardından, zincir kuralına bağlı olarak, gradyanları hesaplamak ve depolamak için geri yayma uygularız. Diziler oldukça uzun olabileceğinden, bağımlılık oldukça uzun olabilir. Örneğin, 1000 karakterlik bir dizi için, ilk andıç nihai konumdaki andıç üzerinde potansiyel olarak önemli bir etkiye sahip olabilir. Bu gerçekten hesaplamalı olarak mümkün değildir (çok uzun sürer ve çok fazla bellek gerektirir) ve bizim bu çok zor gradyana ulaşmadan önce 1000'den fazla matrisi çarpamıza gerek duyar. Bu, hesaplamalı ve istatistiksel belirsizliklerle dolu bir süreçtir. Aşağıda neler olduğunu ve bunu pratikte nasıl ele alacağımızı aydınlatacağız.

## RNN'lerde Gradyanların Çözümlemesi
:label:`subsec_bptt_analysis`

Bir RNN'nin nasıl çalıştığını anlatan basitleştirilmiş bir modelle başlıyoruz. Bu model, gizli durumun özellikleri ve nasıl güncellendiği hakkındaki ayrıntıları görmezden gelir. Buradaki matematiksel gösterim, skalerleri, vektörleri ve matrisleri eskiden olduğu gibi açıkça ayırt etmez. Bu ayrıntılar, analiz için önemsizdir ve öbür türlü yalnızca bu alt bölümdeki gösterimi karıştırmaya hizmet edecekti.

Bu basitleştirilmiş modelde, $h_t$'yi gizli durum, $x_t$'yi girdi ve $o_t$'yi $t$'deki çıktı olarak gösteriyoruz. :numref:`subsec_rnn_w_hidden_states` içindeki tartışmalarımızı hatırlayın, girdi ve gizli durum, gizli katmandaki bir ağırlık değişkeni ile çarpılacak şekilde bitiştirilebilir. Böylece, sırasıyla gizli katmanın ve çıktı katmanının ağırlıklarını belirtmek için $w_h$ ve $w_o$'yi kullanırız. Sonuç olarak, her zaman adımındaki gizli durumlar ve çıktılar aşağıdaki gibi açıklanabilir:

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

burada $f$ ve $g$, sırasıyla gizli katmanının ve çıktı katmanının dönüşümleridir. Bu nedenle, yinelemeli hesaplama yoluyla birbirine bağlı $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$ değerler zincirine sahibiz. İleri yayma oldukça basittir. İhtiyacımız olan tek şey $(x_t, h_t, o_t)$ üçlüleri arasında bir seferde bir zaman adımı atarak döngü yapmaktır. Çıktı $o_t$ ve istenen etiket $y_t$ arasındaki tutarsızlık daha sonra tüm $T$ zaman adımlarında amaç işlevi tarafından değerlendirilir

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$

Geri yayma için, özellikle $L$ amaç fonksiyonun $w_h$ parametreleri ile ilgili olarak gradyanları hesaplarken işler biraz daha zorlaşır. Belirleyici olmak gerekirse, zincir kuralına göre,

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

Çarpımın :eqref:`eq_bptt_partial_L_wh` içindeki birinci ve ikinci faktörlerinin hesaplanması kolaydır. $h_t$'da $w_h$ parametresinin etkisini yeniden hesaplamamız gerektiğinden, üçüncü faktör $\partial h_t/\partial w_h$'de işler zorlaşır. :eqref:`eq_bptt_ht_ot` içindeki yinelemeli hesaplamaya göre, $h_t$ $h_{t-1}$ ve $w_h$'ye bağlıdır, burada $h_{t-1}$'in hesaplanması da $w_h$'ye bağlıdır. Böylece, zincir kuralı aşağıdaki çıkarsamaya varır:

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`

Yukarıdaki gradyanı türetmek için, $t=1, 2,\ldots$ için $a_{0}=0$ ve $a_{t}=b_{t}+c_{t}a_{t-1}$ koşullarını sağlayan $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ üç dizisine sahip olduğumuzu varsayalım. Sonra $t\geq 1$ için, aşağıdaki ifadeyi göstermek kolaydır:

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

$a_t$, $b_t$ ve $c_t$'nin yerlerine aşağıdaki ifadeleri koyarsak

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

:eqref:`eq_bptt_partial_ht_wh_recur` içindeki gradyan hesaplama $a_{t}=b_{t}+c_{t}a_{t-1}$'yı sağlar. Böylece, :eqref:`eq_bptt_at` içindeki, :eqref:`eq_bptt_partial_ht_wh_recur` yinelemeli hesaplamasını kaldırabiliriz.

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

$\partial h_t/\partial w_h$'i yinelemeli olarak hesaplamak için zincir kuralını kullanabilsek de, $t$ büyük olduğunda bu zincir çok uzun sürebilir. Bu sorunla başa çıkmak için bazı stratejileri tartışalım.

### Tam Hesaplama ###

Açıkçası, :eqref:`eq_bptt_partial_ht_wh_gen` içindeki tam toplamı hesaplayabiliriz. Fakat, bu çok yavaştır ve gradyanlar patlayabilir, çünkü ilkleme koşullarındaki narin değişiklikler sonucu potansiyel olarak çok etkileyebilir. Yani, ilk koşullardaki minimum değişikliklerin sonuçta orantısız değişikliklere yol açtığı kelebek etkisine benzer şeyler görebiliriz. Bu aslında tahmin etmek istediğimiz model açısından oldukça istenmeyen bir durumdur. Sonuçta, iyi genelleyen gürbüz tahminciler arıyoruz. Bu nedenle bu strateji pratikte neredeyse hiç kullanılmaz.

### Zaman Adımlarını Kesme ###

Alternatif olarak, $\tau$ adımdan sonra :eqref:`eq_bptt_partial_ht_wh_gen` içindeki toplamı kesebiliriz. Bu aslında şimdiye kadar tartıştığımız şey, örneğin :numref:`sec_rnn_scratch` içindeki gradyanları ayırdığımız zaman gibi. Bu, toplamı $\partial h_{t-\tau}/\partial w_h$'de sonlandırarak, gerçek gradyanın *yaklaşık değerine* götürür. Pratikte bu oldukça iyi çalışır. Genellikle zaman boyunca kesilmiş geri yayma olarak adlandırılır :cite:`Jaeger.2002`. Bunun sonuçlarından biri, modelin uzun vadeli sonuçlardan ziyade kısa vadeli etkilere odaklanmasıdır. Bu aslında *arzu edilendir*, çünkü tahminleri daha basit ve daha kararlı modellere yöneltir.

### Rastgele Kesme ###

Son olarak, $\partial h_t/\partial w_h$'yi, beklentiye göre doğru olan ancak diziyi kesen rastgele bir değişkenle değiştirebiliriz. Bu, önceden tanımlanmış $0 \leq \pi_t \leq 1$ olan bir $\xi_t$ dizisi kullanılarak elde edilir, burada $P(\xi_t = 0) = 1-\pi_t$ ve $P(\xi_t = \pi_t^{-1}) = \pi_t$, dolayısıyla $E[\xi_t] = 1$'dir. Bunu :eqref:`eq_bptt_partial_ht_wh_recur` içindeki $\partial h_t/\partial w_h$'i değiştirmek için kullanırız.

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

Bu $\xi_t$'nin tanımından gelir; $E[z_t] = \partial h_t/\partial w_h$. Ne zaman $\xi_t = 0$ olursa, yinelemeli hesaplama o $t$ zaman adımında sona erer. Bu, uzun dizilerin nadir ancak uygun şekilde fazla ağırlıklandırılmış olduğu çeşitli uzunluklardaki dizilerin ağırlıklı bir toplamına yol açar. Bu fikir Tallec ve Ollivier :cite:`Tallec.Ollivier.2017` tarafından önerilmiştir.

### Stratejilerin Karşılaştırılması

![RNN'lerde gradyanları hesaplama stratejilerinin karşılaştırılması. Yukarıdan aşağıya: Rastgele kesme, düzenli kesme ve tam hesaplama.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

:numref:`fig_truncated_bptt`, RNN'ler için zamanda geri yayma kullanan *Zaman Makinesi* kitabının üç stratejideki ilk birkaç karakterini analiz ederek göstermektedir:

* İlk satır, metni farklı uzunluklardaki bölümlere ayıran rasgele kesmedir.
* İkinci satır, metni aynı uzunlukta altdizilere kıran düzenli kesimdir. Bu RNN deneylerinde yaptığımız şeydir.
* Üçüncü satır, hesaplaması mümkün olmayan bir ifadeye yol açan zamanda tam geri yaymadır.

Ne yazık ki, teoride çekici iken, rasgele kesme, büyük olasılıkla bir dizi faktöre bağlı olarak düzenli kesmeden çok daha iyi çalışmaz. Birincisi, bir gözlemin geçmişe birkaç geri yayma adımından sonraki etkisi, pratikteki bağımlılıkları yakalamak için oldukça yeterlidir. İkincisi, artan varyans, gradyanın daha fazla adımla daha doğru olduğu gerçeğine karşı yarışır. Üçüncüsü, aslında sadece kısa bir etkileşim aralığına sahip modeller istiyoruz. Bu nedenle, zamanda düzenli kesilmiş geri yayma, arzu edilebilecek hafif bir düzenlileştirici etkiye sahiptir.

## Ayrıntılı Zamanda Geri Yayma

Genel prensibi tartıştıktan sonra, zamanda geriye yaymayı ayrıntılı olarak ele alalım. :numref:`subsec_bptt_analysis` içindeki analizden farklı olarak, aşağıda, amaç fonksiyonun gradyanlarının tüm ayrıştırılmış model parametrelerine göre nasıl hesaplanacağını göstereceğiz. İşleri basit tutmak için, gizli katmandaki etkinleştirme işlevi olarak birim eşlemelerini kullanan ek girdi parametresiz bir RNN'yi göz önünde bulunduruyoruz ($\phi(x)=x$). Zaman adımı $t$ için, tek örnek girdinin ve etiketin sırasıyla $\mathbf{x}_t \in \mathbb{R}^d$ ve $y_t$ olduğunu varsayalım. Gizli durum $\mathbf{h}_t \in \mathbb{R}^h$ ve çıktı $\mathbf{o}_t \in \mathbb{R}^q$ aşağıdaki gibi hesaplanır:

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

burada $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ ve $\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$ ağırlık parametreleridir. $l(\mathbf{o}_t, y_t)$ ile belirtilen $t$ zaman adımındaki kayıp olsun. Bizim amaç fonksiyonumuz, yani dizinin başından $T$ zaman adımı üzerinden kayıp şöyle hesaplanır:  

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

RNN'nin hesaplanması sırasında model değişkenleri ve parametreleri arasındaki bağımlılıkları görselleştirmek için, model için :numref:`fig_rnn_bptt` içinde gösterildiği gibi bir hesaplama çizgesi çizebiliriz. Örneğin,  3. zaman adımındaki, $\mathbf{h}_3$ gizli durumlarının hesaplanması model parametreleri $\mathbf{W}_{hx}$ ve $\mathbf{W}_{hh}$'ye, son zaman adımındaki gizli durum $\mathbf{h}_2$'ye ve şimdiki zaman adımının girdisi $\mathbf{x}_3$'e bağlıdır.

![Üç zaman adımlı bir RNN modeli için bağımlılıkları gösteren hesaplamalı çizge. Kutular değişkenleri (gölgeli olmayan) veya parametreleri (gölgeli) ve daireler işlemleri temsil eder.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

Az önce belirtildiği gibi, :numref:`fig_rnn_bptt` içindeki model parametreleri $\mathbf{W}_{hx}$, $\mathbf{W}_{hh}$ ve $\mathbf{W}_{qh}$'dır. Genel olarak, bu modelin eğitimi $\partial L/\partial \mathbf{W}_{hx}$, $\partial L/\partial \mathbf{W}_{hh}$ ve $\partial L/\partial \mathbf{W}_{qh}$ parametrelerine göre gradyan hesaplama gerektirir. :numref:`fig_rnn_bptt` içindeki bağımlılıklara göre, sırayla gradyanları hesaplamak ve depolamak için okların ters yönünde ilerleyebiliriz. Zincir kuralında farklı şekillerdeki matrislerin, vektörlerin ve skalerlerin çarpımını esnek bir şekilde ifade etmek için :numref:`sec_backprop` içinde açıklandığı gibi $\text{prod}$ işlemini kullanmaya devam ediyoruz.

Her şeyden önce, amaç işlevinin türevini herhangi bir $t$ zaman adımındaki model çıktısına göre almak oldukça basittir:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

Şimdi, amaç fonksiyonun gradyanını çıktı katmanındaki $\mathbf{W}_{qh}$ parametresine göre hesaplayabiliriz: $\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$. :numref:`fig_rnn_bptt` temel alınarak amaç fonksiyonu $L$, $\mathbf{o}_1, \ldots, \mathbf{o}_T$ üzerinden $\mathbf{W}_{qh}$'e bağlıdır. Zincir kuralını kullanırsak şu sonuca ulaşırız,

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

burada $\partial L/\partial \mathbf{o}_t$ :eqref:`eq_bptt_partial_L_ot` içindeki gibi hesaplanır.

Daha sonra, :numref:`fig_rnn_bptt` içinde gösterildiği gibi, $T$ son zaman adımındaki amaç işlevi $L$ gizli durum $\mathbf{h}_T$'ye yalnızca $\mathbf{o}_T$ üzerinden bağlıdır. Bu nedenle, zincir kuralını kullanarak $\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$'i kolayca bulabiliriz:

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

$t < T$ adımı için daha karmaşık hale gelir, burada $L$ amaç işlevi $\mathbf{h}_t$'ye $\mathbf{h}_{t+1}$ ve $\mathbf{o}_t$ üzerinden bağlıdır. Zincir kuralına göre, herhangi bir $t < T$ zamanında, gizli durumunun gradyanı, $\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$, yinelemeli hesaplanabilir:

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

Analiz için, herhangi bir zaman adım $1 \leq t \leq T$ için yinelemeli hesaplamayı genişletirsek, şu ifadeye ulaşırız:

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

:eqref:`eq_bptt_partial_L_ht` denkleminden bu basit doğrusal örneğin uzun dizi modellerinin bazı temel problemlerini zaten sergilediğini görebiliyoruz: $\mathbf{W}_{hh}^\top$'nın potansiyel olarak çok büyük kuvvetlerini içerir. İçinde, 1'den küçük özdeğerler kaybolur ve 1'den büyük özdeğerler ıraksar. Bu sayısal olarak kararsızdır, bu da kendini kaybolan ve patlayan gradyanlar şeklinde gösterir. Bunu ele almanın bir yolu, :numref:`subsec_bptt_analysis` içinde tartışıldığı gibi, zaman adımlarını hesaplama açısından uygun bir boyutta kesmektir. Pratikte, bu kesme, belirli bir sayıda zaman adımından sonra gradyanı koparıp ayırarak gerçekleştirilir. Daha sonra uzun ömürlü kısa-dönem belleği gibi daha gelişmiş dizi modellerinin bunu daha da hafifletebileceğini göreceğiz.

Son olarak :numref:`fig_rnn_bptt`, $L$ amaç fonksiyonunun gizli katmandaki $\mathbf{W}_{hx}$ ve $\mathbf{W}_{hh}$ model parametrelerine $\mathbf{h}_1, \ldots, \mathbf{h}_T$ vasıtasıyla bağlı olduğunu gösterir. Bu tür parametrelerin $\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ ve $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$'ye göre gradyanları hesaplamak için, zincir kuralını uygularız:

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

burada :eqref:`eq_bptt_partial_L_hT_final_step` ve :eqref:`eq_bptt_partial_L_ht_recur` ile yinelemeli hesaplanan $\partial L/\partial \mathbf{h}_t$ sayısal kararlılığı etkileyen anahtar değerdir.

Zamanda geri yayma, RNN'lerde geri yayma uygulanması olduğundan, :numref:`sec_backprop` içinde açıkladığımız gibi, RNN'leri eğitmek zamanda geri yayma ile ileriye doğru yaymayı değiştirir. Dahası, zamanda geri yayma yukarıdaki gradyanları hesaplar ve sırayla depolar. Özellikle, depolanan ara değerler yinelemeli hesaplamaları önlemek için yeniden kullanılır, örneğin $\partial L / \partial \mathbf{W}_{hx}$ ve $\partial L / \partial \mathbf{W}_{hh}$ hesaplamalarında kullanılacak $\partial L/\partial \mathbf{h}_t$'yi depolamak gibi.

## Özet

* Zamanda geriye yayma, sadece geri yaymanın gizli bir duruma sahip dizi modellerindeki bir uygulamasıdır.
* Hesaplama kolaylığı ve sayısal kararlılık için kesme gereklidir, örneğin düzenli kesme ve rasgele kesme gibi.
* Matrislerin yüksek kuvvetleri ıraksayan veya kaybolan özdeğerlere yol açabilir. Bu, patlayan veya kaybolan gradyanlar şeklinde kendini gösterir.
* Verimli hesaplama için ara değerler zamanda geri yayma sırasında önbelleğe alınır.

## Alıştırmalar

1. $\lambda_i$ özdeğerleri $\mathbf{v}_i$ ($i = 1, \ldots, n$) özvektörlerine karşılık gelen bir simetrik matrisimiz $\mathbf{M} \in \mathbb{R}^{n \times n}$ olduğunu varsayalım. Genelleme kaybı olmadan, $|\lambda_i| \geq |\lambda_{i+1}|$ diye sıralanmış olduğunu varsayalım.
   1. $\lambda_i^k$'nin $\mathbf{M}^k$'nin özdeğerleri olduğunu gösterin.
   1. Yüksek olasılıkla $\mathbf{x} \in \mathbb{R}^n$ rastgele vektörü için $\mathbf{M}^k \mathbf{x}$ özvektörünün $\mathbf{M}$'deki $\mathbf{v}_1$ özvektörüyle hizalanmış olacağını kanıtlayın. Bu ifadeyi formüle dökün.
   1. Yukarıdaki sonuç RNN'lerdeki gradyanlar için ne anlama geliyor?
1. Gradyan kırpmanın yanı sıra, yinelemeli sinir ağlarında gradyan patlaması ile başa çıkmak için başka yöntemler düşünebiliyor musunuz?

[Tartışmalar](https://discuss.d2l.ai/t/334)
