# Işın Arama
:label:`sec_beam-search`

:numref:`sec_seq2seq`'te, özel sıra sonu belirteci "" <eos> "belirteci tahmin edilene kadar çıktı sırası belirtecini belirteç ile tahmin ettik. Bu bölümde, bu *açgözlü arama* stratejisini resmileştirmeye ve onunla ilgili sorunları araştırmaya başlayacağız, daha sonra bu stratejiyi diğer alternatiflerle karşılaştıracağız:
*kapsamlı arama* ve *kiriş arama*.

Açgözlü aramaya resmi bir girişten önce, :numref:`sec_seq2seq`'ten aynı matematiksel gösterimi kullanarak arama problemini resmileştirelim. Herhangi bir zamanda adım $t'$, $y_{t'}$ dekoder çıkış olasılığı $y_1, \ldots, y_{t'-1}$ önce çıkış alt sırası $y_1, \ldots, y_{t'-1}$ ve giriş dizisinin bilgilerini kodlayan bağlam değişkeni $\mathbf{c}$ koşulludur. Hesaplama maliyetini ölçmek için, <eos> çıktı kelime dağarcığını $\mathcal{Y}$ ile belirtin ("“içerir). Yani bu kelime setinin $\left|\mathcal{Y}\right|$ kardinalliği kelime büyüklüğündedir. Ayrıca, bir çıkış dizisinin maksimum belirteç sayısını $T'$ olarak belirtelim. Sonuç olarak, amacımız tüm $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ olası çıkış sekanslarından ideal bir çıktı aramaktır. Tabii ki, tüm bu çıktı dizileri için, <eos> gerçek çıktıda "" dahil olmak üzere ve sonrası bölümler atılacaktır.

## Açgözlü Arama

İlk olarak, basit bir stratejiye bir göz atalım: *açgözlü arama*. Bu strateji :numref:`sec_seq2seq`'teki dizileri tahmin etmek için kullanılmıştır. Açgözlü aramada, herhangi bir zamanda çıktı dizisinin $t'$'sında, $\mathcal{Y}$'ten en yüksek koşullu olasılığa sahip belirteci ararız, yani,

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$

çıktı olarak. “<eos>" çıktılandıktan veya çıkış sırası maksimum uzunluğuna ulaştıktan sonra çıkış sırası tamamlanır.

Peki açgözlü arama ile ne yanlış gidebilir? Aslında, *optimal dizi*, giriş sırasına dayalı bir çıkış dizisi oluşturmanın koşullu olasılığı olan maksimum $\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$ ile çıkış sırası olmalıdır. Ne yazık ki, optimal dizinin açgözlü arama ile elde edileceğinin garantisi yoktur.

![At each time step, greedy search selects the token with the highest conditional probability.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

Bir örnekle gösterelim. <eos>Çıktı sözlüğünde “A”, “B”, “C” ve "" "dört belirteci olduğunu varsayalım. :numref:`fig_s2s-prob1`'te, her zaman adımının altındaki dört sayı, <eos> o zaman adımında sırasıyla “A”, “B”, “C” ve "" "üretme koşullu olasılıklarını temsil eder. Her adımda, açgözlü arama, en yüksek koşullu olasılığa sahip belirteci seçer. Bu nedenle, “A”, “B”, “C” ve "<eos>" çıkış sırası :numref:`fig_s2s-prob1`'te tahmin edilecektir. Bu çıkış dizisinin koşullu olasılığı $0.5\times0.4\times0.4\times0.6 = 0.048$'dir.

![The four numbers under each time step represent the conditional probabilities of generating "A", "B", "C", and "&lt;eos&gt;" at that time step.  At time step 2, the token "C", which has the second highest conditional probability, is selected.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

Sonra, :numref:`fig_s2s-prob2`'teki başka bir örneğe bakalım. :numref:`fig_s2s-prob1`'ün aksine, zaman adımında :numref:`fig_s2s-prob2`'te “C” belirteci seçiyoruz; bu, *saniye* en yüksek koşullu olasılığa sahip. Zaman adım 3 dayandığı zaman adımları 1 ve 2, çıkış sonradan beri :numref:`fig_s2s-prob1` “A” ve “B” :numref:`fig_s2s-prob2` “A” ve “C” :numref:`fig_s2s-prob2`, her belirteç koşullu olasılık zaman adım 3 de değişti :numref:`fig_s2s-prob2`. Zaman adım 3'te “B” belirteci seçtiğimizi varsayalım. Şimdi adım 4, :numref:`fig_s2s-prob1`'te “A”, “C” ve “B” den farklı olan ilk üç zaman adımlarında çıkış alt sırası üzerinde koşulludur. Bu nedenle, :numref:`fig_s2s-prob2`'teki 4. adımda her belirteci üretmenin koşullu olasılığı da :numref:`fig_s2s-prob1`'teki durumdan farklıdır. Sonuç olarak, <eos> :numref:`fig_s2s-prob2`'te “A”, “C”, “B” ve "çıkış dizisinin koşullu olasılığı $0.5\times0.3 \times0.6\times0.6=0.054$'dır, bu da :numref:`fig_s2s-prob1`'teki açgözlü aramadakinden daha büyüktür. Bu örnekte, <eos> açgözlü arama ile elde edilen “A”, “B”, “C” ve “çıkış sırası en uygun sıra değildir.

## Tükenmez Arama

Amaç en uygun diziyi elde etmekse, *kapsamlı arama* kullanmayı düşünebiliriz: olası tüm çıkış sıralarını koşullu olasılıklarıyla kapsamlı bir şekilde numaralandırın, ardından en yüksek koşullu olasılığa sahip olanı çıktısını alın.

Optimum diziyi elde etmek için kapsamlı arama kullanabilsek de, hesaplama maliyeti $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$'in aşırı derecede yüksek olması muhtemeldir. Örneğin, $|\mathcal{Y}|=10000$ ve $T'=10$ olduğunda, $10000^{10} = 10^{40}$ dizilerini değerlendirmemiz gerekecek. Bu imkansıza yakın! Öte yandan, açgözlü aramanın hesaplama maliyeti $\mathcal{O}(\left|\mathcal{Y}\right|T')$'dir: genellikle kapsamlı aramadakinden önemli ölçüde daha küçüktür. Örneğin, $|\mathcal{Y}|=10000$ ve $T'=10$ olduğunda, sadece $10000\times10=10^5$ dizilerini değerlendirmemiz gerekir.

## Işın Arama

Sıra arama stratejileri ile ilgili kararlar, her iki uç da kolay sorularla birlikte bir spektrumda yatar. Ya sadece doğruluk önemliyse? Açıkça görülüyor ki, kapsamlı bir arama. Ya sadece hesaplamalı maliyet önemliyse? Açıkça, açgözlü arama. Gerçek dünya uygulamaları genellikle karmaşık bir soru sorar, bu iki uç arasında bir yerde.

*Işın arama* açgözlü aramanın geliştirilmiş bir versiyonudur. *ışın boyutu*, $k$ adında bir hiperparametresi vardır.
Zaman adımında, en yüksek koşullu olasılıklara sahip $k$ belirteçleri seçiyoruz. Her biri sırasıyla $k$ aday çıkış dizilerinin ilk simgesi olacak. Sonraki her zaman adımında, önceki zaman adımındaki $k$ aday çıkış dizilerine dayanarak, $k\left|\mathcal{Y}\right|$ olası seçeneklerden en yüksek koşullu olasılıklara sahip $k$ aday çıkış dizilerini seçmeye devam ediyoruz.

![The process of beam search (beam size: 2, maximum length of an output sequence: 3). The candidate output sequences are $A$, $C$, $AB$, $CE$, $ABD$, and $CED$.](../img/beam-search.svg)
:label:`fig_beam-search`

:numref:`fig_beam-search`, bir örnek ile ışın arama sürecini gösterir. Çıktı kelime dağarcığının sadece beş öğe içerdiğini varsayalım: $\mathcal{Y} = \{A, B, C, D, E\}$, bunlardan biri “<eos>”. Işın boyutunun 2 olmasına ve bir çıkış dizisinin maksimum uzunluğu 3 olmasına izin verin. Zaman adımında, $P(y_1 \mid \mathbf{c})$ en yüksek koşullu olasılıklara sahip belirteçlerin $A$ ve $C$ olduğunu varsayalım. Zaman adımında 2, tüm $y_2 \in \mathcal{Y},$ için hesaplıyoruz

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

ve bu on değer arasında en büyük ikisini seçin, diyelim ki $P(A, B \mid \mathbf{c})$ ve $P(C, E \mid \mathbf{c})$. Sonra zaman adım 3'te, tüm $y_3 \in \mathcal{Y}$ için hesaplıyoruz

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

ve bu on değerler arasında en büyük ikisini seçin, diyelim ki $P(A, B, D \mid \mathbf{c})$ ve $P(C, E, D \mid  \mathbf{c}).$ Sonuç olarak, altı aday çıkış dizileri elde ediyoruz: (i) $A$; (ii) $C$; (iii) $A$, $B$; (iv) $C$, $E$; (v) $A$, $A$, $A$, $A$, $P(C, E, D \mid  \mathbf{c}).$ 32293614, $D$; ve (vi) $C$, $E$, $D$.

Sonunda, bu altı diziye dayalı nihai aday çıkış dizileri kümesini elde ederiz (örneğin, “<eos>” dahil ve sonrasında bölümleri atın). Ardından, çıkış sırası olarak aşağıdaki puanın en yüksek seviyesine sahip diziyi seçiyoruz:

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$
:eqlabel:`eq_beam-search-score`

burada $L$, son aday dizisinin uzunluğudur ve $\alpha$ genellikle 0.75 olarak ayarlanır. Daha uzun bir dizi :eqref:`eq_beam-search-score` toplamında daha fazla logaritmik terime sahip olduğundan, paydadaki $L^\alpha$ terimi uzun dizileri cezalandırır.

Işın aramasının hesaplama maliyeti $\mathcal{O}(k\left|\mathcal{Y}\right|T')$'tür. Bu sonuç açgözlü arama ile kapsamlı arama arasında yer alır. Aslında, açgözlü arama, 1 ışın boyutuna sahip özel bir ışın araması türü olarak kabul edilebilir. Kiriş boyutu esnek bir seçim ile ışın arama, hesaplama maliyetine karşı doğruluk arasında bir denge sağlar.

## Özet

* Sıra arama stratejileri, açgözlü arama, kapsamlı arama ve ışın aramasını içerir.
* Işın arama, ışın boyutunun esnek seçimi ile hesaplama maliyetine karşı doğruluk arasında bir denge sağlar.

## Egzersizler

1. Ayrıntılı aramayı özel bir ışın araması türü olarak ele alabilir miyiz? Neden ya da neden olmasın?
1. :numref:`sec_seq2seq`'teki makine çeviri probleminde ışın aramasını uygulayın. Işın boyutu çeviri sonuçlarını ve tahmin hızını nasıl etkiler?
1. :numref:`sec_rnn_scratch`'te kullanıcı tarafından sağlanan önekleri takip eden metin oluşturmak için dil modelleme kullandık. Hangi tür bir arama stratejisi kullanıyor? Bunu geliştirebilir misin?

[Discussions](https://discuss.d2l.ai/t/338)
