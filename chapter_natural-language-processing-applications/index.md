# Doğal Dil İşleme: Uygulamalar
:label:`chap_nlp_app`

Metin dizilerinde belirteçleri nasıl temsil edeceğimizi ve :numref:`chap_nlp_pretrain`'te temsillerini nasıl eğitileceğini gördük. Bu tür önceden eğitilmiş metin temsilleri, farklı alt akış doğal dil işleme görevleri için çeşitli modellere beslenebilir. 

Aslında, daha önceki bölümler zaten bazı doğal dil işleme uygulamalarını tartıştık
*ön eğitim almadan*,
Sadece derin öğrenme mimarilerini açıklamak için. Örneğin, :numref:`chap_rnn`'da, roman benzeri metinler üretmek için dil modelleri tasarlamak için RNN'lere güvendik. :numref:`chap_modern_rnn` ve :numref:`chap_attention`'te, makine çevirisi için RNN'lere ve dikkat mekanizmalarına dayanan modeller de tasarladık. 

Ancak, bu kitap tüm bu tür uygulamaları kapsamlı bir şekilde kapsayacak niyetinde değildir. Bunun yerine, odak noktamız doğal dil işleme problemlerini ele almak için dillerin (derin) temsil öğrenimini nasıl uygulayacağımız*. Önceden eğitilmiş metin temsilleri göz önüne alındığında, bu bölüm iki popüler ve temsili alt akış doğal dil işleme görevlerini inceleyecektir: duygu analizi ve doğal dil çıkarımı, sırasıyla tek metin ve metin çiftlerinin ilişkilerini analiz eder. 

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on how to design models for different downstream natural language processing applications.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

:numref:`fig_nlp-map-app`'te tasvir edildiği gibi, bu bölüm, MLP'ler, CNN'ler, RNN'ler ve dikkat gibi farklı derin öğrenme mimarileri türlerini kullanarak doğal dil işleme modellerinin tasarlanmasına ilişkin temel fikirleri açıklamaya odaklanmaktadır. Herhangi bir önceden eğitilmiş metin temsillerini :numref:`fig_nlp-map-app`'te her iki uygulama için herhangi bir mimariyle birleştirmek mümkün olsa da, birkaç temsilci kombinasyon seçiyoruz. Özellikle, duyarlılık analizi için RNN'ler ve CNN'lere dayalı popüler mimarileri araştıracağız. Doğal dil çıkarımı için, metin çiftlerinin nasıl analiz edileceğini göstermek için dikkat ve MLP'leri seçiyoruz. Sonunda, bir dizi düzeyinde (tek metin sınıflandırması ve metin çifti sınıflandırması) ve bir belirteç düzeyinde (metin etiketleme ve soru yanıtlama) gibi çok çeşitli doğal dil işleme uygulamaları için önceden eğitilmiş bir BERT modelinin nasıl ince ayar yapılacağını tanıtıyoruz. Somut ampirik bir durum olarak, BERT doğal dil çıkarımı için ince ayar yapacağız. 

:numref:`sec_bert`'te tanıtılan gibi BERT, çok çeşitli doğal dil işleme uygulamaları için minimal mimari değişiklikleri gerektirir. Bununla birlikte, bu fayda, aşağı akım uygulamaları için çok sayıda BERT parametresinin ince ayarlanması pahasına gelir. Uzay veya zaman sınırlı olduğunda, MLP'ler, CNN'ler, RNN'ler ve dikkat temelli hazırlanmış modeller daha uygulanabilir. Aşağıda, duyarlılık analizi uygulamasıyla başlıyoruz ve sırasıyla RNN ve CNN'lere dayalı model tasarımını gösteriyoruz.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```
