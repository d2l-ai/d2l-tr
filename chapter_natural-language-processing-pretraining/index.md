# Doğal Dil İşleme: Ön Eğitim
:label:`chap_nlp_pretrain`

İnsanların iletişim kurması gerekiyor. İnsan durumunun bu temel ihtiyacı dışında, günlük olarak çok sayıda yazılı metin oluşturulmuştur. Sosyal medyada zengin metinler, sohbet uygulamaları, e-postalar, ürün incelemeleri, haber makaleleri, araştırma kağıtları ve kitaplar göz önüne alındığında, bilgisayarların yardım sunmalarını veya insan dillerine dayalı kararlar almalarını anlamalarını sağlamak hayati önem taşır. 

*Doğal dil işleme* doğal dilleri kullanarak bilgisayarlar ve insanlar arasındaki etkileşimleri inceler.
Uygulamada, :numref:`sec_language_model`'teki dil modelleri ve :numref:`sec_machine_translation`'teki makine çevirisi modelleri gibi metin (insan doğal dili) verilerini işlemek ve analiz etmek için doğal dil işleme tekniklerinin kullanılması çok yaygındır. 

Metni anlamak için, temsillerini öğrenerek başlayabiliriz. Büyük corpora mevcut metin dizileri yararlanarak,
*kendi kendine denetimli öğrenme*
, metnin gizli bir kısmını çevreleyen metnin başka bir bölümünü kullanarak tahmin etmek gibi metin temsillerini önceden eğitmek için yaygın olarak kullanılmıştır. Bu şekilde modeller, *pahalı* etiketleme çabaları olmadan *kitle* metin verilerinden denetim yoluyla öğrenirler! 

Bu bölümde göreceğimiz gibi, her kelimeyi veya alt kelimeyi bireysel bir belirteç olarak ele alırken, her belirteçin temsili, word2vec, Eldiven veya büyük corpora üzerinde alt kelime gömme modelleri kullanılarak önceden eğitilebilir. Ön eğitim sonrasında, her belirteçin temsili bir vektör olabilir, ancak bağlam ne olursa olsun aynı kalır. Örneğin, “banka” vektör temsili “biraz para yatırmak için bankaya git” ve “oturmak için bankaya git” de aynıdır. Böylece, daha birçok yeni ön eğitim modeli, aynı belirteçin temsilini farklı bağlamlara uyarlar. Bunların arasında, transformatör kodlayıcısına dayanan çok daha derin bir öz denetimli model olan BERT var. Bu bölümde, :numref:`fig_nlp-map-pretrain`'te vurgulandığı gibi, metin için bu tür temsillerin nasıl ön eğitileceğine odaklanacağız. 

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on the upstream text representation pretraining.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`

Büyük resmin görünmesi için :numref:`fig_nlp-map-pretrain`, önceden eğitilmiş metin temsillerinin farklı alt akış doğal dil işleme uygulamaları için çeşitli derin öğrenme mimarilerine beslenebileceğini göstermektedir. Onları :numref:`chap_nlp_app`'te ele alacağız.

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining
```
