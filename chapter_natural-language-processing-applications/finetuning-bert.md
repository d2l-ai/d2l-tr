# Dizi Düzeyi ve Simge Düzeyi Uygulamaları için İnce Ayar BERT
:label:`sec_finetuning-bert`

Bu bölümün önceki bölümlerinde, RNN'ler, CNN'ler, dikkat ve MLP'lere dayanan doğal dil işleme uygulamaları için farklı modeller tasarladık. Bu modeller, alan veya zaman kısıtlaması olduğunda faydalıdır, ancak her doğal dil işleme görevi için belirli bir model oluşturmak pratik olarak imkansızdır. :numref:`sec_bert` yılında, çok çeşitli doğal dil işleme görevleri için minimal mimari değişiklikleri gerektiren bir ön eğitim modeli olan BERT tanıttık. Bir yandan, teklifi sırasında BERT çeşitli doğal dil işleme görevlerinde sanat durumunu geliştirdi. Öte yandan, :numref:`sec_bert-pretraining`'te belirtildiği gibi, orijinal BERT modelinin iki versiyonu 110 milyon ve 340 milyon parametre ile geliyor. Bu nedenle, yeterli hesaplama kaynağı olduğunda, aşağı akış doğal dil işleme uygulamaları için BERT ince ayar düşünebiliriz. 

Aşağıda, doğal dil işleme uygulamalarının bir alt kümesini sıra düzeyi ve belirteç düzeyi olarak genelleştiriyoruz. Sıra düzeyinde, metin girişinin BERT temsilini tek metin sınıflandırmasında ve metin çifti sınıflandırmasında veya regresyonda çıktı etiketine nasıl dönüştüreceğimizi tanıtıyoruz. Simge düzeyinde, metin etiketleme ve soru yanıtlama gibi yeni uygulamaları kısaca tanıtacağız ve BERT'IN girişlerini nasıl temsil edebileceğine ve çıktı etiketlerine dönüştürebileceğine ışık tutacağız. İnce ayar sırasında BERT tarafından farklı uygulamalar için gerekli olan “minimal mimari değişiklikleri” ekstra tam bağlı katmanlardır. Aşağı akış uygulamasının denetimli öğrenimi sırasında ekstra katmanların parametreleri sıfırdan öğrenilirken, önceden eğitilmiş BERT modelindeki tüm parametreler ince ayarlanır. 

## Tek Metin Sınıflandırması

*Tek metin sınıflandırma* giriş olarak tek bir metin dizisi alır ve sınıflandırma sonucunu çıkarır.
Bu bölümde incelediğimiz duyarlılık analizinin yanı sıra, Dilsel Kabul Edillilik Derecesi (CoLA), belirli bir cümlenin dilbilgisi açısından kabul edilebilir olup olmadığına karar veren tek metin sınıflandırması için bir veri kümedir. Mesela, “Çalışmalıyım.” kabul edilebilir ama “Çalışmalıyım” değil. 

![Fine-tuning BERT for single text classification applications, such as sentiment analysis and testing linguistic acceptability. Suppose that the input single text has six tokens.](../img/bert-one-seq.svg)
:label:`fig_bert-one-seq`

:numref:`sec_bert` BERT girdi temsilini açıklar. BERT giriş sırası, hem tek metin hem de metin çiftlerini açıkça temsil eder; burada “” özel sınıflandırma belirteci “<cls>” dizisi sınıflandırması için kullanılır ve “<sep>” özel sınıflandırma belirteci tek metnin sonunu işaretler veya bir metni ayırır. :numref:`fig_bert-one-seq`'te gösterildiği gibi, tek metin sınıflandırma uygulamalarında, “<cls>” özel sınıflandırma belirtecinin BERT temsili, giriş metni dizisinin tüm bilgilerini kodlar. Giriş tek metnin temsili olarak, tüm ayrık etiket değerlerinin dağıtımını yapmak için tam bağlı (yoğun) katmanlardan oluşan küçük bir MLP içine beslenecektir. 

## Metin Çifti Sınıflandırma veya Regresyon

Bu bölümde doğal dil çıkarımı da inceledik. Bir çift metni sınıflandıran bir uygulama türü olan *metin çifti sınıflandırma* aittir. 

Bir çift metni girdi olarak alarak, sürekli bir değer çıktısının alınması,
*anlamsal metin benzerliği* popüler bir *metin çifti regresyon* görevidir.
Bu görev cümlelerin anlamsal benzerliğini ölçer. Örneğin, Anlamsal Metinsel Benzerlik Benchmark veri kümelerinde, bir çift cümlenin benzerlik puanı 0 (anlam çakışmayan) ile 5 (yani eşdeğerlik) :cite:`Cer.Diab.Agirre.ea.2017` arasında değişen bir sıra ölçeğidir. Amaç bu skorları tahmin etmektir. Anlamsal Metin Benzerliği Benchmark veri kümesine örnekler şunlardır (cümle 1, cümle 2, benzerlik puanı): 

* “Bir uçak kalkıyor. “, “Bir uçak kalkıyor. “, 5.000;
* “Bir kadın bir şeyler yiyor. “, “Bir kadın et yiyor. “, 3.000;
* “Bir kadın dans ediyor. “, “Bir adam konuşuyor. “, 0.000.

![Fine-tuning BERT for text pair classification or regression applications, such as natural language inference and semantic textual similarity. Suppose that the input text pair has two and three tokens.](../img/bert-two-seqs.svg)
:label:`fig_bert-two-seqs`

:numref:`fig_bert-one-seq`'teki tek metin sınıflandırmasıyla karşılaştırıldığında, :numref:`fig_bert-two-seqs`'teki metin çifti sınıflandırması için BERT ince ayar girdi temsilinde farklıdır. Anlamsal metinsel benzerlik gibi metin çifti regresyon görevleri için, sürekli bir etiket değeri çıktısı ve ortalama kare kaybının kullanılması gibi önemsiz değişiklikler uygulanabilir: bunlar regresyon için yaygındır. 

## Metin Etiketleme

Şimdi, her belirtecin bir etiket atandığı *text etiketleme* gibi belirteç düzeyinde görevleri ele alalım. Metin etiketleme görevleri arasında
*konuşma parçası etiketleme* her kelimeye bir konuşma parçası etiketi atar (örn., sıfat ve determiner)
cümledeki kelimenin rolüne göre. Örneğin, Penn Treebank II etiket setine göre, “John Smith'in arabası yeni” cümlesi “NNP (isim, uygun tekil) NNP POS (iyelik sonu) NN (isim, tekil veya kütle) VB (fiil, temel form) JJ (sıfat)” olarak etiketlenmelidir. 

![Fine-tuning BERT for text tagging applications, such as part-of-speech tagging. Suppose that the input single text has six tokens.](../img/bert-tagging.svg)
:label:`fig_bert-tagging`

Metin etiketleme uygulamaları için ince ayar BERT :numref:`fig_bert-tagging`'te gösterilmiştir. :numref:`fig_bert-one-seq` ile karşılaştırıldığında, tek ayrım metin etiketlemede yatmaktadır, giriş metninin*her belirteç* BERT temsili, bir konuşma parçası etiketi gibi belirteç etiketinin çıktısı için aynı ekstra tam bağlı katmanlara beslenir. 

## Soru Yanıtlama

Başka bir belirteç düzeyinde uygulama olarak
*soru cevaplama* okuma anlama yeteneklerini yansıtır.
Örneğin, Stanford Soru Yanıtlama Veri Kümesi (SQuAD v1.1), her sorunun cevabı, sorunun :cite:`Rajpurkar.Zhang.Lopyrev.ea.2016` hakkında olduğu pasajdan sadece bir metin (metin yayılma alanı) parçası olduğu okuma pasajlarından ve sorulardan oluşur. Açıklamak için, bir geçit düşünün “Bazı uzmanlar bir maskenin etkinliğinin yetersiz olduğunu bildiriyor. Bununla birlikte, maske üreticileri, N95 solunum maskeleri gibi ürünlerinin virüse karşı korunabileceği konusunda ısrar ediyorlar.” ve bir soru “N95 solunum maskelerinin virüse karşı koruyabileceğini kim söylüyor?” Cevap, geçitteki metin yayılma alanı “maske üreticileri” olmalıdır. Böylece, SQuAD v1.1'deki amaç, bir çift soru ve geçit verilen geçişte metin yayılma başlangıcını ve sonunu tahmin etmektir. 

![Fine-tuning BERT for question answering. Suppose that the input text pair has two and three tokens.](../img/bert-qa.svg)
:label:`fig_bert-qa`

Soru yanıtlama için BERT ince ayar yapmak için, soru ve geçiş BERT girişinde sırasıyla birinci ve ikinci metin dizisi olarak paketlenir. Metin yayılma alanının başlangıcının konumunu tahmin etmek için, aynı ek tam bağlı katman, $i$ konumunun geçişinden herhangi bir belirteçin BERT temsilini $s_i$ skaler bir skaler skora dönüştürecektir. Tüm geçiş belirteçlerinin bu puanları daha da softmax işlemi tarafından bir olasılık dağılımına dönüştürülür, böylece pasajdaki her belirteç konumu $i$ metin yayılma alanının başlangıcı olma olasılığı $p_i$ atanır. Metin yayılma alanının sonunu tahmin etmek yukarıdakiyle aynıdır, ancak ek tamamen bağlı katmanındaki parametrelerin başlangıcı tahmin etmek için olanlardan bağımsız olması dışında. Sonunu tahmin ederken, $i$ pozisyonunun herhangi bir geçiş belirteci, aynı tam bağlı katman tarafından $e_i$ skaler bir skaler skora dönüştürülür. :numref:`fig_bert-qa`, soru yanıtlama için BERT ince ayar tasvir eder. 

Soru yanıtlamak için, denetimli öğrenmenin eğitim hedefi, zemin gerçekliğin başlangıç ve bitiş pozisyonlarının günlük olasılığını en üst düzeye çıkarmak kadar basittir. Yayılma süresini tahmin ederken, $i$ konumundan $j$ ($i \leq j$) konumuna kadar geçerli bir yayılma alanı için $s_i + e_j$ puanını hesaplayabilir ve yayılma alanının en yüksek puanla çıktısını gerçekleştirebiliriz. 

## Özet

* BERT, tek metin sınıflandırması (örn. Duygu analizi ve dilsel kabul edilebilirliği test etme), metin çifti sınıflandırması veya regresyon (örneğin, doğal dil) gibi sıra düzeyi ve belirteç düzeyinde doğal dil işleme uygulamaları için minimal mimari değişiklikleri (ekstra tam bağlantılı katmanlar) gerektirir. çıkarım ve anlamsal metinsel benzerlik), metin etiketleme (örn. konuşma bölümü etiketleme) ve soru yanıtlama.
* Aşağı akış uygulamasının denetimli öğrenimi sırasında ekstra katmanların parametreleri sıfırdan öğrenilirken, önceden eğitilmiş BERT modelindeki tüm parametreler ince ayarlanır.

## Egzersizler

1. Haber makaleleri için bir arama motoru algoritması tasarlayalım. Sistem bir sorgu aldığında (örneğin, “koronavirüs salgını sırasında petrol endüstrisi”), sorgu ile en alakalı haber makalelerinin sıralı listesini döndürmelidir. Büyük bir haber makalesi havuzumuz ve çok sayıda sorgumuz olduğunu varsayalım. Sorunu basitleştirmek için en alakalı makalenin her sorgu için etiketlenmiş olduğunu varsayalım. Algoritma tasarımında negatif örnekleme (bkz. :numref:`subsec_negative-sampling`) ve BERT nasıl uygulayabiliriz?
1. BERT'in eğitim dil modellerinde nasıl kaldıraç yapabiliriz?
1. BERT makine çevirisinde kaldıraç yapabilir miyiz?

[Discussions](https://discuss.d2l.ai/t/396)
