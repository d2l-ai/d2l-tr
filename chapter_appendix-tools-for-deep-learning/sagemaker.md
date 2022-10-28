# Amazon SageMaker'ı Kullanma
:label:`sec_sagemaker`

Birçok derin öğrenme uygulaması önemli miktarda hesaplama gerektirir. Yerel makineniz bu sorunları makul bir süre içinde çözmek için çok yavaş olabilir. Bulut bilgi işlem hizmetleri, bu kitabın GPU yoğun bölümlerini çalıştırmak için daha güçlü bilgisayarlara erişmenizi sağlar. Bu eğitim Amazon SageMaker aracılığıyla size rehberlik edecektir: Bu kitabı kolayca çalıştırmanızı sağlayan bir hizmet. 

## Kaydolma ve Oturum Açma

Öncelikle, https://aws.amazon.com/ adresinde bir hesap kaydetmemiz gerekiyor. Ek güvenlik için iki etkenli kimlik doğrulaması kullanmanızı öneririz. Çalışan herhangi bir örneği durdurmayı unutmanız durumunda beklenmedik sürprizlerden kaçınmak için ayrıntılı faturalandırma ve harcama uyarıları ayarlamak da iyi bir fikirdir. Kredi kartına ihtiyacınız olacağını unutmayın. AWS hesabınıza giriş yaptıktan sonra [konsol - console](http://console.aws.amazon.com/)'unuza gidin ve "SageMaker" öğesini arayın (bkz. :numref:`fig_sagemaker`) ardından SageMaker panelini açmak için tıklayın. 

![SageMaker panelini açın.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## SageMaker Örneği Oluşturma

Ardından, :numref:`fig_sagemaker-create` içinde açıklandığı gibi bir not defteri örneği oluşturalım. 

![SageMaker örneği oluştur](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

SageMaker çoklu [örnek türleri - instance types](https://aws.amazon.com/sagemaker/pricing/instance-types/) farklı hesaplama gücü ve fiyatları sağlar. Bir örnek oluştururken, örnek adını belirtebilir ve türünü seçebiliriz. :numref:`fig_sagemaker-create-2` içinde `ml.p3.2xlarge`'i seçiyoruz. Bir Tesla V100 GPU ve 8 çekirdekli CPU ile, bu örnek çoğu bölüm için yeterince güçlüdür. 

![Örnek türü seç.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
SageMaker'a uyan bu kitabın Jupyter not defteri versiyonu https://github.com/d2l-ai/d2l-en-sagemaker adresinde mevcuttur. Bu GitHub deposu URL'sini SageMaker'ın örnek oluşturması esnasında :numref:`fig_sagemaker-create-3` içinde gösterildiği gibi klonlamasına izin vermek için belirtebiliriz.
:end_tab:

:begin_tab:`pytorch`
SageMaker'a uyan bu kitabın Jupyter not defteri versiyonu https://github.com/d2l-ai/d2l-en-sagemaker adresinde mevcuttur. Bu GitHub deposu URL'sini SageMaker'ın örnek oluşturması esnasında :numref:`fig_sagemaker-create-3` içinde gösterildiği gibi klonlamasına izin vermek için belirtebiliriz.
:end_tab:

:begin_tab:`tensorflow`
SageMaker'a uyan bu kitabın Jupyter not defteri versiyonu https://github.com/d2l-ai/d2l-en-sagemaker adresinde mevcuttur. Bu GitHub deposu URL'sini SageMaker'ın örnek oluşturması esnasında :numref:`fig_sagemaker-create-3` içinde gösterildiği gibi klonlamasına izin vermek için belirtebiliriz.
:end_tab:

![GitHub deposunu belirtme.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## Bir Örneği Çalıştırma ve Durdurma

Örnek hazır olması birkaç dakika sürebilir. Hazır olduğunda, :numref:`fig_sagemaker-open`' içinde gösterildiği gibi "Jupyter'ı Aç - Open Jupyter" bağlantısını tıklayabilirsiniz. 

![Oluşturulan SageMaker örneğinde Jupyter Aç.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`

Daha sonra, :numref:`fig_sagemaker-jupyter` içinde gösterildiği gibi, bu örnekte çalışan Jupyter sunucusunda gezinebilirsiniz. 

![SageMaker örneğinde çalışan Jupyter sunucusu.](../img/sagemaker-jupyter.png)
:width:`400px`
:label:`fig_sagemaker-jupyter`

SageMaker örneğinde Jupyter not defterlerinin çalıştırılması ve düzenlenmesi, :numref:`sec_jupyter` içindeki tartıştığımıza benzer. Çalışmanızı bitirdikten sonra, :numref:`fig_sagemaker-stop` içinde gösterildiği gibi daha fazla ödemeyi önlemek için örneği durdurmayı unutmayın. 

![Bir SageMaker örneğini durdurma.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## Not Defterlerini Güncelleme

:begin_tab:`mxnet`
Not defterlerini [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker) GitHub deposunda düzenli olarak güncelleyeceğiz. En son sürüme güncellemek için `git pull` komutunu kullanabilirsiniz.
:end_tab:

:begin_tab:`pytorch`
Not defterlerini [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker) GitHub deposunda düzenli olarak güncelleyeceğiz. En son sürüme güncellemek için `git pull` komutunu kullanabilirsiniz.
:end_tab:

:begin_tab:`tensorflow`
Not defterlerini [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker) GitHub deposunda düzenli olarak güncelleyeceğiz. En son sürüme güncellemek için `git pull` komutunu kullanabilirsiniz.
:end_tab:

Öncelikle, :numref:`fig_sagemaker-terminal` içinde gösterildiği gibi bir terminal açmanız gerekir. 

![SageMaker örneğinde bir terminal açma](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

Güncelleştirmeleri çekmeden önce yerel değişikliklerinizi tamamlamak isteyebilirsiniz. Alternatif olarak, terminaldeki aşağıdaki komutlarla tüm yerel değişikliklerinizi görmezden gelebilirsiniz.

:begin_tab:`mxnet`
```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`pytorch`
```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`tensorflow`
```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```
:end_tab:

## Özet

* Bu kitabı çalıştırmak için Amazon SageMaker aracılığıyla bir Jupyter sunucusunu başlatabilir ve durdurabiliriz.
* Not defterlerini Amazon SageMaker örneğindeki terminal üzerinden güncelleyebiliriz.

## Alıştırmalar

1. Amazon SageMaker kullanarak bu kitaptaki kodu düzenlemeyi ve çalıştırmayı deneyin.
1. Terminal üzerinden kaynak kod dizinine erişin.

[Tartışmalar](https://discuss.d2l.ai/t/422)
