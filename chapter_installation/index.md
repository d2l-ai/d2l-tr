# Kurulum
:label:`chap_installation`

Sizi uygulamalı öğrenme deneyimine hazır hale getirmek için Python'u, Jupyter not defterlerini, ilgili kütüphaneleri ve kitabın kendisini çalıştırmada gerekli kodu koşturan bir ortam kurmanız gerekiyor.

## Miniconda'yı Yükleme

[Miniconda](https://conda.io/en/latest/miniconda.html)'yı yükleyerek işe başlayabilirsiniz.
Python 3.x sürümü gereklidir. Eğer makinenizde conda önceden kurulmuş ise, aşağıdaki adımları
atlayabilirsiniz. 

Miniconda'nın web sitesini ziyaret edin ve Python 3.x sürümünüze ve makine mimarinize göre sisteminiz için uygun sürümü belirleyin. Örneğin, macOS ve Python 3.x kullanıyorsanız, adı "Miniconda3" ve "MacOSX" dizelerini içeren bash betiğini indirin, indirme konumuna gidin ve kurulumu aşağıdaki gibi yürütün:

```bash
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```

Python 3.x'e sahip bir Linux kullanıcısı, adı "Miniconda3" ve "Linux" dizelerini içeren dosyayı indirmeli ve indirme konumunda aşağıda yazılanları yürütmeli:

```bash
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Ardından, doğrudan `conda`'yı çalıştırabilmeniz için kabuğu (shell) sıfırlayın.

```bash
~/miniconda3/bin/conda init
```

Şimdi mevcut kabuğunuzu kapatın ve yeniden açın. Artık aşağıdaki gibi yeni bir
ortam oluşturabilirsiniz:

```bash
conda create --name d2l python=3.9 -y
```


## D2L Not Defterlerini İndirme

Sonra, bu kitapta kullanılan kodu indirmeniz gerekiyor. Kodu indirmek ve açmak için
herhangi bir HTML sayfasının üst kısmındaki "Not Defterleri" sekmesine
tıklayabilirsiniz. Alternatif olarak, "unzip" varsa kullanabilirsiniz
(yoksa "sudo apt install unzip" yazarak kurabilirsiniz):

```bash
mkdir d2l-tr && cd d2l-tr
curl https://tr.d2l.ai/d2l-tr.zip -o d2l-tr.zip
unzip d2l-tr.zip && rm d2l-tr.zip
```

Şimdi `d2l` ortamını etkinleştirebiliriz:

```bash
conda activate d2l

```


## Çerçeveyi ve `d2l` Paketini Yükleme

Herhangi bir derin öğrenme çerçevesini kurmadan önce, lütfen önce makinenizde uygun GPU'ların olup olmadığını kontrol edin (standart bir dizüstü bilgisayarda ekranı destekleyen GPU'lar bizim amacımıza uygun sayılmaz). GPU'lu bir sunucuda çalışıyorsanız, ilgili kütüphanelerin GPU-dostu sürümlerinin kurulum talimatları için şu adrese ilerleyin :ref:`subsec_gpu`.

Makinenizde herhangi bir GPU yoksa, henüz endişelenmenize gerek yok. CPU'nuz, ilk birkaç bölümü tamamlamanız için fazlasıyla yeterli beygir gücü sağlar. Daha büyük modelleri koşmadan önce GPU'lara erişmek isteyeceğinizi unutmayın. CPU sürümünü kurmak için aşağıdaki komutu yürütün.

:begin_tab:`mxnet`

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:


:begin_tab:`pytorch`

```bash
pip install torch torchvision
```


:end_tab:

:begin_tab:`tensorflow`
TensorFlow'u hem CPU hem de GPU desteğiyle aşağıdaki gibi yükleyebilirsiniz:

```bash
pip install tensorflow tensorflow-probability
```


:end_tab:


Bir sonraki adımımız, bu kitapta bulunan sık kullanılan işlevleri ve sınıfları kapsamak için geliştirdiğimiz `d2l` paketini kurmaktır.

```bash
pip install d2l==0.17.5
```

Bu kurulum adımlarını tamamladıktan sonra, Jupyter not defteri sunucusunu şu şekilde çalıştırarak başlatabiliriz:

```bash
jupyter notebook
```

Bu noktada, http://localhost:8888 (otomatik olarak açılmış olabilir) adresini
web tarayıcınızda açabilirsiniz. Sonra kitabın her bölümü için kodu çalıştırabilirsiniz.
Lütfen kitabın kodunu çalıştırmadan veya derin öğrenme çerçevesini veya `d2l`
paketini güncellemeden önce çalışma zamanı ortamını etkinleştirmek için daima
`conda enable d2l` komutunu çalıştırın. Ortamdan çıkmak için `conda deactivate` komutunu
çalıştırın.


## GPU Desteği
:label:`subsec_gpu`

:begin_tab:`mxnet`
Varsayılan olarak, MXNet, herhangi bir bilgisayarda (çoğu dizüstü bilgisayar dahil) çalışmasını sağlamak için GPU desteği olmadan kurulur. Bu kitabın bir kısmı GPU ile çalışmayı gerektiriyor veya öneriyor. Bilgisayarınızda NVIDIA grafik kartı varsa ve [CUDA](https://developer.nvidia.com/cuda-downloads) yüklüyse, GPU etkin bir sürümü yüklemelisiniz. Yalnızca CPU sürümünü yüklediyseniz, önce şunu çalıştırarak kaldırmanız gerekebilir:

```bash
pip uninstall mxnet
```


Şimdi hangi CUDA sürümünü yüklediğinizi bulmamız gerekiyor. 
Bunu `nvcc --version` veya `cat /usr/local/cuda/version.txt` komutunu çalıştırarak kontrol edebilirsiniz.
CUDA 10.2'i yüklediğinizi varsayalım, o zaman aşağıdaki komutla kurabilirsiniz:

```bash
# Windows kullanıcıları için
pip install mxnet-cu102==1.7.0 -f https://dist.mxnet.io/python

# Linux ve macOS kullanıcıları için
pip install mxnet-cu102==1.7.0
```

Son rakamları CUDA sürümünüze göre değiştirebilirsiniz. Örn. CUDA 10.1 için `cu101` CUDA 9.0 için `cu90`.
:end_tab:


:begin_tab:`pytorch,tensorflow`
Derin öğrenme çerçevesi, aksi belirtilmediyse GPU desteğiyle kurulacaktır. 
Eğer bilgisayarınızda NVIDIA GPU'su varsa ve [CUDA](https://developer.nvidia.com/cuda-downloads) kuruluysa, artık hazırsınız.
:end_tab:

## Alıştırmalar

1. Kitabın kodunu indirin ve çalışma zamanı ortamını yükleyin.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/436)
:end_tab:
