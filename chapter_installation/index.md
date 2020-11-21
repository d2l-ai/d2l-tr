# Kurulum
:label:`chap_installation`

Uygulayarak öğrenmeye başlamadan önce Python'u, Jupyter not defterlerini, 
ilgili kütüphaneleri kurmalı ve kitapta kullanılacak kodu indirmelisiniz.

## Miniconda'yı Yükleme

[Miniconda](https://conda.io/en/latest/miniconda.html)'yı yükleyerek işe başlayabilirsini.
Python 3.x sürümü gereklidir. Conda önceden yüklenmişse, aşağıdaki adımları
atlayabilirsiniz. Sisteminize karşılık gelen Miniconda sh dosyasını web
sitesinden indirin ve ardından komut satırından
`sh <FILENAME> -b` komutunu kullanarak yükleme işlemini başlatın. MacOS kullanıcıları için:

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```


Linux kullanıcıları için:

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Ardından, doğrudan `conda` çalıştırabilmeniz için kabuğu(shell) sıfırlayın.

```bash
~/miniconda3/bin/conda init
```

Şimdi mevcut kabuğunuzu kapatıp yeniden açın. Artık aşağıdaki gibi yeni bir
ortam oluşturabilirsiniz:

```bash
conda create --name d2l python=3.8 -y
```


## D2L Not Defterlerini İndirme

Sonra, bu kitapta kullanılan kodu indirmeniz gerekiyor. Kodu indirmek ve açmak için
herhangi bir HTML sayfasının üst kısmındaki "Tüm Not Defterleri" sekmesine
tıklayabilirsiniz. Alternatif olarak, "unzip" varsa
(yoksa "sudo apt install unzip" yazarak kurabilirsiniz):

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Şimdi `d2l` ortamını etkinleştireceğizç

```bash
conda activate d2l

```


## Çerçeveyi ve `d2l` Paketini Yükleme


Derin öğrenme çerçevesini kurmadan önce, lütfen önce makinenizde uygun GPU'ların
olup olmadığını kontrol edin (standart bir dizüstü bilgisayarda ekranı
destekleyen GPU'lar bizim amacımıza uygun sayılmaz). GPU'lu bir bilgisayara/sunucuya kurulum
yapacaksanız, GPU destekli sürümün kurulum talimatları için şu adrese ilerleyin
:ref:`subsec_gpu`.

GPU'lu bir bilgisayarınız/sunucunuz yoksa, CPU sürümünü aşağıdaki gibi yükleyebilirsiniz.
Bu, ilk birkaç bölümü geçmeniz için yeterli beygir gücünü sağlayacak ancak daha
büyük modelleri çalıştırmak için GPU'lara ihtiyacınız olacak.


:begin_tab:`mxnet`

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:

:begin_tab:`pytorch`

```bash
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```


:end_tab:

:begin_tab:`tensorflow`
TensorFlow'u hem CPU hem de GPU desteğiyle aşağıdakileri kullanarak
yükleyebilirsiniz:

```bash
pip install tensorflow tensorflow-probability
```


:end_tab:

Ayrıca, bu kitapta sık kullanılan işlevleri ve sınıfları içeren `d2l` paketini
de yüklüyoruz.

```bash
# -U: Bütün paketleri en yeni versiyonlarına güncelle
pip install -U d2l
```

Yüklendikten sonra, Jupyter not defterini şu şekilde çalıştırarak açıyoruz:

```bash
jupyter notebook
```

Bu noktada, http://localhost:8888 (genellikle otomatik olarak açılır) adresini
web tarayıcınızda açabilirsiniz. Sonra kitabın her bölümü için kodu çalıştırabilirsiniz.
Lütfen kitabın kodunu çalıştırmadan veya derin öğrenme çerçevesini veya `d2l`
paketini güncellemeden önce çalışma zamanı ortamını etkinleştirmek için daima
`conda enable d2l` komutunu çalıştırın. Ortamdan çıkmak için `conda deactivate` komutunu
çalıştırın.



## GPU Desteği
:label:`subsec_gpu`

:begin_tab:`mxnet`
Varsayılan olarak, derin öğrenme çerçevesi, herhangi bir bilgisayarda
(çoğu dizüstü bilgisayar dahil) çalışmasını sağlamak için GPU desteği olmadan
yüklenir.
Bu kitabın bir kısmı GPU ile çalışmayı gerektirir veya önerir.
Bilgisayarınızda NVIDIA grafik kartları varsa ve
[CUDA](https://developer.nvidia.com/cuda-downloads) yüklüyse, GPU-etkin bir
sürüm yüklemeniz gerekir.
Yalnızca CPU sürümünü yüklediyseniz, önce bunu çalıştırarak
kaldırmanız gerekebilir:


```bash
pip uninstall mxnet
```



Sonrasında yüklü CUDA sürümünüzü bulmanız gerekiyor.
`nvcc --version` veya `cat /usr/local/cuda/version.txt` ile kontrol edebilirsiniz.
CUDA 10.1'i yüklediğinizi varsayalım, o zaman aşağıdaki komutla kurabilirsiniz:

```bash
# Windows kullanıcıları için
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python

# Linux ve macOS kullanıcıları için
pip install mxnet-cu101==1.7.0
```

Son rakamları CUDA versiyonunuza göre değiştirebilirsiniz. Örn. CUDA 10.0 için `cu100` CUDA 9.0 için `cu90`.
:end_tab:


:begin_tab:`pytorch,tensorflow`
Derin öğrenme çercevesi, aksi belirtilmediyse GPU desteğiyle kurulacaktır. 
Eğer bilgisayarınızda NVIDIA GPU'su varsa ve [CUDA](https://developer.nvidia.com/cuda-downloads) kuruluysa, artık hazırsınız.
:end_tab:

## Alıştırmalar

1. Kitabın kodunu indirin ve runtime ortamını yükleyin.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/436)
:end_tab:
