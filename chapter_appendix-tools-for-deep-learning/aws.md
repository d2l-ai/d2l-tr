# AWS EC2 Örneklerini Kullanma
:label:`sec_aws`

Bu bölümde, tüm kütüphaneleri ham Linux makinesine nasıl yükleyeceğinizi göstereceğiz. :numref:`sec_sagemaker`'te Amazon SageMaker'ı nasıl kullanacağınızı tartıştığımızı, kendi başınıza bir örnek oluşturmanın AWS'de daha az maliyeti olduğunu unutmayın. Walkthrough bir dizi adım içerir: 

1. AWS EC2'den bir GPU Linux örneği isteği.
1. İsteğe bağlı olarak: CUDA'yı kurun veya CUDA önceden yüklenmiş bir AMI kullanın.
1. İlgili MXNet GPU sürümünü ayarlayın.

Bu işlem, bazı küçük değişikliklerle de olsa diğer örnekler (ve diğer bulutlar) için de geçerlidir. İleriye gitmeden önce, bir AWS hesabı oluşturmanız gerekir, daha fazla ayrıntı için :numref:`sec_sagemaker`'e bakın. 

## EC2 Örneği Oluşturma ve Çalıştırma

AWS hesabınıza giriş yaptıktan sonra EC2 paneline gitmek için “EC2" (:numref:`fig_aws`'te kırmızı kutuyla işaretlenmiş) tıklayın. 

![Open the EC2 console.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2`, hassas hesap bilgilerinin gri renkte olduğu EC2 panelini gösterir. 

![EC2 panel.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### Yer Ön Ayar Gecikmeyi azaltmak için yakındaki bir veri merkezi seçin, örneğin “Oregon” (:numref:`fig_ec2`'ün sağ üstündeki kırmızı kutuyla işaretlenmiştir). Çin'de bulunuyorsanız, Seul veya Tokyo gibi yakındaki Asya Pasifik bölgesini seçebilirsiniz. Bazı veri merkezlerinin GPU örneklerine sahip olmayabileceğini lütfen unutmayın. 

### Sınırları Artırmak Bir örneği seçmeden önce, :numref:`fig_ec2`'te gösterildiği gibi soldaki çubuktaki “Limitler” etiketine tıklayarak miktar kısıtlamaları olup olmadığını kontrol edin. :numref:`fig_limits`, bu tür bir sınırlamanın bir örneğini gösterir. Hesap şu anda bölge başına “p2.xlarge” örneğini açamıyor. Bir veya daha fazla örnek açmanız gerekiyorsa, daha yüksek bir örnek kotasına başvurmak için “Limit artışı iste” bağlantısına tıklayın. Genellikle, bir uygulamanın işlenmesi bir iş günü sürer. 

![Instance quantity restrictions.](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### Örnek Sonraki Başlatma, örneğinizi başlatmak için :numref:`fig_ec2`'te kırmızı kutuyla işaretlenmiş “Örneği Başlat” düğmesine tıklayın. 

Uygun bir AMI (AWS Machine Image) seçerek başlıyoruz. Arama kutusuna “Ubuntu” girin (:numref:`fig_ubuntu`'te kırmızı kutu ile işaretlenmiş). 

![Choose an operating system.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2, aralarından seçim yapabileceğiniz birçok farklı örnek yapılandırması sağlar. Bu bazen bir acemi için ezici hissedebilir. İşte uygun makinelerin bir tablosu: 

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | ancient                       |
| p2   | Kepler K80  | old but often cheap as spot   |
| g3   | Maxwell M60 | good trade-off                |
| p3   | Volta V100  | high performance for FP16     |
| g4   | Turing T4   | inference optimized FP16/INT8 |

Yukarıdaki tüm sunucular, kullanılan GPU sayısını belirten birden fazla lezzet sunar. Örneğin, bir p2.xlarge 1 GPU'ya ve p2.16xlarge 16 GPU'ya ve daha fazla belleğe sahiptir. Daha fazla ayrıntı için bkz. [AWS EC2 documentation](https732293614). 

**Not: ** uygun sürücülere sahip bir GPU etkin örneği ve GPU etkin bir MXNet sürümünü kullanmanız gerekir. Aksi takdirde GPU'ları kullanmaktan herhangi bir fayda görmezsiniz.

![Choose an instance.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

Şimdiye kadar, :numref:`fig_disk`'ün üstünde gösterildiği gibi, bir EC2 örneğini başlatmak için yedi adımdan ilk ikisini tamamladık. Bu örnekte, “3 adımları için varsayılan yapılandırmaları saklıyoruz. Örneği Yapılandır”, “5. Etiket Ekle” ve “6. Güvenlik Grubunu Yapılandır”. “4'e dokun. Depolama Ekle” ve varsayılan sabit disk boyutunu 64 GB'a yükseltin (:numref:`fig_disk` kırmızı kutuda işaretlenmiş). CUDA'nın kendi başına zaten 4 GB aldığını unutmayın. 

![Modify instance hard disk size.](../img/disk.png)
:width:`700px`
:label:`fig_disk`

Son olarak, 7'ye git. İnceleyin” ve yapılandırılmış örneği başlatmak için “Başlat” ı tıklayın. Sistem artık örneğe erişmek için kullanılan anahtar çiftini seçmenizi ister. Anahtar çiftiniz yoksa, anahtar çifti oluşturmak için :numref:`fig_keypair`'teki ilk açılır menüden “Yeni bir anahtar çifti oluştur” u seçin. Daha sonra, bu menü için “Varolan bir anahtar çiftini seç” i seçip daha önce oluşturulmuş anahtar çiftini seçebilirsiniz. Oluşturulan örneği başlatmak için “Örnekleri Başlat” ı tıklayın. 

![Select a key pair.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

Yeni bir tane oluşturduysanız anahtar çiftini indirdiğinizden ve güvenli bir konumda sakladığınızdan emin olun. Bu sunucuya SSH için tek yoldur. Bu örneğin durumunu görüntülemek için :numref:`fig_launching`'te gösterilen örnek kimliğini tıklatın. 

![Click the instance ID.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### Örnek Bağlanma

:numref:`fig_connect`'te gösterildiği gibi, örnek durumu yeşile döndükten sonra örneği sağ tıklatın ve örnek erişim yöntemini görüntülemek için `Connect`'i seçin. 

![View instance access and startup method.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

Bu yeni bir anahtarsa, SSH'nin çalışması için herkese açık bir şekilde görüntülenmemelidir. `D2L_key.pem`'ü sakladığınız klasöre gidin (örn. İndirilenler klasörü) ve anahtarın genel olarak görüntülenmediğinden emin olun.

```bash
cd /Downloads  ## if D2L_key.pem is stored in Downloads folder
chmod 400 D2L_key.pem
```

![View instance access and startup method.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`

Şimdi, :numref:`fig_chmod`'ün alt kırmızı kutusuna ssh komutunu kopyalayın ve komut satırına yapıştırın:

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

Komut satırı “Bağlanmaya devam etmek istediğinizden emin misiniz (evet/hayır)” sorduğunda, “evet” yazın ve örneğe giriş yapmak için Enter tuşuna basın. 

Sunucunuz şimdi hazır. 

## CUDA Kurulumu

CUDA'yı yüklemeden önce, örneği en son sürücülerle güncellediğinizden emin olun.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

Burada CUDA 10.1'i indiriyoruz. NVIDIA'nın [resmi deposu](https://developer.nvidia.com/cuda-downloads) to find the download link of CUDA 10.1 as shown in :numref:`fig_cuda`'ü ziyaret edin. 

![Find the CUDA 10.1 download address.](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

Talimatları kopyalayın ve CUDA 10.1'i yüklemek için terminale yapıştırın.

```bash
## Paste the copied link from CUDA website
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

Programı yükledikten sonra, GPU'ları görüntülemek için aşağıdaki komutu çalıştırın.

```bash
nvidia-smi
```

Son olarak, diğer kütüphanelerin bulmasına yardımcı olmak için kütüphane yoluna CUDA'yı ekleyin.

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```

## MXNet'i Yükleme ve D2L Dizüstü Bilgisayarları İndirme

Öncelikle, kurulumu basitleştirmek için Linux için [Miniconda](https://conda.io/en/latest/miniconda.html)'i yüklemeniz gerekir. İndirme bağlantısı ve dosya adı değişikliklere tabidir, bu nedenle lütfen Miniconda web sitesine gidin ve :numref:`fig_miniconda`'te gösterildiği gibi “Bağlantı Adresini Kopyala” düğmesine tıklayın. 

![Download Miniconda.](../img/miniconda.png)
:width:`700px`
:label:`fig_miniconda`

```bash
# The link and file name are subject to changes
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Miniconda kurulumundan sonra CUDA ve conda'yı etkinleştirmek için aşağıdaki komutu çalıştırın.

```bash
~/miniconda3/bin/conda init
source ~/.bashrc
```

Ardından, bu kitabın kodunu indirin.

```bash
sudo apt-get install unzip
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Ardından conda `d2l` ortamını oluşturun ve yüklemeye devam etmek için `y` girin.

```bash
conda create --name d2l -y
```

`d2l` ortamını oluşturduktan sonra etkinleştirin ve `pip`'i yükleyin.

```bash
conda activate d2l
conda install python=3.7 pip -y
```

Son olarak, MXNet ve `d2l` paketini yükleyin. Soneki `cu101`, bunun CUDA 10.1 varyantı olduğu anlamına gelir. Farklı sürümler için, sadece CUDA 10.0 deyin, bunun yerine `cu100`'yı seçmek istersiniz.

```bash
pip install mxnet-cu101==1.7.0
pip install git+https://github.com/d2l-ai/d2l-en
```

Her şeyin yolunda olup olmadığını hızlı bir şekilde test edebilirsiniz:

```
$ python
>>> from mxnet import np, npx
>>> np.zeros((1024, 1024), ctx=npx.gpu())
```

## Jupyter Koşu

Jupyter'ı uzaktan çalıştırmak için SSH bağlantı noktası yönlendirme kullanmanız gerekir. Sonuçta, buluttaki sunucunun bir monitörü veya klavyesi yoktur. Bunun için sunucunuza masaüstünüzden (veya dizüstü bilgisayarınızdan) aşağıdaki gibi oturum açın.

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
conda activate d2l
jupyter notebook
```

:numref:`fig_jupyter`, Jupyter Notebook çalıştırdıktan sonra olası çıktıyı gösterir. Son satır 8888 numaralı bağlantı noktasının URL'sini oluşturur. 

![Output after running Jupyter Notebook. The last row is the URL for port 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

Bağlantı noktası 8889 numaralı bağlantı noktasına yönlendirmeyi kullandığınız için bağlantı noktası numarasını değiştirmeniz ve yerel tarayıcınızda URL'yi açarken Jupyter tarafından verilen sırrı kullanmanız gerekir. 

## Kullanılmayan Örnekleri Kapatma

Bulut hizmetleri kullanım süresine göre faturalandırıldığından, kullanılmayan örnekleri kapatmanız gerekir. Alternatifler olduğunu unutmayın: Bir örneği “durdurmak”, yeniden başlatabilmeniz anlamına gelir. Bu, normal sunucunuzun gücünü kapatmaya benzer. Ancak durdurulan örnekler, korunan sabit disk alanı için küçük bir miktar faturalandırılır. “Sonlandır”, onunla ilişkili tüm verileri siler. Bu diski içerir, bu nedenle yeniden başlatamazsınız. Sadece gelecekte ihtiyacınız olmayacağını biliyorsanız bunu yapın. 

Örneği daha birçok örnek için şablon olarak kullanmak istiyorsanız, :numref:`fig_connect`'te örneğe sağ tıklayın ve örneğin görüntüsünü oluşturmak için “Görüntü” $\rightarrow$ “Oluştur” u seçin. Bu işlem tamamlandıktan sonra örneği sonlandırmak için “Örnek Durumu” $\rightarrow$ “Sonlandır"ı seçin. Bu örneği bir sonraki kullanmak istediğinizde, kaydedilen görüntüye dayalı bir örnek oluşturmak için bu bölümde açıklanan bir EC2 örneği oluşturma ve çalıştırma adımlarını uygulayabilirsiniz. Tek fark, “1 “deki. :numref:`fig_ubuntu`'te gösterilen AMI” seçeneğini seçin, kayıtlı resminizi seçmek için soldaki “AMI'lerim” seçeneğini kullanmanız gerekir. Oluşturulan örnek, görüntü sabit diskinde depolanan bilgileri saklar. Örneğin, CUDA ve diğer çalışma zamanı ortamlarını yeniden yüklemeniz gerekmez. 

## Özet

* Kendi bilgisayarınızı satın almak ve oluşturmak zorunda kalmadan isteğe bağlı örnekleri başlatabilir ve durdurabilirsiniz.
* Kullanabilmeniz için uygun GPU sürücülerini yüklemeniz gerekir.

## Egzersizler

1. Bulut kolaylık sağlar, ancak ucuza gelmez. Fiyatları nasıl düşüreceğinizi görmek için [spot instances](https://aws.amazon.com/ec2/spot/)'ü nasıl başlatacağınızı öğrenin.
1. Farklı GPU sunucuları ile deney yapın. Ne kadar hızlılar?
1. Çoklu GPU sunucuları ile deney yapın. İşleri ne kadar iyi büyütebilirsin?

[Discussions](https://discuss.d2l.ai/t/423)
