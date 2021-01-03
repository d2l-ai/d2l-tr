# Kodlayıcı-Kodçözücü Mimarisi
:label:`sec_encoder-decoder`

:numref:`sec_machine_translation`'te tartıştığımız gibi, makine çevirisi, giriş ve çıkışı hem değişken uzunlukta diziler olan dizi iletim modelleri için önemli bir sorun alanıdır. Bu tür giriş ve çıkışları işlemek için iki ana bileşenle bir mimari tasarlayabiliriz. İlk bileşen bir *kodlayıcı*: girdi olarak değişken uzunlukta bir diziyi alır ve sabit bir şekle sahip bir duruma dönüştürür. İkinci bileşen bir *kod çözücü*: sabit bir şeklin kodlanmış durumunu değişken uzunlukta bir diziye eşler. Bu, :numref:`fig_encoder_decoder`'te tasvir edilen bir *kodlayıcı-kod çözücüsü* mimarisi olarak adlandırılır.

![Kodlayıcı-kodçözücü mimarisi.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

Örnek olarak İngilizce'den Fransızca'ya makine çevirisini ele alalım. İngilizce bir giriş dizisi göz önüne alındığında: “Onlar”, “are”, “izlemek”,”.“, bu kodlayıcı-kod çözücü mimarisi önce değişken uzunluklu girdiyi bir duruma kodlar, sonra da çevirilmiş sekans belirteci tarafından çıktı olarak oluşturmak için durumu deşifre eder: “Ils”, “regardent”, “.”. Kodlayıcı-kod çözücü mimarisi, sonraki bölümlerde farklı dizi iletim modellerinin temelini oluşturduğundan, bu bölüm bu mimariyi daha sonra uygulanacak bir arayüze dönüştürecektir.

## Kodlayıcı

Kodlayıcı arayüzünde, kodlayıcının `X` giriş olarak değişken uzunlukta dizileri aldığını belirtiyoruz. Uygulama, bu temel `Encoder` sınıfını miras alan herhangi bir model tarafından sağlanacaktır.

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
from torch import nn

#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

## Kodçözücü

Aşağıdaki kodçözücü arayüzünde, kodlayıcı çıkışını (`enc_outputs`) kodlanmış duruma dönüştürmek için ek bir `init_state` işlevi ekliyoruz. Bu adımın :numref:`subsec_mt_data_loading`'te açıklanan girdinin geçerli uzunluğu gibi ek girdilerin gerekebileceğini unutmayın. Belirteç ile değişken uzunlukta bir dizi belirteci oluşturmak için, kod çözücü bir girdi eşleyebilir (örneğin, önceki zaman adımında oluşturulan belirteç) ve kodlanmış durumu geçerli zaman adımında bir çıkış belirteci içine her zaman.

```{.python .input}
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## Kodlayıcıyı ve Kodçözücüyü Bir Araya Koyma

Sonunda, kodlayıcı-kod çözücü mimarisi, isteğe bağlı olarak ekstra argümanlarla hem bir kodlayıcı hem de bir kod çözücü içerir. İleri yayılımda, kodlayıcının çıkışı kodlanmış durumu üretmek için kullanılır ve bu durum kod çözücü tarafından girdilerinden biri olarak daha da kullanılacaktır.

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab pytorch
#@save
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

Kodlayıcı-kodçözücü mimarisinde “durum” terimi, muhtemelen devletlerle olan sinir ağlarını kullanarak bu mimariyi uygulamak için size ilham vermiştir. Bir sonraki bölümde, bu kodlayıcı-kod çözücü mimarisine dayanan dizi iletim modellerini tasarlamak için RNN'lerin nasıl uygulanacağını göreceğiz.

## Özet

* Kodlayıcı-kodçözücü mimarisi, hem değişken uzunlukta diziler olan giriş ve çıkışları işleyebilir, bu nedenle makine çevirisi gibi dizi iletim problemleri için uygundur.
* Kodlayıcı, giriş olarak değişken uzunlukta bir diziyi alır ve sabit bir şekle sahip bir duruma dönüştürür.
* Kodçözücü, sabit bir şeklin kodlanmış durumunu değişken uzunlukta bir diziye eşler.

## Alıştırmalar

1. Kodlayıcı-kodçözücü mimarisini uygulamak için sinir ağları kullandığımızı varsayalım. Kodlayıcı ve kodçözücü aynı tür sinir ağı olmak zorunda mı?
1. Makine çevirisinin yanı sıra, kodlayıcı-kodçözücü mimarisinin uygulanabileceği başka bir uygulama düşünebiliyor musunuz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1061)
:end_tab:
