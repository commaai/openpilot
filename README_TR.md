<div align="center" style="text-align: center;">

<h1>openpilot</h1>

<p>
  <b>openpilot, robotik için bir işletim sistemidir.</b>
  <br>
  Şu anda 300'den fazla desteklenen aracın sürüş destek sistemlerini (ADAS) geliştirmektedir.
</p>

<h3>
  <a href="https://docs.comma.ai">Belgeler</a>
  <span> · </span>
  <a href="https://docs.comma.ai/contributing/roadmap/">Yol Haritası</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Katkıda Bulun</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Topluluk</a>
  <span> · </span>
  <a href="https://comma.ai/shop">comma four'da Deneyin</a>
</h3>

Hızlı başlangıç: `bash <(curl -fsSL openpilot.comma.ai)`

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63"></a></td>
  </tr>
</table>

## openpilot'ı Araçta Kullanmak

openpilot'ı bir araçta kullanabilmeniz için dört şeye ihtiyacınız vardır:
1. **Desteklenen Cihaz:** [comma.ai/shop/comma-four](https://www.comma.ai/shop/comma-four) adresinden temin edebileceğiniz bir comma four cihazı.
2. **Yazılım:** comma four'un kurulum prosedürü, kullanıcıların özel yazılımlar için URL girmesine olanak tanır. Resmi kararlı sürümü (release) yüklemek için `openpilot.comma.ai` URL'sini kullanın.
3. **Desteklenen Araç:** [300'den fazla desteklenen araçtan](docs/CARS.md) birine sahip olduğunuzdan emin olun.
4. **Araç Tesisatı (Harness):** comma four cihazınızı araca bağlamak için bir [araç tesisatına](https://comma.ai/shop/car-harness) ihtiyacınız olacaktır.

Tesisatın ve cihazın araca nasıl kurulacağına dair [detaylı talimatlarımız mevcuttur](https://comma.ai/setup). openpilot'ın tamamen tak-çalıştır olmasa da [farklı donanımlarda da çalıştırılabileceğini](https://blog.comma.ai/self-driving-car-for-free/) unutmayın.

### Dallar (Branches)

`master` ve diğer dalların doğrudan çalıştırılması desteklenmektedir, ancak önceden derlenmiş aşağıdaki dallardan birini kullanmanız önerilir:

| comma four branch      | comma 3X branch        | URL                                    | Açıklama                                                                            |
|------------------------|------------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release-mici`         | `release-tizi`         | openpilot.comma.ai                     | openpilot'ın ana sürüm (release) dalıdır. Kararlı versiyondur.                      |
| `release-mici-staging` | `release-tizi-staging` | openpilot-test.comma.ai                | Yeni sürümler için hazırlık (staging) dalıdır. Sürümleri biraz daha erken alırsınız.|
| `nightly`              | `nightly`              | openpilot-nightly.comma.ai             | En güncel geliştirme dalıdır (bleeding edge). Stabil çalışması garanti edilmez.     |
| `nightly-dev`          | `nightly-dev`          | installer.comma.ai/commaai/nightly-dev | `nightly` ile aynıdır, ancak bazı araçlar için deneysel geliştirme özellikleri içerir. |

[chestnut](https://comma.ai/shop/chestnut) için aşağıdaki yükleyici URL'lerini kullanın:

| branch                       | URL                                                        | Açıklama                                                                            |
|------------------------------|------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `release-chestnut`           | installer.comma.ai/commaai/release-chestnut                | openpilot'ın ana sürüm (release) dalıdır.                                           |
| `release-chestnut-staging`   | installer.comma.ai/commaai/release-chestnut-staging        | Yeni sürümler için hazırlık (staging) dalıdır. Sürümleri biraz daha erken alırsınız.|
| `nightly-chestnut`           | installer.comma.ai/commaai/nightly-chestnut                | En güncel geliştirme dalıdır (bleeding edge). Stabil çalışması garanti edilmez.     |
| `nightly-chestnut-dev`       | installer.comma.ai/commaai/nightly-chestnut-dev            | `nightly` ile aynıdır, ancak bazı araçlar için deneysel geliştirme özellikleri içerir. |

## openpilot Geliştirmeye Başlamak

openpilot, [comma](https://comma.ai/) ekibi ve sizin gibi kullanıcılar tarafından geliştirilmektedir. [GitHub](http://github.com/commaai/openpilot) üzerindeki "Pull Request" (çekme isteği) ve "Issue" (sorun bildirimi) bildirimlerinizi memnuniyetle karşılıyoruz.

- [Topluluk Discord'una](https://discord.comma.ai) katılın
- [Katkıda bulunma belgelerini (Contributing)](docs/CONTRIBUTING.md) inceleyin
- [openpilot araçlarına (tools)](openpilot/tools/) göz atın
- Kod dokümantasyonu https://docs.comma.ai adresinde yer almaktadır
- openpilot'ı çalıştırma hakkındaki bilgiler [topluluk wikisinde](https://github.com/commaai/openpilot/wiki) bulunmaktadır

openpilot üzerinde çalışarak para kazanmak ister misiniz? [comma işe alım yapıyor](https://comma.ai/jobs#open-positions) ve dışarıdan katkı sağlayan geliştiriciler için çeşitli [ödüller (bounties)](https://comma.ai/bounties) sunuyor.

## Güvenlik ve Test

- openpilot, [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) yönergelerine uyar. Daha fazla detay için [SAFETY.md](docs/SAFETY.md) dosyasına bakın.
- openpilot, her commit'te çalışan software-in-the-loop (yazılım döngü) [testlerine](.github/workflows/tests.yaml) sahiptir.
- Güvenlik modelini uygulayan kod bloğu `panda` içindedir ve C dili ile yazılmıştır, daha fazla detay için [kod titizliğine (code rigor)](https://github.com/commaai/panda#code-rigor) bakın.
- panda, software-in-the-loop [güvenlik testlerine](https://github.com/commaai/panda/tree/master/tests/safety) sahiptir.
- Kendi içimizde, çeşitli süreçleri derleyen ve birim (unit) testlerini yapan bir hardware-in-the-loop (donanım döngü) Jenkins test paketimiz bulunmaktadır.
- panda'nın ek hardware-in-the-loop [testleri](https://github.com/commaai/panda/blob/master/Jenkinsfile) mevcuttur.
- En güncel openpilot sürümünü, rotaları sürekli olarak tekrar oynatan 10 adet comma cihazı içeren bir test kabininde kesintisiz olarak çalıştırıyoruz.

<details>
<summary>MIT Lisanslı</summary>

openpilot, MIT lisansı altında yayınlanmaktadır. Yazılımın bazı bölümleri, belirtildiği gibi farklı lisanslar altında da yayınlanabilmektedir.

Bu yazılımın herhangi bir kullanıcısı; yazılımın kullanımından doğan, bununla ilgili veya bunun bir sonucu olarak ortaya çıkan tüm iddialar, talepler, davalar, tazminatlar, yükümlülükler, kayıplar, uzlaşmalar, hükümler, masraflar ve giderler (avukatlık ücretleri ve masrafları dahil ancak bunlarla sınırlı olmamak üzere) karşısında Comma.ai, Inc. ve yöneticilerini, yetkililerini, çalışanlarını, temsilcilerini, hissedarlarını, iştiraklerini, taşeronlarını ve müşterilerini tazmin edecek ve zarardan muaf tutacaktır.

**BU, YALNIZCA ARAŞTIRMA AMAÇLI ALFA KALİTESİNDE BİR YAZILIMDIR. BU BİR ÜRÜN DEĞİLDİR.
YEREL YASA VE DÜZENLEMELERE UYMAKTAN SİZ SORUMLUSUNUZ.
AÇIK VEYA ZIMNİ HİÇBİR GARANTİ VERİLMEMEKTEDİR.**
</details>

<details>
<summary>Kullanıcı Verileri ve comma Hesabı</summary>

Varsayılan olarak, openpilot sürüş verilerini sunucularımıza yükler. Verilerinize [comma connect](https://connect.comma.ai/) üzerinden de erişebilirsiniz. Verilerinizi, daha iyi yapay zeka modelleri eğitmek ve openpilot'ı herkes için geliştirmek amacıyla kullanıyoruz.

openpilot açık kaynaklı bir yazılımdır ve kullanıcılar dilerlerse veri toplamayı devre dışı bırakabilirler.

openpilot; yola bakan kameraları, CAN veri yolunu, GPS, IMU, manyetometre, termal sensörleri, çarpışma durumlarını ve işletim sistemi günlüklerini (log) kaydeder. Sürücüye bakan kamera ve mikrofon, yalnızca ayarlardan açıkça onay (opt-in) vermeniz durumunda kaydedilir.

openpilot'ı kullanarak [Gizlilik Politikamızı](https://comma.ai/privacy) kabul etmiş olursunuz. Bu yazılımın veya ilgili hizmetlerinin kullanımının, tamamen comma'nın takdirine bağlı olarak kaydedilebilecek ve saklanabilecek belirli türde kullanıcı verileri üreteceğini anlıyorsunuz. Bu sözleşmeyi kabul ederek, comma'ya bu verilerin kullanımı için geri alınamaz, kalıcı ve dünya çapında geçerli bir hak vermiş olursunuz.
</details>