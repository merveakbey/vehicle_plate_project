# KKTS - Kütüphane Kitap Takip Sistemi

Bu proje, kütüphane içerisindeki kullanıcı, kitap, kategori ve ödünç alma işlemlerinin düzenli bir şekilde yönetilmesini amaçlayan bir **Kütüphane Kitap Takip Sistemi** uygulamasıdır.

Projede hem **veritabanı tasarımı** hem de bu veritabanı ile bağlantılı çalışan **web tabanlı bir arayüz** geliştirilmiştir. Uygulama sayesinde kütüphane yönetim süreçleri daha düzenli, hızlı ve kontrol edilebilir hale getirilmektedir.

## Projenin Amacı

Bu projenin temel amacı:

- Kütüphanedeki kitapların kayıt altına alınmasını sağlamak
- Kullanıcı bilgilerini düzenli şekilde tutmak
- Kitap ödünç alma ve iade süreçlerini yönetmek
- Stok takibini yapmak
- Gecikme durumlarında ceza hesaplamasını desteklemek
- SQL tarafında prosedür, fonksiyon ve trigger kullanımını uygulamalı olarak göstermek

## Kullanılan Teknolojiler

Projede aşağıdaki teknolojiler kullanılmıştır:

- **MySQL** – Veritabanı yönetimi
- **Node.js** – Sunucu tarafı geliştirme
- **Express.js** – Web uygulama çatısı
- **EJS** – Arayüz şablonlama
- **HTML / CSS / JavaScript** – Kullanıcı arayüzü
- **Express Session** – Oturum yönetimi

## Proje Yapısı

```bash
KKTS/
│
├── config/               # Veritabanı ve yapılandırma dosyaları
├── middlewares/          # Yetkilendirme ve kontrol middleware yapıları
├── public/               # Statik dosyalar
├── routes/               # Uygulama rotaları
├── views/                # EJS sayfaları
├── kkts_db.sql           # Veritabanı oluşturma scripti
├── server.js             # Ana sunucu dosyası
├── package.json          # Proje bağımlılıkları
└── package-lock.json
'''

Veritabanı İçeriği

Projede oluşturulan veritabanı adı:

kkts_db

Sistemde temel olarak şu tablolar bulunmaktadır:

Kullanicilar
Kategoriler
Kitaplar
OduncAlma

Bu yapı sayesinde kullanıcı bilgileri, kitap bilgileri, kitap kategorileri ve ödünç alma işlemleri birbiriyle ilişkili şekilde tutulmaktadır.

SQL Özellikleri

Bu projede yalnızca tablo yapıları değil, aynı zamanda SQL'in ileri seviye özellikleri de kullanılmıştır:

Stored Procedure'ler
KullaniciEkle
KitapEkle
OduncAlmaEkle
StokGuncelle
KullaniciGuncelle
KullaniciSil
KitapSil
KullaniciOduncListesi
StoktaKitaplar
Function
GecikmeHesapla
Kitap iade tarihi geciktiğinde uygulanacak ceza miktarını hesaplar.
Trigger'lar
StokAzalt
Yeni ödünç alma işlemi yapıldığında kitap stoğunu 1 azaltır.
StokArtir
Kitap iade edildiğinde stok miktarını artırır.
Uygulama Özellikleri

Sistemde yer alan başlıca özellikler şunlardır:

Admin giriş sistemi
Oturum kontrolü
Kullanıcı yönetimi
Kitap yönetimi
Ödünç alma işlemleri
Kitap stoğu takibi
Gecikme cezası mantığı
Yetkisiz erişimi engelleyen middleware yapısı
Kurulum

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

1. Projeyi klonlayın
git clone https://github.com/merveakbey/KKTS.git
cd KKTS
2. Bağımlılıkları yükleyin
npm install
3. Veritabanını oluşturun

MySQL üzerinde kkts_db.sql dosyasını çalıştırarak veritabanını ve gerekli tabloları oluşturun.

4. Veritabanı bağlantı ayarlarını düzenleyin

config klasörü içindeki veritabanı bağlantı dosyasını kendi MySQL bilgilerinizle güncelleyin.

5. Uygulamayı başlatın
npm start

Ardından tarayıcıdan aşağıdaki adrese gidin:

http://localhost:3000
Giriş Sistemi Hakkında

Uygulamada giriş işlemi admin tablosu üzerinden yapılmaktadır. Kullanıcı giriş yaptıktan sonra oturum başlatılır ve yetkili sayfalara erişim sağlanır. Giriş yapılmadan kullanıcı, korumalı sayfalara erişemez.

Öğrenim Kazanımları

Bu proje sayesinde aşağıdaki konularda uygulamalı deneyim kazanılmıştır:

İlişkisel veritabanı tasarımı
SQL tablo oluşturma işlemleri
Primary Key / Foreign Key kullanımı
Stored Procedure yazımı
Function ve Trigger mantığı
Node.js ile MySQL bağlantısı kurma
Express.js ile route yönetimi
Session tabanlı kullanıcı doğrulama
EJS ile dinamik sayfa oluşturma
Geliştirmeye Açık Yönler

Bu proje temel işlevleri yerine getirecek şekilde hazırlanmıştır. İleride aşağıdaki geliştirmeler yapılabilir:

Şifrelerin hashlenmesi
Kullanıcı rolleri eklenmesi
Detaylı raporlama ekranı
Kitap arama ve filtreleme
Gecikmiş kitaplar için otomatik listeleme
Daha modern ve responsive arayüz tasarımı
Geliştirici

Merve Akbey
