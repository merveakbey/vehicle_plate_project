
# Vehicle Plate Project

Bu proje, video akışı üzerinden **araç tespiti**, **araç takibi**, **plaka tespiti** ve **optik karakter tanıma (OCR)** işlemlerini bir araya getiren uçtan uca bir **Araç Plaka Tanıma (ANPR - Automatic Number Plate Recognition)** sistemidir.

Proje kapsamında amaç; bir görüntü veya video içerisindeki araçları tespit etmek, araçlara ait plakaları bulmak, plaka bölgesini işleyerek okunabilir hale getirmek ve plaka metnini OCR ile tanımaktır. Sistem, hem masaüstü ortamında hem de gömülü yapay zekâ sistemlerine uyarlanabilecek şekilde geliştirilmeye uygundur.

---

## Projenin Amacı

Bu projenin temel amacı:

- Video veya kamera görüntüsünden araçları tespit etmek
- Araçlara ait plaka bölgelerini belirlemek
- Plaka görüntüsünü OCR için uygun hale getirmek
- Plaka üzerindeki karakterleri metne dönüştürmek
- Araç ile plaka bilgisini eşleştirmek
- Gerçek zamanlı çalışmaya uygun bir yapı oluşturmak
- Gerekirse gömülü sistemler ve hızlandırılmış backend’ler için optimize edilebilir bir temel sağlamak

---

## Temel Özellikler

- **Araç tespiti** için YOLO tabanlı model kullanımı
- **Plaka tespiti** için ayrı dedektör desteği
- **OCR entegrasyonu** ile plaka okuma
- **Araç takibi (tracking)** ile aynı aracı kareler boyunca izleme
- Araç-plaka eşleştirme mantığı
- Ön işleme adımları ile OCR başarısını artırma
- Takip (tracking) ile aynı araca ait plaka bilgisini kararlı hale getirme
- Çıktı videoya kutu, etiket ve plaka bilgisini yazdırma
- Farklı model formatlarına göre genişletilebilir yapı

---

## Kullanılan Teknolojiler

Projede kullanılan başlıca teknolojiler:

- **Python**
- **OpenCV**
- **Ultralytics YOLO**
- **NumPy**
- **OCR kütüphaneleri**
  - PaddleOCR
  - EasyOCR
  - Tesseract
- Gerekli durumlarda:
  - **PyTorch**
  - **ONNX / TensorRT / RKNN** tabanlı çıkarım yapıları

---

## Sistem Mimarisi

Proje genel olarak aşağıdaki adımlardan oluşur:

### 1. Görüntü / Video Alma
Sistem, video dosyasından veya canlı kamera akışından görüntü alır.

### 2. Araç Tespiti
Her kare üzerinde araç nesneleri tespit edilir. Araçlara ait bounding box koordinatları çıkarılır.

### 3. Araç Takibi (Tracking)
Tespit edilen araçlar, ardışık karelerde takip edilerek her araca bir kimlik atanır. Bu sayede aynı araç farklı karelerde yeniden eşleştirilebilir ve plaka bilgisi doğru araç üzerinde tutulabilir.

### 4. Plaka Tespiti
Tespit edilen araç bölgeleri veya tüm kare üzerinden plaka dedeksiyonu yapılır.

### 5. Araç-Plaka Eşleştirme
Bulunan plaka bölgesi, ilgili araç kutusuyla ilişkilendirilir. Böylece okunan plaka bilgisinin hangi araca ait olduğu belirlenir.

### 6. Ön İşleme
Plaka görüntüsü OCR doğruluğunu artırmak için işlenir.

### 7. OCR ile Metin Okuma
Ön işlenmiş plaka görüntüsü OCR modeline verilerek plaka metni okunur.

### 8. Sonuçların Kararlı Hale Getirilmesi
Tracking ve geçmiş OCR sonuçları birlikte değerlendirilerek aynı araca ait plaka sonucu daha güvenilir hale getirilir.

### 9. Görselleştirme
Araç kutusu, takip kimliği, plaka kutusu ve okunan plaka bilgisi çıktı görüntüsüne çizdirilir.

---

## Kullanım Senaryoları

Bu proje aşağıdaki alanlarda kullanılabilir:

- Akıllı ulaşım sistemleri
- Otopark giriş-çıkış kontrolü
- Güvenlik ve izleme sistemleri
- Trafik analizi
- Gömülü yapay zekâ uygulamaları
- Edge AI / NPU hızlandırmalı görüntü işleme projeleri

---

## Proje Yapısı

Aşağıda örnek bir proje düzeni verilmiştir. Kendi repodaki dosya isimlerine göre güncelleyebilirsin:

```bash
vehicle_plate_project/
│
├── models/                  # Araç ve plaka tespit modelleri
├── outputs/                 # Çıktı video ve görseller
├── samples/                 # Test görselleri / videoları
├── utils/                   # Yardımcı fonksiyonlar
├── main.py                  # Ana çalıştırma dosyası
├── requirements.txt         # Bağımlılıklar
└── README.md
```

## Kurulum
1. Repoyu klonla
```bash
git clone https://github.com/merveakbey/vehicle_plate_project.git
cd vehicle_plate_project
```
2. Sanal ortam oluştur
```bash
python -m venv venv
```
Windows:
venv\Scripts\activate
Linux / macOS:
source venv/bin/activate
3. Gerekli paketleri yükle
```bash
pip install -r requirements.txt
```

## Çalıştırma

Projeyi çalıştırmak için örnek kullanım:
```bash
python main.py
```
Eğer proje komut satırı argümanları ile çalışıyorsa şu yapı da kullanılabilir:
```bash
python main.py --video "input.mp4" --vehicle-model "models/vehicle.pt" --plate-model "models/plate.pt"
```
Not: Bu kısmı repodaki gerçek dosya adına ve çalıştırma formatına göre güncellemen gerekir.

## OCR Ön İşleme Yaklaşımı

Plaka OCR başarısı, görüntü kalitesine doğrudan bağlıdır. Bu nedenle projede plaka bölgesi için çeşitli görüntü iyileştirme adımları uygulanabilir:

- Grayscale dönüştürme
- Resize / upscale
- Bilateral filter
- CLAHE
- Threshold işlemleri
- Gürültü azaltma

Bu yaklaşım özellikle düşük çözünürlüklü veya hareketli görüntülerde daha iyi sonuç alınmasına yardımcı olur.

## Geliştirici

**Merve Akbey**




