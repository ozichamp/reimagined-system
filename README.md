# Optimized Hand Tracking with Key Press Detection

Bu proje, MediaPipe ve YOLO kullanarak gerçek zamanlı el takibi ve klavye tuş basışı algılama sistemi sağlar.

## Özellikler

- **21 Landmark El Takibi**: Her iki el için MediaPipe ile 21 adet landmark takibi
- **Gerçek Zamanlı Performans**: 30+ FPS performans hedefi
- **YOLO Entegrasyonu**: Klavye tuşlarının otomatik tespiti
- **Akıllı Tuş Algılama**: Z-koordinatı derinlik analizi ve mesafe hesaplamaları
- **Görsel Geri Bildirim**: Yazılan metni gerçek zamanlı görüntüleme
- **Hata Yönetimi**: Kapsamlı hata yakalama ve logging

## Gereksinimler

### Donanım
- Webcam (USB veya dahili)
- Standart CPU (GPU gerekli değil)
- En az 4GB RAM

### Yazılım
- Python 3.8+
- Gerekli kütüphaneler (requirements.txt'te listelendi)

## Kurulum

1. **Repoyu klonlayın**:
```bash
git clone <repo-url>
cd hand-tracking-project
```

2. **Virtual environment oluşturun** (önerilen):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

3. **Bağımlılıkları yükleyin**:
```bash
pip install -r requirements.txt
```

4. **YOLO modelini yerleştirin**:
- YOLO model dosyanızı (`best.pt`) uygun konuma yerleştirin
- Kod içinde model yolunu güncelleyin (varsayılan: `"D:/google collab rew/best.pt"`)

## Kullanım

### Temel Çalıştırma

```bash
python HandTracking.py
```

### Kontroller

- **'q'**: Uygulamadan çık
- **'c'**: Yazılan metni temizle

## Sistem Parametreleri

### Performans Ayarları
- **FPS Hedefi**: 30+ FPS
- **Kamera Çözünürlüğü**: 640x480
- **YOLO Görüntü Boyutu**: 320x320 (hız için optimize edildi)

### Algılama Parametreleri
- **Mesafe Eşiği**: 45 piksel
- **Z-koordinat Eşiği**: -0.02
- **Güven Eşiği**: 0.3
- **Algılama Cooldown**: 0.5 saniye

## Test Prosedürleri

### 1. Temel Fonksiyon Testi

**El Landmark Görselleştirmesi**:
- Uygulamayı başlatın
- Ellerinizi kamera önüne getirin
- Her iki elde 21 yeşil nokta ve bağlantıları göründüğünü kontrol edin
- Parmak uçlarının büyük yeşil dairelerle vurgulandığını doğrulayın

### 2. YOLO Tuş Algılama Testi

**Klavye Tespiti**:
- Bir klavyeyi kamera görüş alanına yerleştirin
- Mavi bounding box'ların tuşları çevrelediğini kontrol edin
- Her tuşun etiket ve güven skorunu görüntülediğini doğrulayın

### 3. Tuş Basışı Algılama Testi

**Landmark-Tuş Etkileşimi**:
- Parmağınızı bir tuşa yaklaştırın (45 piksel mesafe içinde)
- Parmağınızı tuşa doğru hareket ettirin (z-koordinatı değişimi)
- Terminalde "Landmark X-Y algıladı: Z" mesajını kontrol edin
- Sol üst köşede (50,50) yazılan metni görüntülediğini doğrulayın

### 4. Performans Testi

**FPS Ölçümü**:
- Sağ üst köşedeki FPS sayacını izleyin
- FPS'nin 30'un üzerinde olduğunu doğrulayın
- Sistem kaynaklarını monitör edin (CPU kullanımı)

### 5. Hata Yönetimi Testi

**Kamera Bağlantısı**:
- Kamerayı çıkarın/kapatın
- Terminalde "Error: Camera initialization failed" mesajını kontrol edin

**YOLO Model Hatası**:
- Yanlış model yolu belirtin
- "Error loading YOLO model" mesajını kontrol edin

### 6. Metin Görüntüleme Testi

**Görsel Çıktı**:
- Birkaç tuşa basın
- Sol üst köşede siyah arka plan üzerinde beyaz metin görüntülendiğini kontrol edin
- Font boyutunun (1.5) okunabilir olduğunu doğrulayın
- Padding'in (20 piksel) yeterli olduğunu kontrol edin

## Beklenen Çıktılar

### Görsel Çıktılar
- **El Landmarks**: Yeşil noktalar ve bağlantılar
- **Tuş Kutucukları**: Mavi bounding box'lar
- **Yazılan Metin**: Sol üst köşede siyah kutu içinde beyaz metin
- **FPS Sayacı**: Sağ üst köşede yeşil metin

### Terminal Çıktıları
```
Starting Optimized Hand Tracking System...
==================================================
YOLO model loaded successfully from: D:/google collab rew/best.pt
Hand tracker initialized successfully!
Distance threshold: 45 pixels
Z-coordinate threshold: -0.02
YOLO confidence threshold: 0.3
Camera initialized successfully!
Press 'q' to quit, 'c' to clear typed text
Landmark 0-8 detected: a
Landmark 1-12 detected: b
```

## Sorun Giderme

### Yaygın Problemler

1. **Düşük FPS**:
   - Kamera çözünürlüğünü düşürün
   - YOLO görüntü boyutunu azaltın (imgsz parametresi)
   - Model karmaşıklığını düşürün (model_complexity=0)

2. **Yanlış Tuş Algılama**:
   - Mesafe eşiğini ayarlayın (distance_threshold)
   - Z-koordinat eşiğini fine-tune edin (z_threshold)
   - Cooldown süresini artırın (detection_cooldown)

3. **YOLO Model Yükleme Hatası**:
   - Model yolunu kontrol edin
   - Model dosyasının var olduğunu doğrulayın
   - Model formatının doğru olduğunu kontrol edin

4. **Kamera Bağlantı Sorunu**:
   - Kamera izinlerini kontrol edin
   - Başka uygulamaların kamerayı kullanmadığından emin olun
   - Farklı kamera indeksi deneyin (cv2.VideoCapture(1))

## Geliştirme Notları

### Optimizasyon İpuçları
- MediaPipe model karmaşıklığı 0'da tutuldu (performans için)
- YOLO görüntü boyutu 320x320 (hız-doğruluk dengesi)
- Cooldown mekanizması tekrarlayan algılamaları önler
- FPS sayacı performans izleme için eklendi

### Genişletme Fırsatları
- Daha fazla el gesture tanıma
- Özel tuş kombinasyonları (Ctrl+C, Alt+Tab)
- Ses geri bildirimi
- Konfigürasyon dosyası desteği
- Multi-threading optimizasyonu

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## İletişim

Sorularınız veya önerileriniz için issue açabilirsiniz.