import cv2
import numpy as np

print("Sanal Ressam (Çıplak El Sürümü) başlatılıyor...")
print("Lütfen kameraya işaret parmağınızı gösterin.")
print("Tuvali temizlemek için klavyeden 'c' tuşuna, çıkmak için 'q' tuşuna basın.")

# 1. Kamerayı başlat
kamera = cv2.VideoCapture(0)

tuval = None
# Çizginin rengi (Kırmızı)
cizgi_rengi = (0, 0, 255)

x_eski, y_eski = 0, 0

while True:
    basarili_mi, kare = kamera.read()
    if not basarili_mi:
        break

    # Kamerayı yatay çevir (Doğal hareket için)
    kare = cv2.flip(kare, 1)

    if tuval is None:
        tuval = np.zeros_like(kare)

    # 2. Renk Formatını Değiştir (BGR -> HSV)
    hsv_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)

    # --- KRİTİK AYAR: İNSAN TEN RENGİ ---
    # Bu sınırlar ışığa ve ten rengine göre değişebilir.
    # Genelde ten rengi HSV'de 0-20 renk tonu aralığındadır.
    alt_ten = np.array([0, 30, 60])
    ust_ten = np.array([20, 150, 255])

    # Sadece ten rengi olan yerleri beyaz, diğer yerleri siyah yap
    maske = cv2.inRange(hsv_kare, alt_ten, ust_ten)

    # Gürültüleri temizlemek için morfolojik işlemler (Hafiflettik)
    maske = cv2.erode(maske, None, iterations=1)
    maske = cv2.dilate(maske, None, iterations=2)

    # 3. Elin Koordinatlarını Bul
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(konturlar) > 0:
        # Ekranda ten rengine sahip en büyük konturu (elini) bul
        en_buyuk_kontur = max(konturlar, key=cv2.contourArea)

        # 1000 pikselden büyükse çizim yap (yüzünü falan almasın)
        if cv2.contourArea(en_buyuk_kontur) > 1000:
            ((x, y), yari_cap) = cv2.minEnclosingCircle(en_buyuk_kontur)
            x_yeni, y_yeni = int(x), int(y)

            # Elin merkezine takip yuvarlağı çiz (Görmemiz için)
            cv2.circle(kare, (x_yeni, y_yeni), int(yari_cap), (0, 255, 255), 2)

            if x_eski == 0 and y_eski == 0:
                x_eski, y_eski = x_yeni, y_yeni

            # Çizgiyi tuvale çiz
            cv2.line(tuval, (x_eski, y_eski), (x_yeni, y_yeni), cizgi_rengi, 8)

            # Koordinatları güncelle
            x_eski, y_eski = x_yeni, y_yeni
    else:
        # El yoksa koordinatları sıfırla
        x_eski, y_eski = 0, 0

    # 4. Görüntüleri Birleştir
    son_goruntu = cv2.add(kare, tuval)

    # Sonucu ekranda göster
    cv2.imshow("Çıplak El ile Sanal Ressam", son_goruntu)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('q'):
        break
    elif tus == ord('c'):
        tuval = np.zeros_like(kare)

kamera.release()
cv2.destroyAllWindows()