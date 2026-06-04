import cv2

# 1. Eğitilmiş Yüz Modelini (XML dosyasını) Yükle
# Bu dosya, Python'a "yüz nedir?" onu öğretir.
yuz_tasarimci = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2. Üzerinde çalışacağımız fotoğrafı oku
# (Klasördeki 'test.jpg' dosyasını kullanır. İçinde insan yüzü olduğundan emin ol!)
resim = cv2.imread('test.jpg')

# Resim bulunamazsa programı durdur
if resim is None:
    print("Hata: 'test.jpg' bulunamadı! Klasörde olduğundan emin ol.")
    exit()

# 3. Fotoğrafı Gri Tonlamaya Çevir
# Yüz algılama algoritması renklerle değil, ışık ve gölge değişimleriyle (kontrast) çalışır.
gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

# 4. Yüzleri Algıla (Ana İşlem)
# detectMultiScale fonksiyonu, resimdeki yüzlerin koordinatlarını (x, y, genişlik, yükseklik) bulur.
# scaleFactor: Resmin ne kadar küçültülerek taranacağını belirler (1.1 = %10 küçült).
# minNeighbors: Bir bölgenin yüz sayılması için kaç komşu bölge tarafından doğrulanması gerektiğini belirler.
yuzler = yuz_tasarimci.detectMultiScale(gri_resim, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print(f"Fotoğrafta {len(yuzler)} tane yüz bulundu.")

# 5. Algılanan Yüzlerin Etrafına Dikdörtgen Çiz
# 'yuzler' değişkeni içinde her bir yüz için [x, y, w (genişlik), h (yükseklik)] bilgilerini tutar.
for (x, y, w, h) in yuzler:
    # cv2.rectangle(resim, başlangıç_noktası, bitiş_noktası, renk, kalınlık)
    # (0, 255, 0) => BGR formatında Yeşil renk demektir.
    cv2.rectangle(resim, (x, y), (x+w, y+h), (0, 255, 0), 3)

# 6. Sonucu Göster
cv2.imshow('Yuzleri Bulan Sistem', resim)

# Kaydetmek istersen (isteğe bağlı)
# cv2.imwrite('bulunan_yuzler.jpg', resim)

cv2.waitKey(0)
cv2.destroyAllWindows()