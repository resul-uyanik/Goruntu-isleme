import cv2

# Yüz modelini yükle
yuz_tasarimci = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Kamerayı başlat (0, bilgisayarının varsayılan kamerasını temsil eder)
kamera = cv2.VideoCapture(0)

print("Kamera açılıyor... Çıkmak için kameranın seçili olduğu penceredeyken klavyeden 'q' tuşuna bas.")

while True:
    # Kameradan o anki kareyi (fotoğrafı) oku
    basarili_mi, kare = kamera.read()

    # Eğer kameradan görüntü alınamazsa döngüyü kır
    if not basarili_mi:
        print("Hata: Kameradan görüntü alınamadı!")
        break

    # İşlemi hızlandırmak için kareyi gri tonlamaya çevir
    gri_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    yuzler = yuz_tasarimci.detectMultiScale(gri_kare, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Bulunan yüzlerin etrafına yeşil dikdörtgen çiz
    for (x, y, w, h) in yuzler:
        cv2.rectangle(kare, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Sonucu ekranda göster (Artık video gibi akacak)
    cv2.imshow('Canli Yuz Bulucu - Cikmak icin Q', kare)

    # Klavyeden 'q' tuşuna basılıp basılmadığını kontrol et
    # (cv2.waitKey(1) 1 milisaniye bekler, canlı akış için gereklidir)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# İşlem bitince kamerayı serbest bırak ve pencereleri kapat
kamera.release()
cv2.destroyAllWindows()