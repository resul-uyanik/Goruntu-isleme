import cv2
import numpy as np

print("Matrix Seviye 2: Saf Veri Görüşü Başlatılıyor...")
print("Dünyayı kodlar olarak görmeye hazır olun. Çıkmak için 'q' tuşuna basın.")

kamera = cv2.VideoCapture(0)

# Matrix dünyasında kullanacağımız karakter havuzu (Karanlıktan aydınlığa doğru sıralı)
# Karanlık yerlerde boşluk, aydınlık yerlerde 0 ve 1 kullanacağız.
karakter_havuzu = "   ..::==xX01"

# Font ayarları
font = cv2.FONT_HERSHEY_SIMPLEX
font_olcek = 0.4
kalinlik = 1

while True:
    basarili_mi, kare = kamera.read()
    if not basarili_mi:
        break

    kare = cv2.flip(kare, 1)
    
    # 1. Çözünürlüğü İyice Düşür (Pikselleri "Karakter" boyutuna getirmek için)
    # Genişliği 100 yapıyoruz. İ3 işlemcinin zorlanmaması için harika bir orandır.
    kucuk_genislik = 100
    oran = kucuk_genislik / kare.shape[1]
    kucuk_yukseklik = int(kare.shape[0] * oran)
    
    kucuk_kare = cv2.resize(kare, (kucuk_genislik, kucuk_yukseklik))
    
    # Rengi algılamak için gri tonlamaya çevir
    gri = cv2.cvtColor(kucuk_kare, cv2.COLOR_BGR2GRAY)
    
    # 2. Üzerine kodları yazacağımız simsiyah, dev bir tuval oluştur
    # Her bir karakter için 10 piksellik boşluk bırakıyoruz
    tuval_genislik = kucuk_genislik * 10
    tuval_yukseklik = kucuk_yukseklik * 10
    matrix_tuvali = np.zeros((tuval_yukseklik, tuval_genislik, 3), dtype=np.uint8)
    
    # 3. Her bir pikseli tek tek bir harfe (koda) dönüştür
    for i in range(kucuk_yukseklik):
        for j in range(kucuk_genislik):
            # O anki pikselin 0 ile 255 arasındaki ışık/parlaklık değerini al
            piksel_parlakligi = gri[i, j]
            
            # Bu parlaklığı bizim karakter havuzumuzun sayısına göre oranla
            indeks = int((piksel_parlakligi / 255) * (len(karakter_havuzu) - 1))
            karakter = karakter_havuzu[indeks]
            
            # Sadece boşluk olmayan karakterleri çiz (Bu sayede sistem çok daha hızlı çalışır)
            if karakter != " ":
                # Karakteri ekrana Matrix Yeşili renginde (0, 255, 0) bas
                cv2.putText(matrix_tuvali, karakter, (j * 10, i * 10), font, font_olcek, (0, 255, 0), kalinlik)
                
    # Sonucu ekranda göster
    cv2.imshow("Matrix - Koda Donusum", matrix_tuvali)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
kamera.release()
cv2.destroyAllWindows()