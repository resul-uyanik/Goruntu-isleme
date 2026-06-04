from ultralytics import YOLO
import cv2

print("Canlı kamera başlatılıyor... Çıkmak için 'q' tuşuna basın.")

# 1. Modeli Yükle
model = YOLO("yolov8n.pt")

# 2. Türkçe İsimleri Ayarla
turkce_isimler = {
    0: 'insan', 1: 'bisiklet', 2: 'araba', 3: 'motosiklet', 4: 'ucak', 5: 'otobus', 6: 'tren', 7: 'kamyon', 8: 'tekne', 9: 'trafik lambasi',
    10: 'yangin muslugu', 11: 'dur tabelasi', 12: 'park metresi', 13: 'bank', 14: 'kus', 15: 'kedi', 16: 'kopek', 17: 'at', 18: 'koyun', 19: 'inek',
    20: 'fil', 21: 'ayi', 22: 'zebra', 23: 'zurafa', 24: 'sirt cantasi', 25: 'semsiye', 26: 'el cantasi', 27: 'kravat', 28: 'valiz', 29: 'frizbi',
    30: 'kayak', 31: 'snowboard', 32: 'spor topu', 33: 'ucurtma', 34: 'beyzbol sopasi', 35: 'beyzbol eldiveni', 36: 'kaykay', 37: 'sorf tahtasi', 38: 'tenis raketi', 39: 'sise',
    40: 'sarap kadehi', 41: 'kupa', 42: 'catal', 43: 'bicak', 44: 'kasik', 45: 'kase', 46: 'muz', 47: 'elma', 48: 'sandvic', 49: 'portakal',
    50: 'brokoli', 51: 'havuc', 52: 'sosisli', 53: 'pizza', 54: 'donut', 55: 'pasta', 56: 'sandalye', 57: 'kanepe', 58: 'saksi bitkisi', 59: 'yatak',
    60: 'yemek masasi', 61: 'tuvalet', 62: 'tv', 63: 'dizustu bilgisayar', 64: 'fare', 65: 'kumanda', 66: 'klavye', 67: 'cep telefonu', 68: 'mikrodalga', 69: 'firin',
    70: 'tost makinesi', 71: 'lavabo', 72: 'buzdolabi', 73: 'kitap', 74: 'saat', 75: 'vazo', 76: 'makas', 77: 'oyuncak ayi', 78: 'sac kurutma makinesi', 79: 'dis fircasi'
}
model.model.names = turkce_isimler

# 3. Kamerayı Başlat (0: Varsayılan kamera)
kamera = cv2.VideoCapture(0)

while True:
    # Kameradan anlık görüntüyü (kareyi) oku
    basarili_mi, kare = kamera.read()
    
    if not basarili_mi:
        print("Kameradan görüntü alınamadı!")
        break

    # 4. YOLO modelini bu anlık kare üzerinde çalıştır
    # stream=True ayarı, canlı yayınlarda modelin daha hızlı çalışmasını sağlar
    sonuclar = model(kare, stream=True)

    # 5. Sonuçları karenin üzerine çiz
    for sonuc in sonuclar:
        cizilmis_kare = sonuc.plot()

    # 6. İşlenmiş canlı görüntüyü ekranda göster
    cv2.imshow("Canli YOLO Nesne Tanima", cizilmis_kare)

    # Klavyeden 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# İşlem bitince kamerayı kapat ve pencereleri temizle
kamera.release()
cv2.destroyAllWindows()