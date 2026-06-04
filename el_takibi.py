from ultralytics import YOLO
import cv2

print("YOLO İskelet ve Vücut Takibi başlatılıyor... Lütfen bekleyin.")

# 1. YOLO'nun İskelet/Duruş (Pose) modelini yüklüyoruz.
# 'yolov8n-pose.pt' dosyası ilk çalıştırmada otomatik olarak inecektir.
model = YOLO("yolov8n-pose.pt")

# 2. Kamerayı başlat
kamera = cv2.VideoCapture(0)

while True:
    basarili_mi, kare = kamera.read()
    if not basarili_mi:
        print("Kameradan görüntü alınamadı!")
        break

    # Kamerayı ayna gibi yatay çevir (Sağ-sol karmaşasını önlemek için)
    kare = cv2.flip(kare, 1)

    # 3. Modeli çalıştır (İnsanları ve iskelet noktalarını bul)
    sonuclar = model(kare, stream=True)

    # 4. Sonuçları ekrana çiz
    for sonuc in sonuclar:
        # YOLO bizim için tüm iskelet bağlantılarını otomatik olarak çizer
        cizilmis_kare = sonuc.plot()

    # 5. Sonucu göster
    cv2.imshow("YOLO Iskelet Takibi", cizilmis_kare)

    # Çıkış kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()