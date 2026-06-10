from ultralytics import YOLO
import cv2
import numpy as np

print("YOLO Segmentasyon modeli yükleniyor... Lütfen bekleyin.")

model = YOLO("yolov8n-seg.pt")

# --- YENİ EKLENEN KISIM: ARKA PLANI YÜKLEME ---
# İndirdiğimiz fotoğrafı Python'a okutuyoruz
yeni_arka_plan = cv2.imread("manzara.jpg")

# Eğer fotoğraf bulunamazsa programı durdurup uyaralım
if yeni_arka_plan is None:
    print("HATA: 'manzara.jpg' klasörde bulunamadı. Lütfen resmin adını ve uzantısını kontrol edin!")
    exit()

kamera = cv2.VideoCapture(0)

while True:
    basarili_mi, kare = kamera.read()
    if not basarili_mi:
        break

    kare = cv2.flip(kare, 1)

    # Kameramızın o anki çözünürlüğünü (Genişlik ve Yükseklik) alıyoruz
    kamera_genislik = kare.shape[1]
    kamera_yukseklik = kare.shape[0]
    kamera_boyutu = (kamera_genislik, kamera_yukseklik)

    # --- KRİTİK İŞLEM ---
    # İndirdiğimiz manzara resmini kameramızın boyutuyla BİREBİR aynı olacak şekilde yeniden boyutlandırıyoruz.
    ayarlanmis_arka_plan = cv2.resize(yeni_arka_plan, kamera_boyutu)

    # Modeli çalıştır
    sonuclar = model(kare, stream=True)
    son_goruntu = kare.copy()

    for sonuc in sonuclar:
        if sonuc.masks is not None:
            maskeler = sonuc.masks.data.cpu().numpy()
            siniflar = sonuc.boxes.cls.cpu().numpy()

            insan_maskesi = np.zeros((kamera_yukseklik, kamera_genislik), dtype=np.uint8)

            for i, sinif_id in enumerate(siniflar):
                # Sınıf 0 = İnsan
                if int(sinif_id) == 0:
                    m = cv2.resize(maskeler[i], kamera_boyutu)
                    insan_maskesi = cv2.bitwise_or(insan_maskesi, (m * 255).astype(np.uint8))
            
            arka_plan_maskesi = cv2.bitwise_not(insan_maskesi)

            # İnsanı orijinal kameradan kes
            insan_kismi = cv2.bitwise_and(kare, kare, mask=insan_maskesi)
            
            # Arka planı bulanık ekrandan değil, kendi manzara resmimizden kesiyoruz!
            arka_plan_kismi = cv2.bitwise_and(ayarlanmis_arka_plan, ayarlanmis_arka_plan, mask=arka_plan_maskesi)

            # Kesilen iki parçayı birleştir
            son_goruntu = cv2.add(insan_kismi, arka_plan_kismi)

    cv2.imshow("Yayinci Yesil Perdesi - Ozel Arka Plan", son_goruntu)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()