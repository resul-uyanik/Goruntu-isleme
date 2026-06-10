import cv2
import numpy as np

print("Akıllı Evrak Tarayıcı Başlatılıyor...")
print("Kameraya dikdörtgen bir nesne (kağıt, kitap, kimlik) gösterin.")
print("Tarama yapmak için 's' tuşuna, çıkmak için 'q' tuşuna basın.")
print("NOT: Sistemin kağıdı bulabilmesi için arka planla zıt renkte olması iyi olur (Örn: Koyu masada beyaz kağıt).")

# 1. Eğik 4 köşeyi, düz bir A4 kağıdının köşeleri gibi sıralayan Matematiksel Fonksiyon
def noktalari_sirala(noktalar):
    # Gelen karmaşık 4 noktayı x,y formatına getir
    noktalar = noktalar.reshape((4, 2))
    sirali_noktalar = np.zeros((4, 2), dtype="float32")

    # X ve Y koordinatlarını topla.
    # Toplamı en küçük olan Sol-Üst köşedir, en büyük olan Sağ-Alt köşedir.
    toplam = noktalar.sum(axis=1)
    sirali_noktalar[0] = noktalar[np.argmin(toplam)] # Sol-Üst
    sirali_noktalar[2] = noktalar[np.argmax(toplam)] # Sağ-Alt

    # Y'den X'i çıkar. 
    # Farkı en küçük olan Sağ-Üst köşedir, en büyük olan Sol-Alt köşedir.
    fark = np.diff(noktalar, axis=1)
    sirali_noktalar[1] = noktalar[np.argmin(fark)] # Sağ-Üst
    sirali_noktalar[3] = noktalar[np.argmax(fark)] # Sol-Alt

    return sirali_noktalar

kamera = cv2.VideoCapture(0)

while True:
    basarili_mi, kare = kamera.read()
    if not basarili_mi:
        break

    kopyasi = kare.copy()
    
    # Görüntüyü griye çevir ve bulanıklaştır (Gürültüleri azaltmak için)
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    bulanik = cv2.GaussianBlur(gri, (5, 5), 1)
    
    # Kenarları (Çizgileri) bul
    kenarlar = cv2.Canny(bulanik, 50, 150)
    
    # Kenarları bulunan şekillerin (konturların) listesini çıkar
    konturlar, _ = cv2.findContours(kenarlar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Şekilleri alanlarına göre büyükten küçüğe sırala (Sadece en büyük 5 şekle bakacağız)
    konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)[:5]
    
    belge_koordinatlari = None

    for kontur in konturlar:
        cevre = cv2.arcLength(kontur, True)
        # Şeklin kıvrımlarını düzeltip, onu basit bir çokgene benzetiyoruz
        yaklasik_cokgen = cv2.approxPolyDP(kontur, 0.02 * cevre, True)
        
        # Eğer en büyük şekil tam olarak 4 köşeye (kenara) sahipse, evrakı bulduk demektir!
        if len(yaklasik_cokgen) == 4:
            belge_koordinatlari = yaklasik_cokgen
            break

    # Eğer 4 köşeli bir belge bulduysa etrafını yeşil kalemle çiz
    if belge_koordinatlari is not None:
        cv2.drawContours(kare, [belge_koordinatlari], -1, (0, 255, 0), 3)
        cv2.putText(kare, "BELGE BULUNDU! Taramak icin 'S'ye basin", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(kare, "Belge Araniyor...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Kamera Gorus Acisi", kare)

    tus = cv2.waitKey(1) & 0xFF
    
    # EĞER 'S' (Scan) TUŞUNA BASILDIYSA VE BELGE BULUNDUYSA:
    if tus == ord('s') and belge_koordinatlari is not None:
        print("Tarama başlatıldı...")
        
        # 4 köşeyi bizim fonksiyonumuzla düzgün sıraya koy
        sirali_koseler = noktalari_sirala(belge_koordinatlari)
        
        # Sündüreceğimiz düz A4 kağıdının sanal boyutlarını belirliyoruz (Örn: 400x600 piksel)
        hedef_noktalar = np.array([
            [0, 0],
            [400, 0],
            [400, 600],
            [0, 600]
        ], dtype="float32")
        
        # İŞİN SİHRİ: Eğik noktalar ile Düz noktalar arasındaki "Dönüşüm Matrisini" hesapla
        matris = cv2.getPerspectiveTransform(sirali_koseler, hedef_noktalar)
        
        # Orijinal görüntüyü bu matrise göre havaya kaldırıp sündür (Perspective Warp)
        taranmis_renkli = cv2.warpPerspective(kopyasi, matris, (400, 600))
        
        # FOTOKOPİ FİLTRESİ (Siyah-Beyaz ve Keskin)
        taranmis_gri = cv2.cvtColor(taranmis_renkli, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Threshold: Ortam ışığına aldırış etmeden kağıdı bembeyaz, yazıları simsiyah yapar
        taranmis_temiz = cv2.adaptiveThreshold(taranmis_gri, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
        
        # Sonuçları ekranda göster
        cv2.imshow("1. Orijinal Kirpilmis", taranmis_renkli)
        cv2.imshow("2. CamScanner Filtresi", taranmis_temiz)
        
        # Bilgisayara kaydet
        cv2.imwrite("taranmis_belge.jpg", taranmis_temiz)
        print("Belge başarıyla tarandı ve 'taranmis_belge.jpg' olarak kaydedildi!")

    elif tus == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()