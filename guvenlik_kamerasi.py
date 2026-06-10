import cv2
import time

print("Güvenlik Kamerası Başlatılıyor...")
print("ÖNEMLİ: Odanın boş halini kaydetmek için lütfen 3 saniye kameranın önünden çekilin.")
print("Sistemi sıfırlamak için 'r', çıkmak için 'q' tuşuna basın.")

kamera = cv2.VideoCapture(0)

# Kameranın ışığa alışması için 3 saniye bekleme süresi veriyoruz
time.sleep(3) 

referans_kare = None

while True:
    basarili_mi, kare = kamera.read()
    if not basarili_mi:
        break

    # Doğal bir görüntü için kamerayı yatay çevir
    kare = cv2.flip(kare, 1)

    # 1. Görüntüyü Siyah-Beyaza çevir ve yoğun bir şekilde bulanıklaştır
    # (Bulanıklaştırmak, ufak ışık değişimlerinin sahte alarm vermesini engeller)
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    bulanik = cv2.GaussianBlur(gri, (21, 21), 0)

    # 2. İLK KAREYİ HAFIZAYA AL (Odanın Boş Hali)
    if referans_kare is None:
        referans_kare = bulanik
        print("Referans (boş oda) kaydedildi! Alarm sistemi AKTİF.")
        continue # İlk turu bitir ve başa dön

    # 3. İŞİN MATEMATİĞİ: Odanın boş hali ile şu anki hali arasındaki farkı bul
    fark = cv2.absdiff(referans_kare, bulanik)

    # 4. Farklılıkları belirginleştir (Beyaz=Hareket, Siyah=Sabit)
    # 25 hassasiyet değeridir. Odanın ışığına göre bunu 15 ile 50 arası değiştirebilirsin.
    _, threshold = cv2.threshold(fark, 25, 255, cv2.THRESH_BINARY)
    threshold = cv2.dilate(threshold, None, iterations=2)

    # 5. Hareket eden bölgelerin (beyaz lekelerin) sınırlarını bul
    konturlar, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hareket_var = False

    for kontur in konturlar:
        # Alanı 5000 pikselden küçük olan ufak hareketleri (örneğin uçan bir sineği) yoksay
        if cv2.contourArea(kontur) < 5000:
            continue

        hareket_var = True
        
        # Hareketin olduğu bölgeyi hesapla ve etrafına kutu çiz
        (x, y, w, h) = cv2.boundingRect(kontur)
        cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # 6. EKRAN BİLGİLENDİRMESİ VE ALARM
    if hareket_var:
        cv2.putText(kare, "TEHLIKE: HAREKET ALGILANDI!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # Bütün ekranın etrafına kalın kırmızı bir alarm çerçevesi çiz
        cv2.rectangle(kare, (0, 0), (kare.shape[1], kare.shape[0]), (0, 0, 255), 10)
    else:
        cv2.putText(kare, "Sistem Guvende", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Sonuçları ekranda göster
    cv2.imshow("1. Ana Guvenlik Kamerasi", kare)
    
    # İşin mutfağını (matematiğini) görmen için hareket algılama ekranını da açıyoruz
    cv2.imshow("2. Bilgisayarin Gorus Acisi (Hareketler)", threshold)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('q'):
        break
    elif tus == ord('r'): 
        # Eğer odanın ışığı değişirse veya sandalyenin yerini değiştirirsen 
        # 'r' tuşuna basarak referans görüntüyü (boş odayı) güncelleyebilirsin.
        referans_kare = bulanik
        print("Sistem Sıfırlandı! Yeni referans alındı.")

kamera.release()
cv2.destroyAllWindows()