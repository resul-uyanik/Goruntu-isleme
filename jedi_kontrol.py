import cv2
import numpy as np
import pyautogui

print("Jedi Zihin Kontrolü Başlatılıyor...")
print("Kameraya YEŞİL bir obje gösterin.")
print("Objeyi ekranın SAĞINA veya SOLUNA çekerek bilgisayarı yönetin.")
print("Çıkmak için 'q' tuşuna basın.")

# Fareyi köşelere götürdüğümüzde PyAutoGUI'nin güvenlik gereği çökmesini engeller
pyautogui.FAILSAFE = False 

kamera = cv2.VideoCapture(0)

# Sistem başlangıçta objenin ortada olduğunu varsayar
durum = "ORTADA"

while True:
    basarili_mi, kare = kamera.read()
    if not basarili_mi:
        break

    # Kamerayı ayna gibi yatay çevir
    kare = cv2.flip(kare, 1)
    
    kamera_genislik = kare.shape[1]
    
    # Ekranı 3 bölgeye ayıran sanal sınırlar (Sol sınır: 200. piksel, Sağ sınır: 440. piksel)
    sol_sinir = 200
    sag_sinir = 440

    # Kullanıcının sınırları görmesi için ekrana iki tane beyaz çizgi çekiyoruz
    cv2.line(kare, (sol_sinir, 0), (sol_sinir, 480), (255, 255, 255), 2)
    cv2.line(kare, (sag_sinir, 0), (sag_sinir, 480), (255, 255, 255), 2)

    # Yazıları ekrana ekle
    cv2.putText(kare, "GERI (SOL)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(kare, "ILERI (SAG)", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Yeşil renk algılama ayarları (En kararlı versiyon)
    hsv_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
    alt_yesil = np.array([40, 100, 100])
    ust_yesil = np.array([80, 255, 255])
    maske = cv2.inRange(hsv_kare, alt_yesil, ust_yesil)
    maske = cv2.erode(maske, None, iterations=2)
    maske = cv2.dilate(maske, None, iterations=2)

    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(konturlar) > 0:
        en_buyuk_kontur = max(konturlar, key=cv2.contourArea)

        if cv2.contourArea(en_buyuk_kontur) > 500:
            ((x, y), yari_cap) = cv2.minEnclosingCircle(en_buyuk_kontur)
            merkez_x = int(x)
            merkez_y = int(y)

            # Objenin etrafına sarı bir takip yuvarlağı çiz
            cv2.circle(kare, (merkez_x, merkez_y), int(yari_cap), (0, 255, 255), 3)

            # --- JEDI KONTROL MANTIĞI ---
            
            # Eğer obje SOL çizginin ötesine geçerse ve en son ORTADA ise:
            if merkez_x < sol_sinir and durum == "ORTADA":
                print("Sola Kaydırıldı! (Önceki Şarkı / Önceki Fotoğraf)")
                
                # Klavyeden sanal olarak "Sol Ok" tuşuna bas
                pyautogui.press('left') 
                
                # Şarkı değiştirmek istersen üsttekini silip şunu kullanabilirsin:
                # pyautogui.press('prevtrack') 
                
                durum = "SOLDA" # Durumu güncelle ki durmadan tuşa basmasın
                cv2.putText(kare, "<<< ISLEM YAPILDI", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Eğer obje SAĞ çizginin ötesine geçerse ve en son ORTADA ise:
            elif merkez_x > sag_sinir and durum == "ORTADA":
                print("Sağa Kaydırıldı! (Sonraki Şarkı / Sonraki Fotoğraf)")
                
                # Klavyeden sanal olarak "Sağ Ok" tuşuna bas
                pyautogui.press('right')
                
                # Şarkı değiştirmek istersen üsttekini silip şunu kullanabilirsin:
                # pyautogui.press('nexttrack')
                
                durum = "SAGDA"
                cv2.putText(kare, "ISLEM YAPILDI >>>", (350, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Eğer obje iki çizginin ARASINDAYSA sistemi yeni komut için "kur"
            elif sol_sinir < merkez_x < sag_sinir:
                durum = "ORTADA"
                cv2.circle(kare, (merkez_x, merkez_y), 10, (0, 255, 0), -1) # Ortadayken yeşil nokta yak

    cv2.imshow("Jedi Zihin Kontrolu", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()