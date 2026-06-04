import cv2

resim = cv2.imread("test.jpg")

if resim is None:
    print("Hata: Resim bulunamadı!")
else:
    # 1. Gri Tonlamaya Çevirme (Siyah-Beyaz)
    # Neden yapıyoruz? Renkler algoritmayı yorar. Şekil ve kenar bulmak için renk bilgisine ihtiyacımız yok.
    gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

    # 2. Bulanıklaştırma (Gaussian Blur)
    # Neden yapıyoruz? Fotoğraftaki ufak pürüzleri ve kumlanmaları (noise) yok ederek net kenarlar elde etmek için.
    bulanik_resim = cv2.GaussianBlur(gri_resim, (7, 7), 0)

    # 3. Kenar Algılama (Canny Edge Detection)
    # Neden yapıyoruz? Resimdeki nesnelerin dış hatlarını bir kalemle çizilmiş gibi belirginleştirmek için.
    # (50 ve 150 eşik değerleridir, detayları ne kadar çizeceğini belirler)
    kenar_resim = cv2.Canny(bulanik_resim, 50, 150)

    # İşlenmiş tüm aşamaları ayrı pencerelerde görmek için ekrana yazdırıyoruz
    cv2.imshow("1 - Orijinal", resim)
    cv2.imshow("2 - Gri Tonlama", gri_resim)
    cv2.imshow("3 - Bulaniklasma", bulanik_resim)
    cv2.imshow("4 - Kenarlar", kenar_resim)

    # İşlenmiş (kenar algılanmış) resmi klasöre yeni bir dosya olarak kaydet
    cv2.imwrite("benim_cizimim.jpg", kenar_resim)
    print("Tebrikler, işlenmiş resim klasöre başarıyla kaydedildi!")

    # Kapatmak için klavyeden bir tuşa basılmasını bekle
    cv2.waitKey(0)
    cv2.destroyAllWindows()