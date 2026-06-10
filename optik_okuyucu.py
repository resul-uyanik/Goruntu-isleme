import cv2
import numpy as np

print("Optik Okuyucu başlatılıyor...")

# 1. Paint'te çizdiğimiz sınav kağıdını oku
resim = cv2.imread("sinav.png")

if resim is None:
    print("Hata: 'sinav.png' bulunamadı! Lütfen Paint'te çizip klasöre kaydettiğinden emin ol.")
    exit()

# 2. Görüntüyü Siyah-Beyaz Yap ve Tersine Çevir
# Görüntü işlemede şekilleri bulmak için arka planın siyah, şekillerin beyaz olması işleri kolaylaştırır.
gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
bulanik = cv2.GaussianBlur(gri, (5, 5), 0)
# THRESH_BINARY_INV: Beyaz kağıdı siyah, siyah kalem izlerini beyaz yapar.
_, threshold = cv2.threshold(bulanik, 150, 255, cv2.THRESH_BINARY_INV)

# 3. Konturları (Şekilleri) Bul
konturlar, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

baloncuklar = []

# Sadece "Çember" olan şekilleri filtrele
for k in konturlar:
    # Şeklin etrafına hayali bir dikdörtgen çiz
    (x, y, g, yuks) = cv2.boundingRect(k)
    oran = g / float(yuks)
    
    # Eğer genişlik ve yükseklik birbirine yakınsa (oran 1'e yakınsa) ve çok küçük değilse bu bir şıktır
    if 0.8 <= oran <= 1.2 and g >= 20 and yuks >= 20:
        baloncuklar.append(k)

print(f"Ekranda toplam {len(baloncuklar)} adet şık (çember) bulundu.")

# Eğer tam 20 çember (5 soru x 4 şık) bulamadıysa Paint çiziminde bir hata vardır
if len(baloncuklar) != 20:
    print("HATA: Sistem tam olarak 20 adet şık bulamadı. Paint'teki çemberlerin birbirine değmediğinden emin ol!")
    exit()

# 4. Şıkları Yukarıdan Aşağıya (Sorulara Göre) Sırala
# Y (dikey) koordinatına göre küçükten büyüğe sıralıyoruz
baloncuklar = sorted(baloncuklar, key=lambda b: cv2.boundingRect(b)[1])

# 5. Cevap Anahtarı (0=A, 1=B, 2=C, 3=D)
# Örnek: 1. Soru B, 2. Soru C, 3. Soru A, 4. Soru B, 5. Soru D
cevap_anahtari = {0: 1, 1: 2, 2: 0, 3: 1, 4: 3}
dogru_sayisi = 0

# 6. Her Soruyu Tek Tek Değerlendir
for soru_no in range(5):
    # O soruya ait 4 şıkkı (A, B, C, D) al ve bu kez X (yatay) ekseninde soldan sağa sırala
    sirali_siklar = sorted(baloncuklar[soru_no*4 : (soru_no+1)*4], key=lambda b: cv2.boundingRect(b)[0])

    isaretli_sik = None
    maksimum_siyah_piksel = 0

    # O sorunun 4 şıkkı içinde gezin
    for i, sik in enumerate(sirali_siklar):
        # Sadece bu şıkkın içinin göründüğü simsiyah bir maske oluştur
        maske = np.zeros(gri.shape, dtype="uint8")
        cv2.drawContours(maske, [sik], -1, 255, -1)

        # Bu maskenin içine düşen beyaz (aslında kurşun kalem) pikselleri say
        piksel_sayisi = cv2.countNonZero(cv2.bitwise_and(threshold, threshold, mask=maske))

        # Eğer bu şıkkın içindeki piksel sayısı şu ana kadar gördüğümüz en büyükse, işaretlenen şık budur
        if piksel_sayisi > maksimum_siyah_piksel:
            maksimum_siyah_piksel = piksel_sayisi
            isaretli_sik = i

    # 7. Öğrencinin İşaretlediği Şıkkı Kontrol Et
    dogru_cevap = cevap_anahtari[soru_no]
    renk = (0, 0, 255) # Varsayılan renk Kırmızı (Yanlış)

    if isaretli_sik == dogru_cevap:
        renk = (0, 255, 0) # Eğer cevap doğruysa Yeşil yap
        dogru_sayisi += 1

    # İşaretlenen şıkkın etrafını doğru/yanlış rengiyle kalınca çiz
    cv2.drawContours(resim, [sirali_siklar[isaretli_sik]], -1, renk, 4)

# 8. Toplam Puanı Hesapla ve Ekrana Yazdır
puan = (dogru_sayisi / 5.0) * 100
cv2.putText(resim, f"SINAV PUANI: {puan}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

# Sonucu Göster
cv2.imshow("Akilli Optik Okuyucu", resim)
cv2.waitKey(0)
cv2.destroyAllWindows()