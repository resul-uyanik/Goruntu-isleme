import cv2
import numpy as np

print("Optik form çiziliyor...")

# 1. Bembeyaz, temiz bir dijital A4 kağıdı oluştur (Genişlik: 400, Yükseklik: 600)
kagit = np.ones((600, 400, 3), dtype="uint8") * 255

# 2. Önceki kodumuzdaki cevap anahtarına uygun olarak işaretlenecek şıkları belirliyoruz
# 1.Soru: B(1), 2.Soru: C(2), 3.Soru: A(0), 4.Soru: B(1), 5.Soru: D(3)
isaretli_siklar = [1, 2, 0, 1, 3]

# Çemberlerin konumu için başlangıç ayarları
baslangic_x = 80
baslangic_y = 100
bosluk_x = 70
bosluk_y = 90
yaricap = 20

# 3. 5 Soru ve her biri için 4 şık (A, B, C, D) çiz
for soru in range(5):
    # Soru numarasını kağıda yaz (1., 2., 3. vb.)
    cv2.putText(kagit, f"{soru+1}.", (20, baslangic_y + (soru * bosluk_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    for sik in range(4):
        merkez_x = baslangic_x + (sik * bosluk_x)
        merkez_y = baslangic_y + (soru * bosluk_y)
        
        # Eğer sıradaki şık, işaretlenmesi gereken şıksa içini tamamen siyah (-1) doldur
        if sik == isaretli_siklar[soru]:
            cv2.circle(kagit, (merkez_x, merkez_y), yaricap, (0, 0, 0), -1)
        # Değilse sadece dış çerçevesini çiz (Kalınlık: 3)
        else:
            cv2.circle(kagit, (merkez_x, merkez_y), yaricap, (0, 0, 0), 3)

# 4. Çizilen kağıdı bilgisayara "sinav.png" adıyla kaydet
cv2.imwrite("sinav.png", kagit)
print("İşlem tamam! 'sinav.png' dosyası klasörüne başarıyla kaydedildi.")

# Çizdiği resmi sana da göstersin
cv2.imshow("Olusturulan Optik Form", kagit)
cv2.waitKey(0)
cv2.destroyAllWindows()