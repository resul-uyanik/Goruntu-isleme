import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog

print("Akıllı Evrak Tarayıcı Başlatılıyor...")

# 1. 4 köşeyi sıralayan matematiksel fonksiyon
def noktalari_sirala(noktalar):
    noktalar = noktalar.reshape((4, 2))
    sirali_noktalar = np.zeros((4, 2), dtype="float32")
    toplam = noktalar.sum(axis=1)
    sirali_noktalar[0] = noktalar[np.argmin(toplam)] # Sol-Üst
    sirali_noktalar[2] = noktalar[np.argmax(toplam)] # Sağ-Alt
    fark = np.diff(noktalar, axis=1)
    sirali_noktalar[1] = noktalar[np.argmin(fark)] # Sağ-Üst
    sirali_noktalar[3] = noktalar[np.argmax(fark)] # Sol-Alt
    return sirali_noktalar

# --- KULLANICI GİRİŞİ: GÖRSEL ARAYÜZ İLE DOSYA SEÇME ---
# Gereksiz boş bir pencere açılmasını engellemek için ana pencereyi gizliyoruz
root = tk.Tk()
root.withdraw()

# İşletim sisteminin orijinal "Dosya Aç" penceresini çağırıyoruz
dosya_yolu = filedialog.askopenfilename(
    title="Taranacak Fotoğrafı Seçin",
    filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")]
)

# Eğer kullanıcı pencereyi kapatırsa veya 'İptal'e basarsa sistemi güvenlice kapat
if not dosya_yolu:
    print("İşlem iptal edildi. Herhangi bir dosya seçilmedi.")
    sys.exit()

print(f"Seçilen Dosya: {dosya_yolu}")
print("İşleniyor, lütfen bekleyin...")

# 2. Fotoğrafı seçilen yoldan oku
orijinal_foto = cv2.imread(dosya_yolu)

if orijinal_foto is None:
    print("HATA: Fotoğraf okunamadı! Lütfen geçerli bir resim dosyası seçtiğinizden emin olun.")
    sys.exit()

kopyasi = orijinal_foto.copy()
gosterim_icin = orijinal_foto.copy()

# 3. Belgeyi Bulma Adımları
gri = cv2.cvtColor(kopyasi, cv2.COLOR_BGR2GRAY)
bulanik = cv2.GaussianBlur(gri, (5, 5), 1)
kenarlar = cv2.Canny(bulanik, 75, 200)

konturlar, _ = cv2.findContours(kenarlar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)[:5]

belge_koordinatlari = None

for kontur in konturlar:
    cevre = cv2.arcLength(kontur, True)
    yaklasik_cokgen = cv2.approxPolyDP(kontur, 0.02 * cevre, True)
    
    if len(yaklasik_cokgen) == 4:
        belge_koordinatlari = yaklasik_cokgen
        break

if belge_koordinatlari is not None:
    cv2.drawContours(gosterim_icin, [belge_koordinatlari], -1, (0, 255, 0), 3)
else:
    print("HATA: Fotoğrafta 4 köşeli belirgin bir evrak algılanamadı.")
    cv2.imshow("Algilama Hatasi", gosterim_icin)
    cv2.waitKey(0)
    sys.exit()

# 4. Perspektif Çarpıtma (Sündürme) Matematiği
sirali_koseler = noktalari_sirala(belge_koordinatlari)

hedef_noktalar = np.array([
    [0, 0],
    [420, 0],
    [420, 594],
    [0, 594]
], dtype="float32")

matris = cv2.getPerspectiveTransform(sirali_koseler, hedef_noktalar)
taranmis_renkli = cv2.warpPerspective(kopyasi, matris, (420, 594))

# 5. Fotokopi Filtresi
taranmis_gri = cv2.cvtColor(taranmis_renkli, cv2.COLOR_BGR2GRAY)
taranmis_temiz = cv2.adaptiveThreshold(taranmis_gri, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)

# --- SONUÇLARI GÖSTER VE KAYDET ---
print("İşlem tamamlandı! Sonuç ekranları açıldı.")

cv2.namedWindow("1. Orijinal ve Algilanan", cv2.WINDOW_NORMAL)
cv2.namedWindow("2. Renkli Tarama", cv2.WINDOW_NORMAL)
cv2.namedWindow("3. Fotokopi Filtresi", cv2.WINDOW_NORMAL)

cv2.imshow("1. Orijinal ve Algilanan", gosterim_icin)
cv2.imshow("2. Renkli Tarama", taranmis_renkli)
cv2.imshow("3. Fotokopi Filtresi", taranmis_temiz)

cv2.imwrite("sonuc_taranmis_renkli.jpg", taranmis_renkli)
cv2.imwrite("sonuc_taranmis_temiz.jpg", taranmis_temiz)

cv2.waitKey(0)
cv2.destroyAllWindows()