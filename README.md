\# 👁️ Python ile Görüntü İşleme ve Nesne Tanıma Projesi



Bu proje; bilgisayarlı görü (computer vision) ve yapay zeka teknolojileri kullanılarak geliştirilmiş, gerçek zamanlı nesne tespiti, yüz bulma, iskelet/duruş takibi ve akıllı alarm sistemleri içeren kapsamlı bir görüntü işleme havuzudur.



\## 🤖 Yapay Zeka Entegrasyonu

Bu projedeki algılama, takip ve analiz süreçlerinin büyük bir bölümü, derin öğrenme tabanlı \*\*Yapay Zeka (AI)\*\* modelleri kullanılarak gerçekleştirilmiştir. Özellikle nesne algılama ve duruş (pose) kestirimi görevlerinde modern ve güçlü \*\*YOLOv8\*\* mimarisinden yararlanılmıştır.



\---



\## 🚀 Öne Çıkan Özellikler ve Modüller



Proje, farklı senaryolara hizmet eden bağımsız Python betiklerinden oluşmaktadır:



\* \*\*Akıllı Alarm Sistemi (`alarm\_sistemi.py`):\*\* Kamera odağına giren belirli hedef nesneleri (örneğin cep telefonu) yapay zeka ile tespit ederek ekran üzerinde anlık görsel uyarı/alarm tetikler.

\* \*\*Gerçek Zamanlı Nesne Tanıma (`canli\_yolo.py` \& `yolo\_nesne\_bulucu.py`):\*\* Hem statik görseller hem de canlı kamera görüntüsü üzerinde yüzlerce farklı nesne sınıfını yüksek doğrulukla, anlık ve Türkçe etiketlerle tanır.

\* \*\*YOLOv8 İskelet Takibi (`el\_takibi.py`):\*\* YOLOv8-pose modelini kullanarak insan vücudunu ve eklem noktalarını (iskelet yapısını) canlı olarak takip eder.

\* \*\*Yüz Algılama (`yuz\_bulucu.py` \& `canli\_yuz\_bulucu.py`):\*\* OpenCV'nin Haar-Cascade yöntemiyle görüntü veya canlı video akışı üzerindeki insan yüzlerini tespit ederek yeşil çerçeve içine alır.

\* \*\*Temel Görüntü İşleme Aşamaları (`main.py`):\*\* Bir resmin gri tonlamaya çevrilmesi, Gaussian Blur ile bulanıklaştırılması ve Canny algoritmasıyla kenarlarının tespit edilmesi gibi temel OpenCV filtreleme adımlarını gösterir.



\---



\## 🛠️ Kullanılan Teknolojiler



\* \*\*Programlama Dili:\*\* Python 3.x

\* \*\*Yapay Zeka \& Derin Öğrenme:\*\* Ultralytics YOLOv8 (yolov8n.pt, yolov8n-pose.pt)

\* \*\*Görüntü İşleme Kütüphanesi:\*\* OpenCV (Open Source Computer Vision Library)



\---



\## 📦 Kurulum ve Çalıştırma



Projenizi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:



1\. \*\*Projeyi Klonlayın:\*\*

&#x20;  ```bash

&#x20;  git clone \[https://github.com/resul-uyanik/Goruntu-isleme.git](https://github.com/resul-uyanik/Goruntu-isleme.git)

&#x20;  cd Goruntu-isleme

2\. \*\*Gerekli Kütüphaneleri Yükleyin:\*\*

&#x20;  ```bash

&#x20;  pip install opencv-python ultralytics



💡 Not: .pt uzantılı yapay zeka model dosyaları (YOLOv8 ağırlıkları), depo boyutunu optimize etmek amacıyla GitHub'a yüklenmemiştir. Kod ilk kez çalıştırıldığında ilgili ağırlıklar ultralytics kütüphanesi tarafından otomatik olarak internetten indirilecektir.

