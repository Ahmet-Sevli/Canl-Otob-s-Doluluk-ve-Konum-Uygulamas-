from ultralytics import YOLO
import cv2 as cv
import firebase_admin
from firebase_admin import credentials, db
import os

# dosya yolları her pc de çalışşsın diye
current_dir = os.path.dirname(os.path.abspath(__file__))
firebase_json = os.path.join(current_dir, "yolcu-rtdb-firebase-adminsdk-fbsvc-0187da93a5.json")

# kameradan  alınacak kurulum yapıldıktan sonra video yolu olacak şimdilik demo ile kullandım veri setleri ile eğitilerek gerçek bir ortamda test edilecek

video_path = os.path.join(current_dir, "demo.mp4")


# Firebase başlatmak veritabanının  için json dosyası bi de linki lazım 
cred = credentials.Certificate(firebase_json)  # JSON 
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://yolcu-rtdb-default-rtdb.europe-west1.firebasedatabase.app/'
})

otobus_id = "Arifiye-Kampus"  # Güncel yolcu sayısını tutacağımız otobüs
kapasite=5 # otobüsün kapasitesi şimdilik 5 verildi  çünkü demo videomuzun kısıtı ve kamera açısından dolayı şuan başarımı düşük ama danışman hocamla bir belediyeye bu konuyla ilgili baaşvurup görüntü almaya çalışacaz veri seti için


def firebase_guncelle(otobus_id, binen, inen):
    ref = db.reference(f"/otobusler/{otobus_id}")
    
    # Mevcut toplam yolcu
    toplam_yolcu = ref.child("toplam_yolcu").get() or 0
    toplam_yolcu = binen - inen
    
    # Firebase'e yaz
    ref.set({
        "toplam_yolcu": toplam_yolcu,
        "binen": binen,
        "inen": inen,
        "kapasite":kapasite
    })


# YOLO modelini yükle (insan tespiti için) , yolo8 ile 11 arasında kaldım eğer rasberry pi a geçersem 8 e geri dönerim
model = YOLO("yolo11n.pt")  # COCO dataset, class 0 = insan

# Çizgi koordinatı (y ekseni)
line_y = 250

tolerans=15# şuan özel bi veri seti kullanmadığım için takip ederken frameler arasında insanları kaçırıyor bunu tolare edebilmek için gerekli 

iceridekiler=set() # bu kamera açısına giren ve otobüüsn içinde binmiş olan insanlar
disaridakiler=set() # bu otobüsü dışında ve binme potansiyeli olup kart okutma alanına yaklaşanlar

son_yer={} # içeri girip girmediklerini son konum ve mevcut konum karşılaştırması yaparak bulacaz

# Sayaçlar
binen = 0
inen = 0

# Video capture
cap = cv.VideoCapture(video_path)

genislik = 640
yukseklik = 480 # performans için normal görüntüyü şekillendirmek zorunda kaldım

fps=0 # her frami alıp yük oluşturmak yerine bazı frameleri atlamamı düşündüm

while True:
    ret, frame_ = cap.read()
    if not ret:
        break
    
    if fps%3==0:
        fps+=1
        continue # burada bazı frameleri atlıyorum performans için

    frame= cv.resize(frame_, (genislik, yukseklik))
   
    # YOLO botsort ile takip kısmı->bytetrack daha hızlı ama bot sort deep sort gibi daha güvenli burası değişebilir performansa göre
    results = model.track(source=frame, persist=True, classes=[0], tracker="botsort.yaml" )[0]

    frame=results.plot() # tespitleri çizmek için kontol amaçlı
    
    for r in results.boxes:
        if r.id is None:
            continue  # buradaki kod atlanır
        
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        tid = int(r.id)  # Tracker ID
        #cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1) # merkezlerine göre giriş çıkışları gözle görebilmek için tespit kutularıın ortasına küçük bir nokta
        
        # Son pozisyonu kaydet
        son_yer[tid] = cy

        
        # içeridikileri listeye eklemek lazım ki dışarı çıkış anlarında id lerinden yararlanayım

        if line_y > cy :
           iceridekiler.add(tid)

        # şimdi de dışarıdakileri eklemek lazım içeri giriş anlarında id lerinden yararlanayım
        if line_y <cy:
              disaridakiler.add(tid)    


        # binen kişileri sayacaz bunu da eğer dışarıdakilerden birinin cy si line_ye den küçük olursa demek ki bindi
        # ama onun için de hep en son nerede kaldıklarını bilmek lazım

        for disari in list(disaridakiler):
            if  disari in son_yer and son_yer[disari]< (line_y-tolerans):
                binen+=1
                disaridakiler.remove(disari)  # Çifte saymayı önler
                firebase_guncelle(otobus_id, binen, inen)
        # şimdi de inenler için
        for iceri in list(iceridekiler):
            if  iceri in son_yer and son_yer[iceri] >(line_y+tolerans):
                inen+=1
                iceridekiler.remove(iceri)  # Çifte saymayı önler bunu koymadığım zaman o listeden silinmiyorlar her framede tekrar sayılıyorlar o yüzden kalkması lazım
                firebase_guncelle(otobus_id, binen, inen)

        
        
    # Çizgi ve sayaçları ekrana yazma yeri
    cv.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv.putText(frame, f"Binen: {binen}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f"Inen: {inen}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Otobus Yolcu Sayimi", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

