# ai_server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, ExifTags
import io
import math
import os
import traceback
import boto3
import gdown
import gc
import tensorflow.keras.backend as K
import logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
# 🔐 Sadece belirli frontend origin'lerine izin veriyoruz
CORS(app, resources={r"/*": {"origins": [
    "https://burn-application-frontend.vercel.app",
    "https://burn-application-frontend.vercel.app/",
    "https://burn-application-frontend-git-main-zeynepgamzetopays-projects.vercel.app",
    "https://burn-application-frontend-git-main-zeynepgamzetopays-projects.vercel.app/",
    "https://burn-application-frontend-zeynepgamzetopays-projects.vercel.app",
    "https://burn-application-frontend-zeynepgamzetopays-projects.vercel.app/",
    "https://burn-application-frontend-rlqzzskn4-zeynepgamzetopays-projects.vercel.app",
    "https://burn-application-frontend-rlqzzskn4-zeynepgamzetopays-projects.vercel.app/"
]}})
#CORS(app)  # En basit ve fetch kodunla uyumlu hali
#CORS(app, resources={r"/predict": {"origins": "https://burn-application-frontend.vercel.app"}}, supports_credentials=True)
#CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)


# 🧠 Model global değişkenleri
depth_model = None
segmentation_model = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
class_labels_depth = [ "First_degree","Second_degree_superficial","Second_degree_deep","Third_degree"]


def download_model_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Google Drive'dan indirildi: {output_path}")
    except Exception as e:
        print(f"❌ Google Drive'dan indirme hatası: {e}")

# 📦 Google Drive File IDs
BEST_MODEL_GDRIVE_ID = "10vn10MaOKvNM7qoPoRMqql7aP8JPnIAh"   # best_model.h5 ID'si
SEG_MODEL_GDRIVE_ID = "1WnazvELJLG53Pdim1y1aJwxfxFupgWcL"   # segmentation_model.h5 ID'si

def load_ai_models():
    global depth_model, segmentation_model
    print("Klasör içeriği:", os.listdir(SCRIPT_DIR))

    try:
        depth_model_path = os.path.join(SCRIPT_DIR, "best_model.h5")
        if not os.path.exists(depth_model_path):
            print("📥 Derinlik modeli bulunamadı, Google Drive'dan indiriliyor...")
            download_model_from_drive(BEST_MODEL_GDRIVE_ID, depth_model_path)
        depth_model = tf.keras.models.load_model(depth_model_path, compile=False)
        print("✅ Derinlik modeli yüklendi.")
    except Exception as e:
        print(f"❌ Derinlik modeli yüklenirken hata: {e}")
        traceback.print_exc()

    # try:
    #     seg_model_path = os.path.join(SCRIPT_DIR, "segmentation_model.h5")
    #     if not os.path.exists(seg_model_path):
    #         print("📥 Segmentasyon modeli bulunamadı, Google Drive'dan indiriliyor...")
    #         download_model_from_drive(SEG_MODEL_GDRIVE_ID, seg_model_path)
    #     segmentation_model = tf.keras.models.load_model(seg_model_path, compile=False)
    #     print("✅ Segmentasyon modeli yüklendi.")
    # except Exception as e:
    #     print(f"❌ Segmentasyon modeli yüklenirken hata: {e}")
    #     traceback.print_exc()

# HEMEN ALTINA çağır:
load_ai_models()

# # 📦 S3 bilgileri
# S3_BUCKET = "burn-models-bucket"  # Kendi bucket adınla değiştir
# BEST_MODEL_S3_KEY = "best_model.h5"
# SEG_MODEL_S3_KEY = "segmentation_model.h5"

# # 🔽 S3'ten model indirme fonksiyonu
# def download_model_from_s3(bucket_name, s3_key, local_path):
#     s3 = boto3.client("s3")
#     try:
#         s3.download_file(bucket_name, s3_key, local_path)
#         print(f"✅ S3'ten indirildi: {s3_key}")
#     except Exception as e:
#         print(f"❌ S3'ten indirme hatası ({s3_key}): {e}")

# # 🚀 Model yükleyici
# def load_ai_models():
#     global depth_model, segmentation_model

#     try:
#         depth_model_path = os.path.join(SCRIPT_DIR, "best_model.h5")
#         if not os.path.exists(depth_model_path):
#             print("📥 Derinlik modeli bulunamadı, S3'ten indiriliyor...")
#             download_model_from_s3(S3_BUCKET, BEST_MODEL_S3_KEY, depth_model_path)
#         depth_model = tf.keras.models.load_model(depth_model_path, compile=False)
#         print("✅ Derinlik modeli yüklendi.")
#     except Exception as e:
#         print(f"❌ Derinlik modeli yüklenirken hata: {e}")
#         traceback.print_exc()

#     try:
#         seg_model_path = os.path.join(SCRIPT_DIR, "segmentation_model.h5")
#         if not os.path.exists(seg_model_path):
#             print("📥 Segmentasyon modeli bulunamadı, S3'ten indiriliyor...")
#             download_model_from_s3(S3_BUCKET, SEG_MODEL_S3_KEY, seg_model_path)
#         segmentation_model = tf.keras.models.load_model(seg_model_path, compile=False)
#         print("✅ Segmentasyon modeli yüklendi.")
#     except Exception as e:
#         print(f"❌ Segmentasyon modeli yüklenirken hata: {e}")
#         traceback.print_exc()

# Resimden DPI bilgisini okuma fonksiyonu
def get_image_dpi(image_pil_object):
    dpi_val = image_pil_object.info.get('dpi')
    if dpi_val:
        # DPI (yatay, dikey) tuple olabilir, genellikle aynıdır
        return float(dpi_val[0]) if isinstance(dpi_val, (tuple, list)) else float(dpi_val)

    # JPEG için EXIF kontrolü
    try:
        exif_data = image_pil_object.getexif() # _getexif() yerine getexif() daha güncel
        if exif_data:
            # EXIF tag'lerini isimlere çevir (Pillow 9.0.0 ve sonrası için direkt tag ID kullanılabilir)
            # Basitlik için direkt ID'leri kullanalım (XResolution: 282, YResolution: 283, ResolutionUnit: 296)
            x_res_tag = 282
            res_unit_tag = 296

            x_res_value = exif_data.get(x_res_tag)
            res_unit_value = exif_data.get(res_unit_tag)

            if x_res_value and res_unit_value:
                # XResolution genellikle (pay, payda) şeklinde bir Rational nesnesidir
                if hasattr(x_res_value, 'numerator') and hasattr(x_res_value, 'denominator'):
                    val_res = float(x_res_value.numerator) / float(x_res_value.denominator) if x_res_value.denominator != 0 else float(x_res_value.numerator)
                else: # Bazen direkt float gelebilir
                    val_res = float(x_res_value)

                if res_unit_value == 2: # Inç
                    return val_res
                elif res_unit_value == 3: # Santimetre (PPC -> PPI)
                    return val_res * 2.54
    except Exception as e:
        print(f"EXIF okunurken hata (normal olabilir): {e}")
        pass # EXIF okuma başarısız olursa sorun değil

    print("Uyarı: Resimden DPI bilgisi okunamadı, varsayılan 96 DPI kullanılacak.")
    return 96.0

# GÖRSEL ÖN İŞLEME FONKSİYONLARI
# BU FONKSİYONLARI KENDİ MODELLERİNİZİN GEREKSİNİMLERİNE GÖRE AYARLAYIN!
def preprocess_image_for_depth(pil_image):
    # Derinlik modelinin beklediği boyut (örneğin 640x640, 128x128)
    image = pil_image.convert("RGB").resize((128,128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0) # Batch boyutu ekle [1, H, W, C]

def preprocess_image_for_segmentation(pil_image, target_height=256, target_width=256):
    # Segmentasyon modelinin beklediği boyut ve ön işleme
    # target_height ve target_width, modelin input_shape'inden alınabilir.
    image = pil_image.convert("RGB").resize((target_width, target_height))
    image_array = np.array(image) / 255.0 # Yaygın bir normalizasyon
    return np.expand_dims(image_array, axis=0) # Batch boyutu ekle

# HESAPLAMA FONKSİYONLARI
def calculate_tbsa_cm2(height_cm, weight_kg):
    if not height_cm or not weight_kg or float(height_cm) <= 0 or float(weight_kg) <= 0:
        return 0.0
    # Mosteller formülü (m^2 cinsinden)
    tbsa_m2 = math.sqrt((float(height_cm) * float(weight_kg)) / 3600.0)
    return tbsa_m2 * 10000.0  # cm^2'ye çevir

def calculate_burn_area_cm2(burned_pixels_in_mask, image_dpi,
                            mask_width, mask_height, # Segmentasyon maskesinin boyutları
                            original_image_width, original_image_height):
    if image_dpi <= 0 or burned_pixels_in_mask == 0:
        return 0.0
    if mask_width == 0 or mask_height == 0 : # Sıfıra bölme hatasını önle
        print("Hata: Maske boyutları sıfır olamaz.")
        return 0.0

    # Segmentasyon maskesindeki bir pikselin, orijinal resimde kapladığı alanın oranı
    pixel_width_ratio_orig_to_mask = float(original_image_width) / float(mask_width)
    pixel_height_ratio_orig_to_mask = float(original_image_height) / float(mask_height)
    
    # Segmentasyon maskesindeki bir pikselin orijinal resimdeki alan karşılığı (orijinal piksel^2 cinsinden)
    area_scale_factor = pixel_width_ratio_orig_to_mask * pixel_height_ratio_orig_to_mask

    # Yanmış piksellerin orijinal resimdeki toplam piksel sayısı karşılığı
    total_burned_pixels_in_original_image_scale = float(burned_pixels_in_mask) * area_scale_factor

    # Orijinal resimdeki bir pikselin alanı (inç kare)
    area_per_original_pixel_inch_sq = (1.0 / float(image_dpi))**2

    # Toplam yanık alanı (inç kare)
    total_burn_area_inch_sq = total_burned_pixels_in_original_image_scale * area_per_original_pixel_inch_sq
    
    # Toplam yanık alanı (cm kare)
    total_burn_area_cm2 = total_burn_area_inch_sq * (2.54**2)
    
    return total_burn_area_cm2

# API ENDPOINT
@app.route("/api/predict", methods=["POST"])
def predict_route():
    print("✅ /predict endpointine istek geldi")
    print("🔍 Form alanları:", request.form)
    print("🔍 Dosyalar:", request.files)
    
    # Bellek temizliği
    gc.collect()

    global depth_model, segmentation_model # Global değişkenlere erişim

    # Modellerin yüklenip yüklenmediğini kontrol et
    if depth_model is None: # Segmentasyon modeli opsiyonel olabilir, ama derinlik olmalı
        return jsonify({"error": "Derinlik AI modeli yüklenemedi. Lütfen sunucu loglarını kontrol edin."}), 503

    if "image" not in request.files:
        return jsonify({"error": "Görsel yüklenmedi."}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = pil_image.size
        image_dpi = get_image_dpi(pil_image)
        print(f"Tespit edilen DPI: {image_dpi}, Orijinal Boyut: {original_width}x{original_height}")
    except Exception as e:
        print(f"Görsel işlenemedi: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Görsel işlenemedi: {e}"}), 400

    # Form verilerini al ve doğrula
    try:
        height_cm_str = request.form.get("height_cm")
        weight_kg_str = request.form.get("weight_kg")
        if not height_cm_str or not weight_kg_str:
            return jsonify({"error": "Boy ve kilo bilgileri eksik veya alınamadı."}), 400
        height_cm = float(height_cm_str)
        weight_kg = float(weight_kg_str)
    except ValueError:
        return jsonify({"error": "Boy veya kilo için geçersiz sayısal format."}), 400
    except Exception as e:
        print(f"Form verisi alınırken hata: {e}")
        traceback.print_exc()
        return jsonify({"error": "Form verileri işlenirken bir hata oluştu."}), 400

    # 1. Yanık Derinliği Tahmini
    predicted_burn_depth = "Hata (Derinlik)"
    confidence_depth = 0.0
    try:
        processed_image_depth = preprocess_image_for_depth(pil_image.copy()) # Orijinali bozmamak için kopya
        predictions_depth = depth_model.predict(processed_image_depth)
        class_index_depth = np.argmax(predictions_depth[0]) # predictions_depth[0] batch'in ilk elemanı
        predicted_burn_depth = class_labels_depth[class_index_depth]
        confidence_depth = float(predictions_depth[0][class_index_depth])
    except Exception as e:
        print(f"Derinlik tahmini sırasında hata: {e}")
        traceback.print_exc()

    # # 2. Yanık Alanı ve Yüzdesi Tahmini
    # burn_area_cm2_val = 0.0
    # burn_percentage_val = 0.0

    # if segmentation_model is not None:
    #     try:
    #         # Segmentasyon modelinin beklediği input boyutlarını al
    #         seg_model_input_shape = segmentation_model.input_shape 
    #         if isinstance(seg_model_input_shape, list): # Eğer model birden fazla input alıyorsa
    #             seg_model_input_shape = seg_model_input_shape[0] # İlk input'un şeklini al

    #         # input_shape (None, H, W, C) veya (H, W, C) olabilir. H ve W'yi al.
    #         # Model.input_shape TensorFlow/Keras versiyonuna göre tuple veya list of tuples dönebilir.
    #         seg_target_height = seg_model_input_shape[1] if seg_model_input_shape[1] is not None else 256 # Varsayılan
    #         seg_target_width = seg_model_input_shape[2] if seg_model_input_shape[2] is not None else 256  # Varsayılan
            
    #         print(f"Segmentasyon için hedef boyutlar: H={seg_target_height}, W={seg_target_width}")

    #         processed_img_seg = preprocess_image_for_segmentation(pil_image.copy(), 
    #                                                               target_height=seg_target_height, 
    #                                                               target_width=seg_target_width)
            
    #         # Model tahminini al, [0] ile batch boyutunu kaldır ([H, W, C] veya [H, W] elde et)
    #         segmentation_mask_pred_raw = segmentation_model.predict(processed_img_seg)[0] 

    #         # Maskenin gerçek boyutlarını al (preprocess sonrası)
    #         mask_height_for_calc = segmentation_mask_pred_raw.shape[0]
    #         mask_width_for_calc = segmentation_mask_pred_raw.shape[1]
            
    #         print(f"Segmentasyon modeli çıktı (maske) boyutu: {mask_width_for_calc}x{mask_height_for_calc}")
    #         print(f"Ham segmentasyon maskesi şekli: {segmentation_mask_pred_raw.shape}")

    #         # --- BU KISMI SEGMENTASYON MODELİNİZİN ÇIKTISINA GÖRE ÇOK DİKKATLİ AYARLAYIN! ---
    #         # Modelinizin kaç sınıfı var? Yanık sınıfının indeksi nedir? Çıktı sigmoid mi softmax mı?
    #         burn_pixels_mask = np.zeros((mask_height_for_calc, mask_width_for_calc), dtype=np.uint8) # Varsayılan boş maske

    #         if len(segmentation_mask_pred_raw.shape) == 3 and segmentation_mask_pred_raw.shape[-1] == 1:
    #             # Binary segmentasyon (yanık/yanık değil), çıktı [H, W, 1] ve sigmoid ise:
    #             print("İkili segmentasyon maskesi (tek kanal) işleniyor.")
    #             burn_pixels_mask = (segmentation_mask_pred_raw > 0.5).astype(np.uint8).squeeze(axis=-1)
    #         elif len(segmentation_mask_pred_raw.shape) == 2:
    #             # Binary segmentasyon, çıktı zaten [H, W] (örn: bazı modeller doğrudan binary mask dönebilir)
    #             print("İkili segmentasyon maskesi (kanalsız) işleniyor.")
    #             burn_pixels_mask = (segmentation_mask_pred_raw > 0.5).astype(np.uint8)
    #         elif len(segmentation_mask_pred_raw.shape) == 3 and segmentation_mask_pred_raw.shape[-1] > 1:
    #             # Multi-class segmentasyon, çıktı [H, W, NumClasses] ve softmax ise:
    #             NUM_CLASSES = segmentation_mask_pred_raw.shape[-1]
    #             print(f"Çok sınıflı segmentasyon maskesi işleniyor. Sınıf sayısı: {NUM_CLASSES}")
    #             # ÖNEMLİ: Yanık sınıfınızın doğru indeksini (veya indekslerini) buraya girin!
    #             # Örneğin, 0: arka plan, 1: yanık. Veya farklı dereceler için farklı sınıflar.
    #             # Eğer birden fazla sınıf "yanık" olarak kabul edilecekse, bu mantığı genişletin.
    #             BURN_CLASS_INDEX = 1 # BU DEĞERİ KESİNLİKLE KONTROL EDİN VE GÜNCELLEYİN!
    #             if BURN_CLASS_INDEX < NUM_CLASSES:
    #                 burn_pixels_mask = (np.argmax(segmentation_mask_pred_raw, axis=-1) == BURN_CLASS_INDEX).astype(np.uint8)
    #                 print(f"Yanık sınıfı indeksi {BURN_CLASS_INDEX} kullanıldı.")
    #             else:
    #                 print(f"HATA: BURN_CLASS_INDEX ({BURN_CLASS_INDEX}) sınıf sayısından ({NUM_CLASSES}) büyük veya eşit olamaz! Yanık alanı 0 olarak hesaplanacak.")
    #         else:
    #             print(f"Uyarı: Beklenmedik segmentasyon maskesi şekli: {segmentation_mask_pred_raw.shape}. Yanık alanı 0 olarak hesaplanacak.")
    #         # --- AYARLAMA BİTİŞ ---

    #         burned_pixel_count_in_mask = np.sum(burn_pixels_mask)
    #         print(f"Segmentasyon maskesindeki ({mask_width_for_calc}x{mask_height_for_calc}) yanık piksel sayısı: {burned_pixel_count_in_mask}")

    #         if burned_pixel_count_in_mask > 0:
    #             burn_area_cm2_val = calculate_burn_area_cm2(
    #                 burned_pixel_count_in_mask,
    #                 image_dpi,
    #                 mask_width_for_calc, # Segmentasyon maskesinin genişliği
    #                 mask_height_for_calc, # Segmentasyon maskesinin yüksekliği
    #                 original_width,       # Orijinal resmin genişliği
    #                 original_height       # Orijinal resmin yüksekliği
    #             )
    #             tbsa_cm2 = calculate_tbsa_cm2(height_cm, weight_kg)
    #             if tbsa_cm2 > 0:
    #                 burn_percentage_val = (burn_area_cm2_val / tbsa_cm2) * 100.0
    #             else:
    #                 print("TBSA hesaplanamadı (boy/kilo sıfır veya geçersiz). Yanık yüzdesi 0 olacak.")
    #         else:
    #              print("Maskede yanık piksel bulunamadı. Alan ve yüzde 0 olacak.")

    #     except Exception as e:
    #         print(f"Segmentasyon, alan veya yüzde hesaplama sırasında genel hata: {e}")
    #         traceback.print_exc()
    # else:
    #     print("Segmentasyon modeli yüklenmemiş veya bulunamadı. Alan ve yüzde hesaplaması atlanıyor.")

    # Sonuçları JSON olarak döndür
    return jsonify({
        "burn_depth": predicted_burn_depth,
        "confidence_depth": round(confidence_depth, 4), # Güven skorunu da ekleyelim
        "burn_area_cm2": 0.0, #round(burn_area_cm2_val, 2),
        "burn_percentage": 0.0, #round(burn_percentage_val, 2),
        "detected_dpi": round(image_dpi, 2),
        "original_dimensions": f"{original_width}x{original_height}"
    })

@app.route("/healthz", methods=["GET"])
def health_check():
    return "OK", 200

# # 🔁 En sonda uygulama başlatma:
# if __name__ == "__main__":
#     #load_ai_models()
#     app.run(host="0.0.0.0", port=5000)

    


    
# # ai_server.py

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import numpy as np
# from PIL import Image, ExifTags # ExifTags JPEG DPI için
# import io
# import math
# import os
# import traceback # Hata ayıklama için
# import boto3

# app = Flask(__name__)

# # CORS Yapılandırması: React uygulamanızın çalıştığı origin'e izin verin.
# CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# # Global değişkenler modeller için
# depth_model = None
# segmentation_model = None

# # Betiğin bulunduğu dizini al
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Sınıf isimleri (Derinlik modeli için)
# class_labels_depth = ["Birinci Derece Yanık", "İkinci Derece Yüzeysel Yanık", "İkinci Derece Derin Yanık", "Üçüncü Derece Yanık"]

# # Modelleri yükleme fonksiyonu
# def load_ai_models():
#     global depth_model, segmentation_model

#     # Derinlik Modelini Yükle
#     try:
#         depth_model_path = os.path.join(SCRIPT_DIR, "best_model.h5")
#         if os.path.exists(depth_model_path):
#             depth_model = tf.keras.models.load_model(depth_model_path, compile=False)
#             print(f"Derinlik modeli '{depth_model_path}' başarıyla yüklendi.")
#         else:
#             print(f"HATA: Derinlik modeli bulunamadı: {depth_model_path}")
#     except Exception as e:
#         print(f"Derinlik modeli yüklenirken hata: {e}")
#         traceback.print_exc()

#     # Segmentasyon Modelini Yükle
#     try:
#         # ÖNEMLİ: Kendi segmentasyon modelinizin dosya adını buraya doğru girin!
#         segmentation_model_filename = "segmentation_model.h5" # GERÇEK MODEL ADINIZI YAZIN
#         segmentation_model_path = os.path.join(SCRIPT_DIR, segmentation_model_filename)
#         if os.path.exists(segmentation_model_path):
#             segmentation_model = tf.keras.models.load_model(segmentation_model_path, compile=False)
#             print(f"Segmentasyon modeli '{segmentation_model_path}' başarıyla yüklendi.")
#         else:
#             print(f"HATA: Segmentasyon modeli bulunamadı: {segmentation_model_path}")
#     except Exception as e:
#         print(f"Segmentasyon modeli '{segmentation_model_path}' yüklenirken hata: {e}")
#         traceback.print_exc()

# # Resimden DPI bilgisini okuma fonksiyonu
# def get_image_dpi(image_pil_object):
#     dpi_val = image_pil_object.info.get('dpi')
#     if dpi_val:
#         # DPI (yatay, dikey) tuple olabilir, genellikle aynıdır
#         return float(dpi_val[0]) if isinstance(dpi_val, (tuple, list)) else float(dpi_val)

#     # JPEG için EXIF kontrolü
#     try:
#         exif_data = image_pil_object.getexif() # _getexif() yerine getexif() daha güncel
#         if exif_data:
#             # EXIF tag'lerini isimlere çevir (Pillow 9.0.0 ve sonrası için direkt tag ID kullanılabilir)
#             # Basitlik için direkt ID'leri kullanalım (XResolution: 282, YResolution: 283, ResolutionUnit: 296)
#             x_res_tag = 282
#             res_unit_tag = 296

#             x_res_value = exif_data.get(x_res_tag)
#             res_unit_value = exif_data.get(res_unit_tag)

#             if x_res_value and res_unit_value:
#                 # XResolution genellikle (pay, payda) şeklinde bir Rational nesnesidir
#                 if hasattr(x_res_value, 'numerator') and hasattr(x_res_value, 'denominator'):
#                     val_res = float(x_res_value.numerator) / float(x_res_value.denominator) if x_res_value.denominator != 0 else float(x_res_value.numerator)
#                 else: # Bazen direkt float gelebilir
#                     val_res = float(x_res_value)

#                 if res_unit_value == 2: # Inç
#                     return val_res
#                 elif res_unit_value == 3: # Santimetre (PPC -> PPI)
#                     return val_res * 2.54
#     except Exception as e:
#         print(f"EXIF okunurken hata (normal olabilir): {e}")
#         pass # EXIF okuma başarısız olursa sorun değil

#     print("Uyarı: Resimden DPI bilgisi okunamadı, varsayılan 96 DPI kullanılacak.")
#     return 96.0

# # GÖRSEL ÖN İŞLEME FONKSİYONLARI
# # BU FONKSİYONLARI KENDİ MODELLERİNİZİN GEREKSİNİMLERİNE GÖRE AYARLAYIN!
# def preprocess_image_for_depth(pil_image):
#     # Derinlik modelinin beklediği boyut (örneğin 640x640, 128x128)
#     image = pil_image.convert("RGB").resize((128,128))
#     image_array = np.array(image) / 255.0
#     return np.expand_dims(image_array, axis=0) # Batch boyutu ekle [1, H, W, C]

# def preprocess_image_for_segmentation(pil_image, target_height=256, target_width=256):
#     # Segmentasyon modelinin beklediği boyut ve ön işleme
#     # target_height ve target_width, modelin input_shape'inden alınabilir.
#     image = pil_image.convert("RGB").resize((target_width, target_height))
#     image_array = np.array(image) / 255.0 # Yaygın bir normalizasyon
#     return np.expand_dims(image_array, axis=0) # Batch boyutu ekle

# # HESAPLAMA FONKSİYONLARI
# def calculate_tbsa_cm2(height_cm, weight_kg):
#     if not height_cm or not weight_kg or float(height_cm) <= 0 or float(weight_kg) <= 0:
#         return 0.0
#     # Mosteller formülü (m^2 cinsinden)
#     tbsa_m2 = math.sqrt((float(height_cm) * float(weight_kg)) / 3600.0)
#     return tbsa_m2 * 10000.0  # cm^2'ye çevir

# def calculate_burn_area_cm2(burned_pixels_in_mask, image_dpi,
#                             mask_width, mask_height, # Segmentasyon maskesinin boyutları
#                             original_image_width, original_image_height):
#     if image_dpi <= 0 or burned_pixels_in_mask == 0:
#         return 0.0
#     if mask_width == 0 or mask_height == 0 : # Sıfıra bölme hatasını önle
#         print("Hata: Maske boyutları sıfır olamaz.")
#         return 0.0

#     # Segmentasyon maskesindeki bir pikselin, orijinal resimde kapladığı alanın oranı
#     pixel_width_ratio_orig_to_mask = float(original_image_width) / float(mask_width)
#     pixel_height_ratio_orig_to_mask = float(original_image_height) / float(mask_height)
    
#     # Segmentasyon maskesindeki bir pikselin orijinal resimdeki alan karşılığı (orijinal piksel^2 cinsinden)
#     area_scale_factor = pixel_width_ratio_orig_to_mask * pixel_height_ratio_orig_to_mask

#     # Yanmış piksellerin orijinal resimdeki toplam piksel sayısı karşılığı
#     total_burned_pixels_in_original_image_scale = float(burned_pixels_in_mask) * area_scale_factor

#     # Orijinal resimdeki bir pikselin alanı (inç kare)
#     area_per_original_pixel_inch_sq = (1.0 / float(image_dpi))**2

#     # Toplam yanık alanı (inç kare)
#     total_burn_area_inch_sq = total_burned_pixels_in_original_image_scale * area_per_original_pixel_inch_sq
    
#     # Toplam yanık alanı (cm kare)
#     total_burn_area_cm2 = total_burn_area_inch_sq * (2.54**2)
    
#     return total_burn_area_cm2

# # API ENDPOINT
# @app.route("/predict", methods=["POST"])
# def predict_route():
#     global depth_model, segmentation_model # Global değişkenlere erişim

#     # Modellerin yüklenip yüklenmediğini kontrol et
#     if depth_model is None: # Segmentasyon modeli opsiyonel olabilir, ama derinlik olmalı
#         return jsonify({"error": "Derinlik AI modeli yüklenemedi. Lütfen sunucu loglarını kontrol edin."}), 503

#     if "image" not in request.files:
#         return jsonify({"error": "Görsel yüklenmedi."}), 400

#     file = request.files["image"]
#     image_bytes = file.read()

#     try:
#         pil_image = Image.open(io.BytesIO(image_bytes))
#         original_width, original_height = pil_image.size
#         image_dpi = get_image_dpi(pil_image)
#         print(f"Tespit edilen DPI: {image_dpi}, Orijinal Boyut: {original_width}x{original_height}")
#     except Exception as e:
#         print(f"Görsel işlenemedi: {e}")
#         traceback.print_exc()
#         return jsonify({"error": f"Görsel işlenemedi: {e}"}), 400

#     # Form verilerini al ve doğrula
#     try:
#         height_cm_str = request.form.get("height_cm")
#         weight_kg_str = request.form.get("weight_kg")
#         if not height_cm_str or not weight_kg_str:
#             return jsonify({"error": "Boy ve kilo bilgileri eksik veya alınamadı."}), 400
#         height_cm = float(height_cm_str)
#         weight_kg = float(weight_kg_str)
#     except ValueError:
#         return jsonify({"error": "Boy veya kilo için geçersiz sayısal format."}), 400
#     except Exception as e:
#         print(f"Form verisi alınırken hata: {e}")
#         traceback.print_exc()
#         return jsonify({"error": "Form verileri işlenirken bir hata oluştu."}), 400

#     # 1. Yanık Derinliği Tahmini
#     predicted_burn_depth = "Hata (Derinlik)"
#     confidence_depth = 0.0
#     try:
#         processed_image_depth = preprocess_image_for_depth(pil_image.copy()) # Orijinali bozmamak için kopya
#         predictions_depth = depth_model.predict(processed_image_depth)
#         class_index_depth = np.argmax(predictions_depth[0]) # predictions_depth[0] batch'in ilk elemanı
#         predicted_burn_depth = class_labels_depth[class_index_depth]
#         confidence_depth = float(predictions_depth[0][class_index_depth])
#     except Exception as e:
#         print(f"Derinlik tahmini sırasında hata: {e}")
#         traceback.print_exc()

#     # 2. Yanık Alanı ve Yüzdesi Tahmini
#     burn_area_cm2_val = 0.0
#     burn_percentage_val = 0.0

#     if segmentation_model is not None:
#         try:
#             # Segmentasyon modelinin beklediği input boyutlarını al
#             seg_model_input_shape = segmentation_model.input_shape 
#             if isinstance(seg_model_input_shape, list): # Eğer model birden fazla input alıyorsa
#                 seg_model_input_shape = seg_model_input_shape[0] # İlk input'un şeklini al

#             # input_shape (None, H, W, C) veya (H, W, C) olabilir. H ve W'yi al.
#             # Model.input_shape TensorFlow/Keras versiyonuna göre tuple veya list of tuples dönebilir.
#             seg_target_height = seg_model_input_shape[1] if seg_model_input_shape[1] is not None else 256 # Varsayılan
#             seg_target_width = seg_model_input_shape[2] if seg_model_input_shape[2] is not None else 256  # Varsayılan
            
#             print(f"Segmentasyon için hedef boyutlar: H={seg_target_height}, W={seg_target_width}")

#             processed_img_seg = preprocess_image_for_segmentation(pil_image.copy(), 
#                                                                   target_height=seg_target_height, 
#                                                                   target_width=seg_target_width)
            
#             # Model tahminini al, [0] ile batch boyutunu kaldır ([H, W, C] veya [H, W] elde et)
#             segmentation_mask_pred_raw = segmentation_model.predict(processed_img_seg)[0] 

#             # Maskenin gerçek boyutlarını al (preprocess sonrası)
#             mask_height_for_calc = segmentation_mask_pred_raw.shape[0]
#             mask_width_for_calc = segmentation_mask_pred_raw.shape[1]
            
#             print(f"Segmentasyon modeli çıktı (maske) boyutu: {mask_width_for_calc}x{mask_height_for_calc}")
#             print(f"Ham segmentasyon maskesi şekli: {segmentation_mask_pred_raw.shape}")

#             # --- BU KISMI SEGMENTASYON MODELİNİZİN ÇIKTISINA GÖRE ÇOK DİKKATLİ AYARLAYIN! ---
#             # Modelinizin kaç sınıfı var? Yanık sınıfının indeksi nedir? Çıktı sigmoid mi softmax mı?
#             burn_pixels_mask = np.zeros((mask_height_for_calc, mask_width_for_calc), dtype=np.uint8) # Varsayılan boş maske

#             if len(segmentation_mask_pred_raw.shape) == 3 and segmentation_mask_pred_raw.shape[-1] == 1:
#                 # Binary segmentasyon (yanık/yanık değil), çıktı [H, W, 1] ve sigmoid ise:
#                 print("İkili segmentasyon maskesi (tek kanal) işleniyor.")
#                 burn_pixels_mask = (segmentation_mask_pred_raw > 0.5).astype(np.uint8).squeeze(axis=-1)
#             elif len(segmentation_mask_pred_raw.shape) == 2:
#                 # Binary segmentasyon, çıktı zaten [H, W] (örn: bazı modeller doğrudan binary mask dönebilir)
#                 print("İkili segmentasyon maskesi (kanalsız) işleniyor.")
#                 burn_pixels_mask = (segmentation_mask_pred_raw > 0.5).astype(np.uint8)
#             elif len(segmentation_mask_pred_raw.shape) == 3 and segmentation_mask_pred_raw.shape[-1] > 1:
#                 # Multi-class segmentasyon, çıktı [H, W, NumClasses] ve softmax ise:
#                 NUM_CLASSES = segmentation_mask_pred_raw.shape[-1]
#                 print(f"Çok sınıflı segmentasyon maskesi işleniyor. Sınıf sayısı: {NUM_CLASSES}")
#                 # ÖNEMLİ: Yanık sınıfınızın doğru indeksini (veya indekslerini) buraya girin!
#                 # Örneğin, 0: arka plan, 1: yanık. Veya farklı dereceler için farklı sınıflar.
#                 # Eğer birden fazla sınıf "yanık" olarak kabul edilecekse, bu mantığı genişletin.
#                 BURN_CLASS_INDEX = 1 # BU DEĞERİ KESİNLİKLE KONTROL EDİN VE GÜNCELLEYİN!
#                 if BURN_CLASS_INDEX < NUM_CLASSES:
#                     burn_pixels_mask = (np.argmax(segmentation_mask_pred_raw, axis=-1) == BURN_CLASS_INDEX).astype(np.uint8)
#                     print(f"Yanık sınıfı indeksi {BURN_CLASS_INDEX} kullanıldı.")
#                 else:
#                     print(f"HATA: BURN_CLASS_INDEX ({BURN_CLASS_INDEX}) sınıf sayısından ({NUM_CLASSES}) büyük veya eşit olamaz! Yanık alanı 0 olarak hesaplanacak.")
#             else:
#                 print(f"Uyarı: Beklenmedik segmentasyon maskesi şekli: {segmentation_mask_pred_raw.shape}. Yanık alanı 0 olarak hesaplanacak.")
#             # --- AYARLAMA BİTİŞ ---

#             burned_pixel_count_in_mask = np.sum(burn_pixels_mask)
#             print(f"Segmentasyon maskesindeki ({mask_width_for_calc}x{mask_height_for_calc}) yanık piksel sayısı: {burned_pixel_count_in_mask}")

#             if burned_pixel_count_in_mask > 0:
#                 burn_area_cm2_val = calculate_burn_area_cm2(
#                     burned_pixel_count_in_mask,
#                     image_dpi,
#                     mask_width_for_calc, # Segmentasyon maskesinin genişliği
#                     mask_height_for_calc, # Segmentasyon maskesinin yüksekliği
#                     original_width,       # Orijinal resmin genişliği
#                     original_height       # Orijinal resmin yüksekliği
#                 )
#                 tbsa_cm2 = calculate_tbsa_cm2(height_cm, weight_kg)
#                 if tbsa_cm2 > 0:
#                     burn_percentage_val = (burn_area_cm2_val / tbsa_cm2) * 100.0
#                 else:
#                     print("TBSA hesaplanamadı (boy/kilo sıfır veya geçersiz). Yanık yüzdesi 0 olacak.")
#             else:
#                  print("Maskede yanık piksel bulunamadı. Alan ve yüzde 0 olacak.")

#         except Exception as e:
#             print(f"Segmentasyon, alan veya yüzde hesaplama sırasında genel hata: {e}")
#             traceback.print_exc()
#     else:
#         print("Segmentasyon modeli yüklenmemiş veya bulunamadı. Alan ve yüzde hesaplaması atlanıyor.")

#     # Sonuçları JSON olarak döndür
#     return jsonify({
#         "burn_depth": predicted_burn_depth,
#         "confidence_depth": round(confidence_depth, 4), # Güven skorunu da ekleyelim
#         "burn_area_cm2": round(burn_area_cm2_val, 2),
#         "burn_percentage": round(burn_percentage_val, 2),
#         "detected_dpi": round(image_dpi, 2),
#         "original_dimensions": f"{original_width}x{original_height}"
#     })

# # Uygulama başlangıcında modelleri yükle
# if __name__ == "__main__":
#     load_ai_models()
#     app.run(host="0.0.0.0", port=5000, debug=True) # Geliştirme için debug=True