import os 
import shutil
import json
import psycopg2
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# 🌱 .env dosyasını yükle
load_dotenv()

# 🔐 Veritabanı bağlantı bilgileri
DB_PARAMS = {
    'host': os.getenv("DB_HOST"),
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'port': int(os.getenv("DB_PORT", 5432))  # varsayılan port 5432
}

# 📁 Yol ayarları
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR.parent / "wwwroot" / "uploads"
BEST_MODEL_PATH = BASE_DIR / "best_model.h5"
CLASS_INDEX_PATH = BASE_DIR / "class_indices.json"
MODEL_BACKUP_DIR = BASE_DIR / "old_versions"

# 📥 Yeni veriyi veritabanından çek
def fetch_verified_untrained_data():
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT "PatientID", "PhotoPath", "BurnDepth" FROM "Patients" WHERE "Verified" = TRUE AND "Trained" = FALSE LIMIT 100')
            # cur.execute("""
            #     SELECT "PatientID", "PhotoPath",
            #         CASE
            #             WHEN "BurnDepth" = 'Birinci Derece Yanık' THEN 'First_degree'
            #             WHEN "BurnDepth" = 'İkinci Derece Yüzeysel Yanık' THEN 'Second_degree_superficial'
            #             WHEN "BurnDepth" = 'İkinci Derece Derin Yanık' THEN 'Second_degree_deep'
            #             WHEN "BurnDepth" = 'Üçüncü Derece Yanık' THEN 'Third_degree'
            #             ELSE NULL
            #         END AS "BurnDepth"
            #     FROM "Patients"
            #     WHERE "Verified" = TRUE AND "Trained" = FALSE
            #     LIMIT 100
            # """)
            return cur.fetchall()


# ✅ Trained flag'lerini güncelle
def mark_as_trained(patient_ids):
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.executemany('UPDATE "Patients" SET "Trained" = TRUE WHERE "PatientID" = %s', [(pid,) for pid in patient_ids])
        conn.commit()

# 🔄 DataFrame hazırla
def prepare_dataframe(records):
    images, labels = [], []
    for pid, photo_path, label in records:
        abs_path = UPLOADS_DIR / os.path.basename(photo_path)
        if abs_path.exists():
            images.append(str(abs_path))
            labels.append(str(label))
        else:
            print(f"⚠️ {abs_path} bulunamadı, atlanıyor.")
    return pd.DataFrame({'filename': images, 'class': labels})

# 🔄 Eğitim ve doğrulama veri üreteçleri
def load_data(df, class_indices):
    datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)
    train = datagen.flow_from_dataframe(df, x_col='filename', y_col='class',
                                        target_size=(128, 128), batch_size=8,
                                        class_mode='categorical', subset='training', shuffle=False)
    val = datagen.flow_from_dataframe(df, x_col='filename', y_col='class',
                                      target_size=(128, 128), batch_size=8,
                                      class_mode='categorical', subset='validation', shuffle=False)
    train.class_indices = class_indices
    val.class_indices = class_indices
    return train, val

# 🧠 Modeli ince ayarla
def fine_tune_model(train, val):
    model = load_model(BEST_MODEL_PATH)
    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers[-5:]:
        layer.trainable = True
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train, validation_data=val, epochs=3)
    return model, history.history['accuracy'][-1]

# 🧪 Mevcut modeli değerlendir
def evaluate_model(model, dataset):
    return model.evaluate(dataset, verbose=0)[1]

# 💾 Daha iyi model varsa kaydet
def save_model_if_better(new_model, new_acc, old_acc):
    MODEL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if new_acc > old_acc:
        backup_path = MODEL_BACKUP_DIR / f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        shutil.copy2(BEST_MODEL_PATH, backup_path)
        print(f"📦 Eski model yedeklendi: {backup_path}")
        new_model.save(BEST_MODEL_PATH)
        print(f"✅ Yeni model kaydedildi: {BEST_MODEL_PATH}")
        return True
    else:
        print(f"⚠️ Yeni model daha düşük doğrulukta. Kaydedilmedi.")
        return False

# 🔁 Tüm pipeline akışı
def run_retraining_pipeline():
    print("🚀 Continual Learning başlatılıyor...")
    data = fetch_verified_untrained_data()
    if len(data) < 10:
        print("🚫 Yeterli doğrulanmış eğitilmemiş veri yok. Çıkılıyor.")
        return

    with open(CLASS_INDEX_PATH) as f:
        class_indices = json.load(f)

    df = prepare_dataframe(data)
    if df.empty:
        print("🚫 Uygun veri bulunamadı.")
        return
    
    print("🎯 Sınıf dağılımı:")
    print(df['class'].value_counts())


    train, val = load_data(df, class_indices)
    new_model, new_train_acc = fine_tune_model(train, val)

    old_model = load_model(BEST_MODEL_PATH)
    old_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    old_train_acc = evaluate_model(old_model, train)

    print(f"🆕 Yeni model acc: {new_train_acc:.4f}, 🏅 Eski model acc: {old_train_acc:.4f}")
    is_better = save_model_if_better(new_model, new_train_acc, old_train_acc)

    if is_better:
        mark_as_trained([r[0] for r in data])
        print("📌 Veritabanı güncellendi.")
    else:
        print("📌 Veritabanında değişiklik yapılmadı.")

# 🧪 Ana giriş
if __name__ == "__main__":
    run_retraining_pipeline()

# import os
# import shutil
# import json
# import psycopg2
# import pandas as pd
# from datetime import datetime
# from pathlib import Path
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam

# # Bağlantı bilgileri
# DB_PARAMS = {
#     'host': 'localhost',
#     'dbname': 'BurnAnalysisDB',
#     'user': 'postgres',
#     'password': 'admin123',
#     'port': 5432
# }

# # Yol ayarları
# BEST_MODEL_PATH = "best_model.h5"
# CLASS_INDEX_PATH = "class_indices.json"
# MODEL_BACKUP_DIR = "old_versions"
# #NEW_IMAGES_DIR = "new_training_images"

# BASE_DIR = Path(__file__).resolve().parent.parent
# UPLOADS_BASE_DIR = BASE_DIR / "wwwroot" / "uploads"

# def fetch_verified_untrained_data():
#     with psycopg2.connect(**DB_PARAMS) as conn:
#         with conn.cursor() as cur:
#             cur.execute('SELECT "PatientID", "PhotoPath", "BurnDepth" FROM "Patients" WHERE "Verified" = TRUE AND "Trained" = FALSE LIMIT 100')
#             return cur.fetchall()

# def mark_as_trained(patient_ids):
#     with psycopg2.connect(**DB_PARAMS) as conn:
#         with conn.cursor() as cur:
#             cur.executemany('UPDATE "Patients" SET "Trained" = TRUE WHERE "PatientID" = %s', [(pid,) for pid in patient_ids])
#         conn.commit()

# def prepare_dataframe(records):
#     images = []
#     labels = []
#     for pid, photo_path, label in records:
#         abs_path = os.path.join(UPLOADS_BASE_DIR, os.path.basename(photo_path))
#         if not os.path.exists(abs_path):
#             print(f"⚠️ {abs_path} bulunamadı, atlanıyor.")
#             continue
#         images.append(abs_path)
#         labels.append(str(label))
#     df = pd.DataFrame({'filename': images, 'class': labels})
#     return df

# def load_data(df, class_indices):
#     datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)

#     train_gen = datagen.flow_from_dataframe(
#         df,
#         x_col='filename',
#         y_col='class',
#         target_size=(128, 128),
#         batch_size=8,
#         class_mode='categorical',
#         shuffle=False,  # 👈 evaluate ederken kesinlikle bu olmalı!
#         subset='training'
#     )

#     val_gen = datagen.flow_from_dataframe(
#         df,
#         x_col='filename',
#         y_col='class',
#         target_size=(128, 128),
#         batch_size=8,
#         class_mode='categorical',
#         shuffle=False,
#         subset='validation'
#     )

#     train_gen.class_indices = class_indices
#     val_gen.class_indices = class_indices

#     return train_gen, val_gen

# def fine_tune_model(train, val):
#     model = load_model(BEST_MODEL_PATH)

#     for layer in model.layers[:-5]:
#         layer.trainable = False
#     for layer in model.layers[-5:]:
#         layer.trainable = True

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     history = model.fit(train, validation_data=val, epochs=3)
#     new_train_acc = history.history['accuracy'][-1]
#     return model, new_train_acc

# def evaluate_model(model, dataset):
#     return model.evaluate(dataset, verbose=0)[1]

# def save_model_if_better(new_model, new_acc, old_acc):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)

#     if new_acc > old_acc:
#         backup_path = os.path.join(MODEL_BACKUP_DIR, f"model_backup_{timestamp}.h5")
#         shutil.copy2(BEST_MODEL_PATH, backup_path)
#         print(f"📦 Eski model yedeklendi: {backup_path}")
#         new_model.save(BEST_MODEL_PATH)
#         print(f"✅ Yeni model kaydedildi: {BEST_MODEL_PATH}")
#         return True
#     else:
#         print(f"⚠️ Yeni model doğruluğu daha düşük. Mevcut model korundu.")
#         return False

# def run_retraining_pipeline():
#     print("🚀 Continual Learning Pipeline Başlıyor...")
#     data = fetch_verified_untrained_data()
#     print(f"📦 Eğitilmemiş kayıt sayısı: {len(data)}")

#     if len(data) < 100:
#         print("🚫 Yeterli veri yok (en az 100 kayıt gerekir).")
#         return

#     with open(CLASS_INDEX_PATH) as f:
#         class_indices = json.load(f)

#     df = prepare_dataframe(data)
#     if df.empty:
#         print("🚫 Uygun görsel bulunamadı. Çıkılıyor.")
#         return

#     train, val = load_data(df, class_indices)

#     print("Val class indices:", val.class_indices)
#     print("Yüklenen class_indices.json:", class_indices)

#     new_model, new_train_acc = fine_tune_model(train, val)

#     old_model = load_model(BEST_MODEL_PATH)
#     old_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     old_train_acc = evaluate_model(old_model, train)

#     print(f"🆕 Yeni model *train* doğruluğu: {new_train_acc:.4f}, 🏅 Mevcut model *train* doğruluğu: {old_train_acc:.4f}")
#     is_better = save_model_if_better(new_model, new_train_acc, old_train_acc)

#     if is_better:
#         mark_as_trained([r[0] for r in data])
#         print("📌 Veritabanı güncellendi: trained = TRUE")
#     else:
#         print("📌 trained alanı değişmedi (FALSE olarak kaldı)")

# if __name__ == "__main__":
#      run_retraining_pipeline()
     
# # import os 
# # import shutil
# # import json
# # import psycopg2
# # import pandas as pd
# # from datetime import datetime
# # from pathlib import Path
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# # from tensorflow.keras.optimizers import Adam

# # # Bağlantı bilgileri
# # DB_PARAMS = {
# #     'host': 'localhost',
# #     'dbname': 'BurnAnalysisDB',
# #     'user': 'postgres',
# #     'password': 'admin123',
# #     'port': 5432
# # }

# # # Yol ayarları
# # BEST_MODEL_PATH = "best_model.h5"
# # CLASS_INDEX_PATH = "class_indices.json"
# # MODEL_BACKUP_DIR = "old_versions"
# # NEW_IMAGES_DIR = "new_training_images"

# # BASE_DIR = Path(__file__).resolve().parent.parent
# # UPLOADS_BASE_DIR = BASE_DIR / "wwwroot" / "uploads"

# # def fetch_verified_untrained_data():
# #     with psycopg2.connect(**DB_PARAMS) as conn:
# #         with conn.cursor() as cur:
# #             cur.execute('SELECT "PatientID", "PhotoPath", "BurnDepth" FROM "Patients" WHERE "Verified" = TRUE AND "Trained" = FALSE LIMIT 100')
# #             return cur.fetchall()

# # def mark_as_trained(patient_ids):
# #     with psycopg2.connect(**DB_PARAMS) as conn:
# #         with conn.cursor() as cur:
# #             cur.executemany('UPDATE "Patients" SET "Trained" = TRUE WHERE "PatientID" = %s', [(pid,) for pid in patient_ids])
# #         conn.commit()

# # def prepare_dataframe(records):
# #     images = []
# #     labels = []
# #     for pid, photo_path, label in records:
# #         abs_path = os.path.join(UPLOADS_BASE_DIR, os.path.basename(photo_path))
# #         if not os.path.exists(abs_path):
# #             print(f"⚠️ {abs_path} bulunamadı, atlanıyor.")
# #             continue
# #         images.append(abs_path)
# #         labels.append(str(label))
# #     df = pd.DataFrame({'filename': images, 'class': labels})
# #     return df

# # def load_data(df, class_indices):
# #     datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)

# #     train_gen = datagen.flow_from_dataframe(
# #         df,
# #         x_col='filename',
# #         y_col='class',
# #         target_size=(640, 640),
# #         batch_size=8,
# #         class_mode='categorical',
# #         shuffle=True,
# #         subset='training'
# #     )

# #     val_gen = datagen.flow_from_dataframe(
# #         df,
# #         x_col='filename',
# #         y_col='class',
# #         target_size=(640, 640),
# #         batch_size=8,
# #         class_mode='categorical',
# #         shuffle=False,
# #         subset='validation'
# #     )

# #     train_gen.class_indices = class_indices
# #     val_gen.class_indices = class_indices

# #     return train_gen, val_gen

# # def fine_tune_model(train, val):
# #     model = load_model(BEST_MODEL_PATH)
# #     for layer in model.layers[:-5]:
# #         layer.trainable = False
# #     for layer in model.layers[-5:]:
# #         layer.trainable = True

# #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #     history = model.fit(train, validation_data=val, epochs=3)
# #     return model, history.history['val_accuracy'][-1]

# # def evaluate_model(model, df, class_indices):
# #     datagen = ImageDataGenerator(rescale=1./255.)

# #     val_gen = datagen.flow_from_dataframe(
# #         df,
# #         x_col='filename',
# #         y_col='class',
# #         target_size=(640, 640),
# #         batch_size=8,
# #         class_mode='categorical',
# #         shuffle=False
# #     )

# #     val_gen.class_indices = class_indices
# #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #     return model.evaluate(val_gen, verbose=0)[1]

# # def save_model_if_better(new_model, new_acc, old_acc):
# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)

# #     if new_acc > old_acc:
# #         backup_path = os.path.join(MODEL_BACKUP_DIR, f"model_backup_{timestamp}.h5")
# #         shutil.copy2(BEST_MODEL_PATH, backup_path)
# #         print(f"📦 Eski model yedeklendi: {backup_path}")
# #         new_model.save(BEST_MODEL_PATH)
# #         print(f"✅ Yeni model kaydedildi: {BEST_MODEL_PATH}")
# #         return True
# #     else:
# #         print(f"⚠️ Yeni model doğruluğu daha düşük. Mevcut model korundu.")
# #         return False

# # def run_retraining_pipeline():
# #     print("🚀 Continual Learning Pipeline Başlıyor...")
# #     data = fetch_verified_untrained_data()
# #     print(f"📦 Eğitilmemiş kayıt sayısı: {len(data)}")

# #     if len(data) < 10:
# #         print("🚫 Yeterli veri yok (en az 10 kayıt gerekir).")
# #         return

# #     with open(CLASS_INDEX_PATH) as f:
# #         class_indices = json.load(f)

# #     df = prepare_dataframe(data)
# #     if df.empty:
# #         print("🚫 Uygun görsel bulunamadı. Çıkılıyor.")
# #         return

# #     train, val = load_data(df, class_indices)

# #     print("Val class indices:", val.class_indices)
# #     print("Yüklenen class_indices.json:", class_indices)

# #     new_model, new_acc = fine_tune_model(train, val)

# #     old_model = load_model(BEST_MODEL_PATH)
# #     val_df = df.sample(frac=0.2, random_state=42)  # sabit validation subset
# #     old_acc = evaluate_model(old_model, val_df, class_indices)

# #     print(f"🆕 Yeni model doğruluğu: {new_acc:.4f}, 🏅 Mevcut model doğruluğu: {old_acc:.4f}")
# #     is_better = save_model_if_better(new_model, new_acc, old_acc)

# #     if is_better:
# #         mark_as_trained([r[0] for r in data])
# #         print("📌 Veritabanı güncellendi: trained = TRUE")
# #     else:
# #         print("📌 trained alanı değişmedi (FALSE olarak kaldı)")

# # if __name__ == "__main__":
# #     run_retraining_pipeline()
