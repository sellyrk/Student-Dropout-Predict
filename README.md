# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding
Saat ini, Jaya Jaya Institut telah memiliki ribuan mahasiswa aktif dan telah mencetak lulusan yang membanggakan untuk negeri. Selain lulusan yang membanggakan, tidak dapat dipungkiri, dari seluruh mahasiswa masih ada mahasiswa yang mengalami dropout. Dalam menghadapi jumlah dropout yang dapat saja meningkat, institusi harus memahami data mahasiswa yang ada sebagai dasar pengambilan keputusan untuk mengurangi jumlah dropout yang mungkin terjadi di masa mendatang.

### Permasalahan Bisnis
Apa saja faktor yang menyebabkan seorang individu mahasiswa berpotensi untuk melakukan dropout?

### Cakupan Proyek
Proyek ini bertujuan untuk membantu institusi dalam mengenali faktor yang menyebabkan dropout dengan menggunakan pendekatan analisis data dan algoritma klasifikasi. Adapun ruang lingkup proyek meliputi:

1. Persiapan: Menyiapkan library yang dibutuhkan serta melakukan proses loading dan pengecekan awal terhadap dataset yang digunakan
2. Data Understanding: Memahami struktur dan isi data melalui eksplorasi awal, serta mengidentifikasi tipe variabel numerik dan kategorik yang akan dianalisis.
3. Data Cleaning: Melakukan pembersihan data pada variabel numerik dan kategorik untuk mengatasi nilai kosong, duplikat, atau inkonsistensi data. Proses agregasi juga dilakukan untuk menyederhanakan beberapa fitur.
4. Analisis Data Eksploratif (EDA): Menggali pola-pola dan insight dalam data dengan analisis statistik deskriptif dan visualisasi, termasuk hubungan antara fitur dan target label (status mahasiswa).
5. Data Pre-Processing: Meliputi proses pembagian data menjadi data latih dan data uji (train-test split), encoding variabel kategorikal, scaling data numerik, serta penerapan Principal Component Analysis (PCA) untuk reduksi fitur.
6. Pemodelan: Membangun model klasifikasi menggunakan beberapa algoritma yaitu: Support Vector Machine (SVM), Decision Tree, Random Forest, dan XGBoost.
7. Evaluasi Model: Melakukan evaluasi terhadap performa masing-masing model dengan menggunakan metrik evaluasi klasifikasi seperti akurasi, precision, recall, F1-score, dan confusion matrix untuk menentukan model terbaik yang akan digunakan pada sistem prediksi.


### Persiapan

Sumber data: [https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md]

Setup environment: 
```
# Setup Env - Google Colab (Versi Python yang digunakan: 3.10+ (default Google Colab))
# Install library 
!pip install -q numpy pandas matplotlib seaborn scikit-learn joblib xgboost

# Import Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing & Model Selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

# Classification Algorithms
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Model Saving
import joblib
```

## Business Dashboard
![image](https://github.com/user-attachments/assets/97d826fe-8be7-4927-b37c-1e1ca8e73e6e)

Secara keseluruhan, dari dashboard di atas, dapat dilihat jika total mahasiswa pada institut sebanyak 4424 mahasiswa, yang terdiri dari 1556 mahasiswa laki-laki dan 2868 mahasiswa perempuan. Dari dashboard, dapat dilihat jika:
1. Dari bar chart pertama, dapat dilihat jika mahasiswa didominasi oleh Graduate atau yang sudah lulus dengan angka 2209, lalu diikuti Dropout sebanyak 1421, dan Erolled atau yang terdaftar sebanyak 794. Dari angka ini, jumlah Dropout terbilang sangat banyak, berbanding jauh dengan jumlah yang terdaftar.
2. Pada bar chart kedua, nilai rata-rata yang diperoleh mahasiswa dalam semester yang ditempuh sangat berpengaruh terhadap kelulusannya. Semakin tinggi nilai yang didapat, maka semakin tinggi juga potensi kelulusannya, dibuktikan dari nilai Graduate yang sejalan dengan tingginya rata-rata nilai yang didapat. Sementara itu, nilai Dropout cukup selisih jauh dari nilai Graduate yang rata-rata di angka 27.9, nilai mahasiswa Dropout memiliki rata-rata 9.3.
3. Pada pie chart, dapat dilihat jika mahasiswa Graduate mendominasi hampir setengah populasi, yaitu 49.9%. Sementara itu, mahasiswa Dropout sebesar 32.1%. Dan terakhir, yang paling sedikit, mahasiswa terdaftar sebesar 17.9%.
4. Dari hasil scatterplot, dapat dilihat keberhasilan kelulusan mahasiswa dipengaruhi oleh semakin banyaknya total SKS yang diambil dengan total SKS yang lulus. Hubungan antara total SKS yang diambil dan yang lulus terlihat berbanding lurus. Sehingga penyebab Dropout dikarenakan sedikitnya SKS yang diambil dan SKS yang dapat lulus. 
5. Scatterplot kedua menunjukkan demikian, hubungan antara total SKS yang diaveluasi dan yang tidak dievaluasi. Dari hasil, dapat dilihat hubungan kedua fitur ini adalah berbanding terbalik. Pada Graduate, semakin banyak evaluasi yang diikuti (ujian/praktikum/tugas), maka semakin tinggi prediksi kelulusannya. Begitu juga dengan Dropout, yang menunjukkan tingginya total SKS yang tidak dievaluasi. Pada yang terdaftar, nilai total yang tidak dievaluasi sangat sedikit, ini dapat menjadi potensi untuk mengarahkan meningkatkan nilai evaluasinya.
6. Hasil visualisasi terakhir, menunjukkan jika Nursing memiliki banyak siswa yang berhasil graduate, begitu juga pada Social Servis dan Journalism. Ini berbeda dengan Management yang terlihat lebih banyak yang dropout daripada graduate, sama seperti Informatics Engeineering. Secara keseluruhan, datanya terlihat cukup acak dan bermacam-macam di program studi yang ada.

Link Dashboard: [https://lookerstudio.google.com/reporting/76f11265-a200-4efa-be52-fc21c93e9037/page/Cg3JF]

## Menjalankan Sistem Machine Learning
Sistem machine learning ini telah dikembangkan dalam bentuk aplikasi web interaktif menggunakan Streamlit, yang memungkinkan pengguna untuk memasukkan data mahasiswa dan mendapatkan prediksi status kelulusan mahasiswa tersebut. Cara menjalankannya adalah seperti berikut:

1. Instalasi Library
Pastikan semua library berikut telah terinstal:
```
pip install streamlit pandas scikit-learn joblib
```
2. Menjalankan Aplikasi
Jalankan perintah berikut di terminal atau command prompt pada direktori tempat file aplikasi disimpan:
```
streamlit run student_predict_app.py
```
3. Input Data
Setelah aplikasi berjalan, pengguna dapat mengisi data mahasiswa melalui form yang tersedia, lalu menekan tombol Predict untuk mendapatkan hasil prediksi.

Prototype aplikasi dapat diakses melalui link berikut: [https://student-dropout-predict-app.streamlit.app/]

## Conclusion
Berdasarkan hasil analisis data mahasiswa dari Jaya Jaya Institut serta penerapan algoritma machine learning, diperoleh beberapa insight penting yang dapat membantu pihak institusi dalam menangani permasalahan dropout. Faktor-faktor seperti jumlah kelas yang diambil dan diluluskan (Total_enrolled dan Total_approved), nilai rata-rata (Avg_grade), dan usia saat mendaftar (Age_at_Enrollment) memiliki korelasi yang kuat terhadap potensi meningkatnya kelulusan mahasiswa.

Model machine learning terbaik yang berhasil dibangun adalah Random Forest, dengan performa evaluasi yang unggul dibandingkan model lainnya. Aplikasi ini telah diimplementasikan dalam bentuk prototype berbasis web menggunakan Streamlit, yang dapat membantu institusi dalam melakukan prediksi dropout secara cepat dan interaktif.

Dari visualisasi dan eksplorasi data, ditemukan bahwa mahasiswa yang usinya lebih muda, banyak mengambil kelas, sering menghadiri evaluasi, dan memiliki nilai yang tinggi cenderung lulus. Sebaliknya, mahasiswa yang mengambil jumlah kelas sedikit, nilai cukup rendah, dan berusia di atas 25 tahun saat mendaftar lebih rentan mengalami dropout. Fitur-fitur seperti Tuition_fees_up_to_date (mahasiswa mendapat biaya terkini), Scholarship_holder (pemegang beassiwa), dan Previous_qualification_grade (nilai kualifikasi di pendidikan terakhir) juga memberikan kontribusi signifikan dalam membedakan profil mahasiswa yang lulus dan yang dropout.

### Rekomendasi Action Items
Berikut beberapa recommendation action items yang dapat dilakukan, yaitu:
1. Fokus Intervensi pada Mahasiswa Berusia >25 Tahun
Mahasiswa yang lebih tua saat pendaftaran (enrolled) cenderung memiliki risiko dropout lebih tinggi, sehingga diperlukan pembuatan pendekatan pendampinganatau fleksibilitas waktu belajar bagi kelompok usia ini.
2. Evaluasi dan Bimbingan Berdasarkan Nilai dan Kehadiran
Mahasiswa dengan nilai rendah dan kehadiran evaluasi yang buruk harus mendapat perhatian seperti mentoring, kelas remedial, atau pembinaan akademik yang aktif.
3. Penguatan Kebijakan Biaya Pendidikan dan Beasiswa
Mahasiswa yang tidak up-to-date dalam pembayaran biaya kuliah lebih rentan dropout. Institusi dapat mempertimbangkan skema beasiswa atau cicilan untuk kelompok ini, dengan memperbanyak program kerja sama beasiswa atau memberikan keringanan biaya terhadap mahasiswa yang cenderung memiliki ekonomi rendah.
4. Pemantauan SKS dan Kinerja Akademik Mahasiswa
Pihak akademik dapat memonitor aktif terhadap jumlah SKS yang diambil dan yang berhasil diluluskan, ini supaya bisa menjadi indikator kesehatan akademik, sehingga pendeteksian lebih awal dapat dilakukan pada mahasiswa yang cenderung berpotensi dropout.
