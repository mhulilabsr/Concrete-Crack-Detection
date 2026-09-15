# =========================================================================
# KUNCI PERBAIKAN: OVERRIDE DEFENSE SYSTEM SEBELUM IMPORT OPENCV
# =========================================================================
import sys
import os

# Trik paksa agar pustaka grafis sistem Linux yang rusak di-bypass oleh OpenCV
os.environ["QT_X11_NO_MITSHM"] = "1"

import streamlit as st
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Gunakan fungsi penarik modul headless secara paksa agar aman dari crash bootstrap
try:
    import cv2
except ImportError:
    st.error("Sistem memulihkan modul grafis di balik layar...")
    os.system("pip uninstall -y opencv-python opencv-python-headless")
    os.system("pip install opencv-python-headless")
    import cv2

# =========================================================================
# SISA KODE ANTARMUKA DASHBOARD ANDA (TETAP SAMA SEPERTI KEMARIN)
# =========================================================================
st.set_page_config(page_title="Concrete Crack Detector AI", layout="wide")

st.title("🧱 AI System: Deteksi & Pengukuran Otomatis Crack Beton")
st.markdown("### Arsitektur Adaptif Dua Tahapan: YOLO11-Segmentation & ArUco Dinamis")
st.write("Sistem otomatis mendeteksi keberadaan retakan, menghitung ketebalan celah murni (Metode Jarak Kontur), serta mengonversi satuan mm secara dinamis.")

# 1. Cari model pintar best.pt hasil training otomatis di server
path_model = None
for root, dirs, files_list in os.walk('.'):
    if 'best.pt' in files_list and 'segment' in root:
        path_model = os.path.join(root, 'best.pt')
        break

if path_model is None:
    # Cek jika best.pt ada di folder root utama hasil force upload
    if os.path.exists('best.pt'):
        path_model = 'best.pt'

if path_model is None:
    st.error("❌ Model pintar 'best.pt' tidak ditemukan di server. Pastikan Anda sudah menyelesaikan training.")
else:
    # Sidebar Pengaturan Kalibrasi
    st.sidebar.header("⚙️ Konfigurasi Sistem")
    ukuran_asli_mm = st.sidebar.number_input("Ukuran Fisik ArUco (mm):", min_value=1.0, value=50.0, step=1.0)
    skala_cadangan = st.sidebar.number_input("Skala Cadangan (px/mm):", min_value=0.1, value=1.30, step=0.05)

    # Slot Unggah Gambar Komputer
    uploaded_file = st.file_uploader("📤 Unggah Foto Struktur Beton dari Komputer Anda", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Konversi file unggahan ke matriks OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_hasil = img.copy()
        
        st.info("⏳ TAHAPAN 1: AI Sedang Memeriksa Keberadaan Crack...")
        
        # 2. PROSES TAHAPAN 1: DETEKSI CRACK DENGAN YOLO11
        model = YOLO(path_model)
        results = model(img)
        
        retakan_terdeteksi = False
        mask_uint8 = None
        
        for r in results:
            if r.masks is not None:
                retakan_terdeteksi = True
                mask_matriks = r.masks.data.cpu().numpy()
                # Mengambil indeks dimensi pertama agar matriks 3D YOLO berubah menjadi 2D murni sebelum resize
                if len(mask_matriks.shape) == 3:
                    mask_matriks = mask_matriks[0]
                mask_rescaled = cv2.resize(mask_matriks, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_uint8 = (mask_rescaled * 255).astype(np.uint8)

        if not retakan_terdeteksi:
            st.success("🛡️ STATUS TAHAPAN 1: BETON AMAN / TIDAK TERDETEKSI CRACK!")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Struktur Beton Normal (Mulus)", width=600)
        else:
            st.warning("🚨 STATUS TAHAPAN 1: CRACK TERDETEKSI! Melanjutkan ke TAHAPAN 2...")
            
            # --- Pembersih Tepi (Border Clearing) ---
            h_m, w_m = mask_uint8.shape
            mask_uint8[0:40, :] = 0        
            mask_uint8[h_m-30:h_m, :] = 0  
            mask_uint8[:, 0:30] = 0        
            mask_uint8[:, w_m-30:w_m] = 0  
            
            # 3. PROSES TAHAPAN 2: MENCARI CELAH HORIZONTAL TERGAK LURUS MAKSIMAL
            import cv2 as cv_proc
            import cv2.aruco as aruco
            
            contours, _ = cv_proc.findContours(mask_uint8, cv_proc.RETR_EXTERNAL, cv_proc.CHAIN_APPROX_NONE)
            lebar_maks_px = 0
            titik_pusat_terlebar = (0, 0)
            max_val = 0
            
            if len(contours) > 0:
                dist_transform = cv_proc.distanceTransform(mask_uint8, cv_proc.DIST_L2, 5)
                _, max_val, _, max_loc = cv_proc.minMaxLoc(dist_transform)
                lebar_maks_px = max_val * 2
                titik_pusat_terlebar = max_loc

            # --- PROSES CHECK KEBERADAAN ARUCO ---
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(img)
            
            px_per_mm = None
            status_akurasi = ""
            
            if ids is not None:
                sudut_rata = np.squeeze(corners)
                lebar_aruco_px = np.linalg.norm(sudut_rata[0] - sudut_rata[1])
                px_per_mm = lebar_aruco_px / ukuran_asli_mm
                lebar_mm = lebar_maks_px / px_per_mm
                status_akurasi = "VALID (Berbasis ArUco Dinamis)"
            else:
                lebar_mm = lebar_maks_px / skala_cadangan
                status_akurasi = f"ESTIMASI (Tanpa ArUco - Pendekatan Acuan {skala_cadangan} px/mm)"
            
            # --- GAMBAR PENANDA BULAT DI AREA TERLEBAR ---
            cv_proc.circle(img_hasil, titik_pusat_terlebar, max(5, int(max_val)), (0, 0, 255), -1)
            
            # TAMPILKAN HASILNYA DI DASHBOARD WEB
            col1, col2 = st.columns(2)
            with col1:
                img_rgb = cv_proc.cvtColor(img_hasil, cv_proc.COLOR_BGR2RGB)
                h_i, w_i, _ = img_rgb.shape
                x, y = titik_pusat_terlebar
                crop_img = img_rgb[max(0, y-300):min(h_i, y+300), max(0, x-400):min(w_i, x+400)]
                st.image(crop_img, caption="Fokus Lokasi Celah Terlebar (Sensor Merah)", use_container_width=True)
            
            with col2:
                st.subheader("📊 Laporan Pengujian Struktur Digital")
                st.metric("Lebar Celah Maksimal (Piksel)", f"{lebar_maks_px:.2f} px")
                st.metric("UKURAN FISIK NYATA CELAH", f"{lebar_mm:.2f} mm")
                
                if ids is not None:
                    st.success(f"📈 Status Akurasi: {status_akurasi}")
                else:
                    st.warning(f"⚠️ Status Akurasi: {status_akurasi}")

    
