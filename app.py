import io
import os
import tempfile
import urllib.request
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
from PIL import Image, ImageDraw


st.set_page_config(page_title="Face Similarity Checker", page_icon="🧑‍🤝‍🧑", layout="wide")

if os.environ.get("FACE_SIMILARITY_OFFLINE") == "1":
    deepface_home = os.environ.get("DEEPFACE_HOME", os.path.expanduser("~"))
    weights_dir = os.path.join(deepface_home, ".deepface", "weights")
    required_weights = ("arcface_weights.h5", "retinaface.h5")
    missing_weights = [
        filename for filename in required_weights
        if not os.path.isfile(os.path.join(weights_dir, filename))
    ]
    cascade_path = os.path.join(
        os.path.dirname(cv2.__file__),
        "data",
        "haarcascade_frontalface_default.xml",
    )
    if missing_weights or not os.path.isfile(cascade_path):
        missing_assets = missing_weights.copy()
        if not os.path.isfile(cascade_path):
            missing_assets.append("haarcascade_frontalface_default.xml")
        st.error(
            "Aset aplikasi portable tidak lengkap; file berikut tidak ditemukan: "
            + ", ".join(missing_assets)
            + ". Salin ulang seluruh folder aplikasi."
        )
        st.stop()


def save_uploaded_file(uploaded_file, prefix: str) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getvalue())
    temp.close()
    return temp.name


@st.cache_data(show_spinner=False)
def detect_face_areas(image_bytes: bytes) -> List[Tuple[int, int, int, int, float]]:
    faces = DeepFace.extract_faces(
        img_path=io.BytesIO(image_bytes),
        detector_backend="retinaface",
        enforce_detection=True,
        align=False,
    )
    return [
        (
            int(face["facial_area"]["x"]),
            int(face["facial_area"]["y"]),
            int(face["facial_area"]["w"]),
            int(face["facial_area"]["h"]),
            float(face["confidence"]),
        )
        for face in faces
    ]


def render_image_preview(uploaded_file, caption: str, show_face_boxes: bool) -> None:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if show_face_boxes:
        try:
            with st.spinner(f"Mendeteksi wajah pada {caption}..."):
                face_areas = detect_face_areas(image_bytes)
        except FaceNotDetected:
            st.warning(f"Tidak ada wajah yang terdeteksi pada {caption}.")
        except Exception as exc:
            st.error(f"Gagal mendeteksi wajah pada {caption}: {exc}")
        else:
            draw = ImageDraw.Draw(image)
            line_width = max(2, round(min(image.size) / 200))
            for x, y, width, height, _confidence in face_areas:
                draw.rectangle(
                    (x, y, x + width, y + height),
                    outline="#00e676",
                    width=line_width,
                )
            st.caption(f"{len(face_areas)} wajah terdeteksi dengan RetinaFace.")

    st.image(image, caption=caption, width="stretch")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if np.isclose(denom, 0.0):
        return 0.0
    return float(np.dot(a, b) / denom)


def calculate_comprehensive_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> Dict[str, float]:
    emb1 = np.asarray(embedding1, dtype=np.float64)
    emb2 = np.asarray(embedding2, dtype=np.float64)
    if emb1.ndim != 1 or emb2.ndim != 1 or emb1.shape != emb2.shape or emb1.size == 0:
        raise ValueError("Embedding harus berupa dua vektor 1D dengan ukuran yang sama.")
    if not np.all(np.isfinite(emb1)) or not np.all(np.isfinite(emb2)):
        raise ValueError("Embedding mengandung nilai yang tidak valid.")

    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if np.isclose(norm1, 0.0) or np.isclose(norm2, 0.0):
        raise ValueError("Embedding tidak boleh berupa vektor nol.")

    emb1_norm = emb1 / (norm1 + 1e-8)
    emb2_norm = emb2 / (norm2 + 1e-8)
    cosine_sim = float(np.dot(emb1_norm, emb2_norm))
    euclidean_sim = float(1.0 / (1.0 + np.linalg.norm(emb1_norm - emb2_norm)))
    manhattan_sim = float(
        1.0 - np.sum(np.abs(emb1_norm - emb2_norm)) / emb1_norm.size
    )

    if np.isclose(np.std(emb1_norm), 0.0) or np.isclose(np.std(emb2_norm), 0.0):
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(emb1_norm, emb2_norm)[0, 1])
    corr_sim = (correlation + 1.0) / 2.0 if np.isfinite(correlation) else 0.0
    metrics = {
        "cosine": cosine_sim,
        "euclidean": euclidean_sim,
        "correlation": float(corr_sim),
        "manhattan": manhattan_sim,
    }
    primary_similarity = (
        cosine_sim * 0.45
        + euclidean_sim * 0.25
        + corr_sim * 0.15
        + manhattan_sim * 0.15
    )

    return {
        "primary_similarity": float(primary_similarity),
        "confidence_score": float(
            1.0 - np.std([cosine_sim, euclidean_sim, corr_sim])
        ),
        **metrics,
    }


def calculate_threshold_percent(threshold_value: float) -> float:
    return max(0.0, min(100.0, (1.0 - float(threshold_value)) * 100.0))


def match_recommendation(similarity_percent: float, threshold_value: float, verified: bool) -> str:
    threshold_percent = calculate_threshold_percent(threshold_value)
    if verified and similarity_percent >= threshold_percent:
        if similarity_percent >= 90:
            return "Skor hasil transformasi jarak tergolong tinggi dan pasangan lolos threshold model."
        return "Pasangan lolos threshold model; tinjau juga kualitas dan kondisi kedua foto."
    if similarity_percent >= threshold_percent * 0.9:
        return "Hasil relatif dekat dengan threshold, tetapi belum cukup untuk dinyatakan lolos oleh model."
    return "Pasangan tidak lolos threshold model pada pemeriksaan ini."


def explain_model_decision(distance: float, threshold_value: float, verified: bool) -> str:
    distance = float(distance)
    threshold_value = float(threshold_value)
    if verified:
        return (
            f"Jarak model {distance:.4f} berada pada atau di bawah threshold "
            f"{threshold_value:.4f}, sehingga pasangan dinyatakan lolos oleh ArcFace. "
            "Ini adalah hasil pencocokan model, bukan kepastian identitas."
        )
    return (
        f"Jarak model {distance:.4f} berada di atas threshold "
        f"{threshold_value:.4f}, sehingga pasangan tidak lolos pada pengaturan model ini. "
        "Hasil dapat dipengaruhi sudut, ekspresi, pencahayaan, dan kualitas foto."
    )


def render_threshold_visual(current_percent: float, threshold_value: float) -> None:
    current_percent = max(0.0, min(100.0, float(current_percent)))
    threshold_percent = calculate_threshold_percent(threshold_value)
    threshold_percent = max(0.0, min(100.0, threshold_percent))
    color = "#00c853" if current_percent >= threshold_percent else "#ffb000"

    st.markdown(
        f"""
        <div style="padding: 1rem; border: 1px solid #3d3d3d; border-radius: 0.85rem; background: #0e1117; margin-top: 0.75rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.9rem; color: #EAEAEA;">Perbandingan dengan threshold</span>
                <span style="font-size: 0.9rem; color: #EAEAEA;">{current_percent:.2f}%</span>
            </div>
            <div style="position: relative; height: 22px; background: linear-gradient(90deg, #1f2937 0%, #2b2d42 50%, #0f172a 100%); border-radius: 999px; overflow: hidden;">
                <div style="position: absolute; inset: 0 auto 0 0; width: {current_percent}%; background: linear-gradient(90deg, {color} 0%, #66ffa6 100%); border-radius: 999px;"></div>
                <div style="position: absolute; top: -3px; left: {threshold_percent}%; width: 3px; height: 28px; background: #FF4B4B; box-shadow: 0 0 10px rgba(255,75,75,0.9); border-radius: 999px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top: 0.5rem; font-size: 0.74rem; color: #BDBDBD;">
                <span>0%</span>
                <span>Threshold: {threshold_percent:.2f}%</span>
                <span>100%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_experimental_score_guide() -> None:
    with st.expander("Panduan lengkap membaca skor eksperimental"):
        st.markdown(
            """
            **Skor gabungan eksperimental** merangkum empat perhitungan yang diterapkan
            pada embedding ArcFace dari kedua foto. Nilai ini adalah indeks internal
            rumus saat ini, **bukan persentase, probabilitas identitas, atau keputusan
            lolos/tidak lolos**. Belum ada threshold yang divalidasi untuk skor ini.

            ### Arti angka yang ditampilkan

            - **Skor gabungan (experimental):** gabungan empat metrik dengan bobot tetap.
              Nilai lebih tinggi berarti metrik gabungan menghasilkan skor lebih tinggi
              pada rumus ini, tetapi tidak berarti persentase kemungkinan kedua foto
              menampilkan orang yang sama. Jangan bandingkan nilainya dengan threshold
              DeepFace atau menafsirkannya memakai batas seperti 0,5 atau 0,8.
            - **Konsistensi metrik (experimental):** dihitung sebagai `1 - simpangan
              baku` dari skor cosine, Euclidean, dan korelasi yang sudah dipetakan.
              Ini hanya menunjukkan seberapa berdekatan tiga angka tersebut menurut
              rumus sederhana ini. Nilai tinggi bukan bukti bahwa hasil benar; metrik
              keempat (Manhattan) tidak ikut dihitung di sini.

            ### Empat komponen skor gabungan

            1. **Cosine similarity (bobot 45%)** membandingkan arah dua embedding.
               Makin dekat ke 1 berarti arahnya makin searah; 0 berarti tegak lurus;
               nilai negatif berarti arahnya berlawanan.
            2. **Euclidean similarity (bobot 25%)** adalah `1 / (1 + jarak Euclidean)`
               setelah embedding dinormalisasi. Makin tinggi berarti jarak makin kecil.
               Komponen ini berkaitan erat dengan cosine pada vektor ternormalisasi,
               jadi keduanya bukan dua bukti independen.
            3. **Korelasi (bobot 15%)** mengukur kemiripan pola nilai setelah masing-masing
               embedding dipusatkan terhadap rata-ratanya, lalu dipetakan ke rentang
               0–1. Angka 1 berarti pola sangat searah, 0,5 sekitar tidak berkorelasi,
               dan 0 berarti berkorelasi negatif. Ini bukan probabilitas.
            4. **Manhattan similarity (bobot 15%)** dihitung dari selisih absolut rata-rata
               per dimensi setelah normalisasi. Pada embedding berdimensi tinggi,
               rumus ini dapat memberi angka dekat 1 secara bawaan; karena itu, nilainya
               bisa terlihat tinggi walaupun metrik lain lebih rendah.

            Rumus gabungannya adalah
            `0,45 × cosine + 0,25 × Euclidean + 0,15 × korelasi + 0,15 × Manhattan`.
            Rentang komponen dan karakteristiknya berbeda, sehingga hasil gabungan
            **tidak boleh langsung diperlakukan sebagai skor terkalibrasi**.

            ### Cara memakai hasil dengan aman

            - Lihat skor eksperimental sebagai informasi tambahan, bukan dasar keputusan.
            - Jika skor gabungan dan komponen lain tampak berbeda arah, anggap hasil
              eksperimental tidak meyakinkan; periksa kembali foto dan hasil ArcFace.
            - Jika konsistensi tinggi, artinya hanya tiga metrik yang dihitung tampak
              berdekatan—bukan berarti ketiganya benar atau identitas terkonfirmasi.
            - Gunakan **Verified**, **Distance**, dan **Threshold** dari DeepFace/ArcFace
              untuk membaca keputusan aplikasi saat ini. `Verified` tidak memakai skor
              eksperimental.
            - Sudut, ekspresi, pencahayaan, blur, kualitas gambar, dan deteksi wajah
              dapat memengaruhi embedding. Hasil model bukan kepastian identitas.

            Agar skor gabungan layak dipakai untuk keputusan, rumus dan threshold perlu
            diuji serta dikalibrasi pada kumpulan pasangan foto representatif yang
            memiliki label benar/salah.
            """
        )


def compare_two_images(image_a_path: str, image_b_path: str) -> Dict:
    embedding_score, comprehensive_metrics, deepface_result = detect_face_and_compute_similarity(
        image_a_path,
        image_b_path,
    )
    similarity_percent = round(max(0.0, min(1.0, embedding_score)) * 100, 2)
    deepface_percent = round(max(0.0, min(1.0, 1 - float(deepface_result.get("distance", 0)))) * 100, 2)
    threshold_value = float(deepface_result.get("threshold", 0.0) or 0.0)

    return {
        "Embedding Similarity %": similarity_percent,
        "DeepFace Similarity %": deepface_percent,
        "Comprehensive Similarity (experimental)": comprehensive_metrics["primary_similarity"],
        "Metric Consistency (experimental)": comprehensive_metrics["confidence_score"],
        "Distance": deepface_result.get("distance", "-"),
        "Threshold": deepface_result.get("threshold", "-"),
        "Verified": bool(deepface_result.get("verified")),
        "Recommendation": match_recommendation(deepface_percent, threshold_value, bool(deepface_result.get("verified"))),
        "Interpretation": explain_model_decision(
            float(deepface_result["distance"]),
            threshold_value,
            bool(deepface_result.get("verified")),
        ),
        "Threshold Visual %": round(calculate_threshold_percent(threshold_value), 2),
    }


def compare_batch_images(image_paths, image_names):
    from itertools import combinations

    if len(image_paths) != len(image_names):
        raise ValueError("Jumlah path foto harus sama dengan jumlah nama foto.")

    results = []
    path_pairs = combinations(image_paths, 2)
    name_pairs = combinations(image_names, 2)
    for index, ((img_a_path, img_b_path), (img_a_name, img_b_name)) in enumerate(
        zip(path_pairs, name_pairs),
        start=1,
    ):
        comparison = compare_two_images(img_a_path, img_b_path)
        results.append(
            {
                "No": index,
                "Foto 1": img_a_name,
                "Foto 2": img_b_name,
                "Embedding Similarity %": comparison["Embedding Similarity %"],
                "DeepFace Similarity %": comparison["DeepFace Similarity %"],
                "Comprehensive Similarity (experimental)": comparison[
                    "Comprehensive Similarity (experimental)"
                ],
                "Metric Consistency (experimental)": comparison[
                    "Metric Consistency (experimental)"
                ],
                "Distance": comparison["Distance"],
                "Threshold": comparison["Threshold"],
                "Verified": "Ya" if comparison["Verified"] else "Tidak",
                "Rekomendasi": comparison["Recommendation"],
                "Interpretasi": comparison["Interpretation"],
            }
        )
    return results


def ensure_opencv_haarcascade() -> str:
    cascade_name = "haarcascade_frontalface_default.xml"
    data_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
    target = os.path.join(data_dir, cascade_name)

    if os.path.exists(target):
        return target

    if os.environ.get("FACE_SIMILARITY_OFFLINE") == "1":
        raise FileNotFoundError(
            f"File OpenCV {cascade_name} tidak ditemukan di paket portable."
        )

    os.makedirs(data_dir, exist_ok=True)
    url = f"https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{cascade_name}"
    with urllib.request.urlopen(url, timeout=30) as response, open(target, "wb") as file:
        file.write(response.read())

    return target


def detect_face_and_compute_similarity(
    image_a_path: str,
    image_b_path: str,
) -> Tuple[float, Dict[str, float], Dict]:
    embed_a = DeepFace.represent(
        img_path=image_a_path,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=True,
    )
    embed_b = DeepFace.represent(
        img_path=image_b_path,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=True,
    )

    embedding_a = np.asarray(embed_a[0]["embedding"], dtype=np.float32)
    embedding_b = np.asarray(embed_b[0]["embedding"], dtype=np.float32)
    embedding_score = cosine_similarity(embedding_a, embedding_b)
    comprehensive_metrics = calculate_comprehensive_similarity(embedding_a, embedding_b)

    deepface_result = DeepFace.verify(
        img1_path=image_a_path,
        img2_path=image_b_path,
        model_name="ArcFace",
        detector_backend="retinaface",
        distance_metric="cosine",
        enforce_detection=True,
    )

    return embedding_score, comprehensive_metrics, deepface_result


st.title("Face Similarity Checker")
st.caption("Aplikasi untuk mengecek similaritas wajah menggunakan embedding model dan deepface similarity")

compare_mode = st.radio(
    "Mode pengecekan",
    ["Single Compare", "One vs Many", "Batch Compare", "Review Visual"],
    index=0,
    horizontal=True,
)
show_face_boxes = st.checkbox(
    "Tampilkan kotak deteksi wajah",
    value=False,
    help="Menggunakan RetinaFace untuk menandai setiap wajah yang terdeteksi. "
    "Fitur ini membantu tinjauan visual dan tidak mengidentifikasi seseorang.",
)
if compare_mode != "Review Visual":
    render_experimental_score_guide()

if compare_mode == "Single Compare":
    col1, col2 = st.columns(2)
    with col1:
        image_a = st.file_uploader("Upload foto 1", type=["jpg", "jpeg", "png"])
    with col2:
        image_b = st.file_uploader("Upload foto 2", type=["jpg", "jpeg", "png"])

    if image_a and image_b:
        left_col, right_col = st.columns(2)
        with left_col:
            render_image_preview(image_a, "Foto 1", show_face_boxes)
        with right_col:
            render_image_preview(image_b, "Foto 2", show_face_boxes)

        if st.button("Cek Similarity", width="stretch"):
            temp_a = save_uploaded_file(image_a, "img_a")
            temp_b = save_uploaded_file(image_b, "img_b")

            try:
                with st.spinner("Sedang menghitung embedding dan similarity wajah..."):
                    embedding_score, comprehensive_metrics, deepface_result = (
                        detect_face_and_compute_similarity(temp_a, temp_b)
                    )

                similarity_percent = round(max(0.0, min(1.0, embedding_score)) * 100, 2)
                deepface_percent = round(max(0.0, min(1.0, 1 - float(deepface_result.get("distance", 0)))) * 100, 2)

                st.success("Analisis selesai")

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Embedding Similarity", f"{similarity_percent}%")
                with metric_col2:
                    st.metric("DeepFace Similarity", f"{deepface_percent}%")
                with metric_col3:
                    st.metric("Verified", "Ya" if deepface_result.get("verified") else "Tidak")
                with metric_col4:
                    st.metric(
                        "Skor gabungan (eksperimental)",
                        f"{comprehensive_metrics['primary_similarity']:.4f}",
                    )
                st.caption(
                    f"Konsistensi metrik eksperimental: "
                    f"{comprehensive_metrics['confidence_score']:.4f}. "
                    "Skor gabungan dan konsistensi bukan probabilitas atau threshold; "
                    "status Verified tetap ditentukan oleh DeepFace ArcFace."
                )
                st.markdown("#### Penjelasan skor gabungan")
                st.info(
                    f"Skor gabungan **{comprehensive_metrics['primary_similarity']:.4f}** "
                    "adalah hasil penjumlahan empat skor komponen dengan bobot berbeda. "
                    "Angka ini **bukan persentase kemiripan atau peluang identitas** "
                    "(misalnya, 0,5174 bukan berarti 51,74%). "
                    "Tabel berikut menunjukkan nilai setiap komponen dan sumbangannya "
                    "ke hasil gabungan. Komponen yang lebih dekat ke 1 memberi skor "
                    "lebih besar menurut rumus, tetapi skor ini belum memiliki batas "
                    "lolos/tidak lolos yang tervalidasi."
                )
                component_rows = [
                    {
                        "Komponen": "Cosine",
                        "Nilai komponen": comprehensive_metrics["cosine"],
                        "Bobot": 0.45,
                        "Sumbangan ke skor": comprehensive_metrics["cosine"] * 0.45,
                        "Makna": "Membandingkan arah kedua embedding.",
                    },
                    {
                        "Komponen": "Euclidean",
                        "Nilai komponen": comprehensive_metrics["euclidean"],
                        "Bobot": 0.25,
                        "Sumbangan ke skor": comprehensive_metrics["euclidean"] * 0.25,
                        "Makna": "Skor jarak; makin dekat embedding, nilainya makin besar.",
                    },
                    {
                        "Komponen": "Korelasi",
                        "Nilai komponen": comprehensive_metrics["correlation"],
                        "Bobot": 0.15,
                        "Sumbangan ke skor": comprehensive_metrics["correlation"] * 0.15,
                        "Makna": "Kemiripan pola nilai embedding yang dipetakan ke 0–1.",
                    },
                    {
                        "Komponen": "Manhattan",
                        "Nilai komponen": comprehensive_metrics["manhattan"],
                        "Bobot": 0.15,
                        "Sumbangan ke skor": comprehensive_metrics["manhattan"] * 0.15,
                        "Makna": "Berdasarkan rata-rata selisih absolut per dimensi.",
                    },
                ]
                st.dataframe(component_rows, hide_index=True, width="stretch")
                st.caption(
                    "Rumus: 0,45 × Cosine + 0,25 × Euclidean + 0,15 × Korelasi + "
                    "0,15 × Manhattan. Cosine dan Euclidean saling berkaitan; "
                    "Manhattan dapat terlihat tinggi secara bawaan pada embedding "
                    "berdimensi besar. Karena skala dan sifat metrik berbeda, baca "
                    "rincian ini sebagai penjelasan rumus, bukan ukuran akurasi."
                )

                threshold_value = float(deepface_result.get("threshold", 0.0) or 0.0)
                status_text = match_recommendation(deepface_percent, threshold_value, bool(deepface_result.get("verified")))

                if deepface_result.get("verified"):
                    st.success(status_text)
                elif deepface_percent >= calculate_threshold_percent(threshold_value) * 0.9:
                    st.warning(status_text)
                else:
                    st.error(status_text)

                st.markdown("### Penjelasan hasil")
                st.write(
                    explain_model_decision(
                        float(deepface_result["distance"]),
                        threshold_value,
                        bool(deepface_result.get("verified")),
                    )
                )
                st.caption(
                    "Persentase similarity adalah skor yang diturunkan dari jarak model, bukan probabilitas "
                    "bahwa dua foto adalah orang yang sama."
                )
                render_threshold_visual(deepface_percent, threshold_value)

                st.subheader("Detail Hasil")
                details = {
                    "Model": deepface_result.get("model", "ArcFace"),
                    "Distance": deepface_result.get("distance", "-"),
                    "Threshold": deepface_result.get("threshold", "-"),
                    "Threshold Visual (%)": round(calculate_threshold_percent(threshold_value), 2),
                    "Embedding Similarity": round(embedding_score, 6),
                    "DeepFace Similarity": round(1 - float(deepface_result.get("distance", 0)), 6),
                    "Comprehensive Similarity (experimental)": comprehensive_metrics[
                        "primary_similarity"
                    ],
                    "Metric Consistency (experimental)": comprehensive_metrics[
                        "confidence_score"
                    ],
                    "Experimental metric components": {
                        name: comprehensive_metrics[name]
                        for name in ("cosine", "euclidean", "correlation", "manhattan")
                    },
                }
                st.json(details)

                st.caption("Catatan: semakin dekat ke 100% maka wajah semakin mirip, dan garis merah menunjukkan ambang threshold model.")

            except Exception as exc:
                st.error(f"Terjadi kesalahan saat memproses wajah: {exc}")
            finally:
                for temp_path in [temp_a, temp_b]:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        st.info("Silakan upload dua foto wajah untuk memulai pengecekan similarity.")
elif compare_mode == "One vs Many":
    primary_image = st.file_uploader("Upload foto utama", type=["jpg", "jpeg", "png"])
    compared_images = st.file_uploader(
        "Upload foto pembanding (bisa lebih dari satu)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if primary_image and compared_images:
        render_image_preview(primary_image, "Foto utama", show_face_boxes)

        preview_cols = st.columns(min(4, len(compared_images)))
        for idx, image in enumerate(compared_images):
            with preview_cols[idx % min(4, len(compared_images))]:
                render_image_preview(image, image.name, show_face_boxes)

        if st.button("Bandingkan dengan semua foto", width="stretch"):
            temp_primary = save_uploaded_file(primary_image, "primary")
            temp_compared = [save_uploaded_file(file, f"compare_{idx}") for idx, file in enumerate(compared_images)]

            try:
                if len(temp_compared) == 0:
                    st.warning("Harap upload foto pembanding terlebih dahulu.")
                else:
                    with st.spinner("Membandingkan foto utama terhadap semua foto lain..."):
                        results = []
                        for idx, candidate_path in enumerate(temp_compared, start=1):
                            comparison = compare_two_images(temp_primary, candidate_path)
                            results.append(
                                {
                                    "No": idx,
                                    "Foto utama": primary_image.name,
                                    "Foto pembanding": compared_images[idx - 1].name,
                                    "Embedding Similarity %": comparison["Embedding Similarity %"],
                                    "DeepFace Similarity %": comparison["DeepFace Similarity %"],
                                    "Comprehensive Similarity (experimental)": comparison[
                                        "Comprehensive Similarity (experimental)"
                                    ],
                                    "Metric Consistency (experimental)": comparison[
                                        "Metric Consistency (experimental)"
                                    ],
                                    "Distance": comparison["Distance"],
                                    "Threshold": comparison["Threshold"],
                                    "Verified": "Ya" if comparison["Verified"] else "Tidak",
                                    "Rekomendasi": comparison["Recommendation"],
                                    "Interpretasi": comparison["Interpretation"],
                                }
                            )

                    st.success(f"Analisis selesai: {len(results)} pembandingan dilakukan")
                    results.sort(key=lambda item: item["Distance"])
                    passed_count = sum(result["Verified"] == "Ya" for result in results)
                    best_match = results[0]
                    st.markdown("### Penjelasan hasil One vs Many")
                    st.info(
                        f"{passed_count} dari {len(results)} foto pembanding lolos threshold model. "
                        f"Jarak terendah ada pada {best_match['Foto pembanding']} "
                        f"(jarak {best_match['Distance']:.4f}; similarity turunan "
                        f"{best_match['DeepFace Similarity %']}%)."
                    )
                    st.write(best_match["Interpretasi"])
                    st.caption(
                        "Tabel diurutkan dari jarak terendah. Skor tertinggi atau lolos threshold bukan "
                        "kepastian identitas; hasil hanya berlaku untuk kumpulan foto yang diunggah."
                    )
                    st.subheader("Hasil One vs Many")
                    st.dataframe(results, use_container_width=True)

            except Exception as exc:
                st.error(f"Terjadi kesalahan saat memproses foto utama dan pembanding: {exc}")
            finally:
                if os.path.exists(temp_primary):
                    os.remove(temp_primary)
                for temp_path in temp_compared:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        st.info("Silakan upload satu foto utama dan satu atau lebih foto pembanding.")
elif compare_mode == "Batch Compare":
    uploaded_images = st.file_uploader(
        "Upload beberapa foto wajah untuk batch compare",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_images:
        preview_cols = st.columns(min(4, len(uploaded_images)))
        for idx, image in enumerate(uploaded_images):
            with preview_cols[idx % min(4, len(uploaded_images))]:
                render_image_preview(image, image.name, show_face_boxes)

        if st.button("Mulai Batch Compare", width="stretch"):
            temp_paths = [save_uploaded_file(file, f"batch_{idx}") for idx, file in enumerate(uploaded_images)]
            try:
                if len(temp_paths) < 2:
                    st.warning("Minimal dibutuhkan 2 foto untuk batch compare.")
                else:
                    with st.spinner("Memproses semua kombinasi wajah..."):
                        batch_results = compare_batch_images(
                            temp_paths,
                            [image.name for image in uploaded_images],
                        )

                    st.success(f"Batch compare selesai: {len(batch_results)} pasangan dianalisis")

                    if batch_results:
                        batch_results.sort(key=lambda item: item["Distance"])
                        passed_count = sum(result["Verified"] == "Ya" for result in batch_results)
                        top_match = batch_results[0]
                        st.markdown("### Penjelasan hasil Batch Compare")
                        st.info(
                            f"{passed_count} dari {len(batch_results)} pasangan lolos threshold model. "
                            f"Jarak terendah ditemukan pada {top_match['Foto 1']} dan {top_match['Foto 2']} "
                            f"(jarak {top_match['Distance']:.4f}; similarity turunan "
                            f"{top_match['DeepFace Similarity %']}%)."
                        )
                        st.write(top_match["Interpretasi"])
                        st.caption(
                            "Urutan menunjukkan pasangan dengan jarak model terendah, bukan identifikasi "
                            "pasti. Setiap pasangan dinilai secara terpisah terhadap threshold ArcFace."
                        )

                    st.subheader("Hasil Batch Compare")
                    st.dataframe(batch_results, use_container_width=True)

            except Exception as exc:
                st.error(f"Terjadi kesalahan saat memproses batch foto: {exc}")
            finally:
                for temp_path in temp_paths:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        st.info("Silakan upload minimal 2 foto wajah untuk memulai batch compare.")
else:
    st.caption("Tinjau dua foto berdampingan secara manual. Checklist ini hanya membantu observasi, bukan memastikan identitas seseorang.")
    visual_primary = st.file_uploader(
        "Upload foto utama",
        type=["jpg", "jpeg", "png"],
        key="visual_primary",
    )
    visual_candidates = st.file_uploader(
        "Upload foto pembanding",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="visual_candidates",
    )

    if visual_primary and visual_candidates:
        selected_candidate_name = st.selectbox(
            "Pilih foto yang ingin ditinjau",
            [image.name for image in visual_candidates],
            key="visual_candidate_selection",
        )
        selected_candidate = next(
            image for image in visual_candidates if image.name == selected_candidate_name
        )

        left_col, right_col = st.columns(2)
        with left_col:
            render_image_preview(
                visual_primary,
                f"Foto utama: {visual_primary.name}",
                show_face_boxes,
            )
        with right_col:
            render_image_preview(
                selected_candidate,
                f"Foto pembanding: {selected_candidate.name}",
                show_face_boxes,
            )

        st.subheader("Checklist pengamatan")
        st.caption("Pilih penilaian untuk setiap ciri yang terlihat. Pilih “Tidak jelas” jika foto atau sudutnya tidak mendukung penilaian.")
        observations = {}
        for feature in ("Bentuk wajah", "Mata dan alis", "Hidung", "Mulut", "Ciri khas yang terlihat"):
            observations[feature] = st.radio(
                feature,
                ["Tidak jelas", "Tidak mirip", "Mirip"],
                horizontal=True,
                key=f"visual_review_{feature}",
            )

        notes = st.text_area(
            "Catatan reviewer (opsional)",
            key="visual_review_notes",
            placeholder="Contoh: sudut foto berbeda atau pencahayaan kurang jelas.",
        )
        similar_features = sum(value == "Mirip" for value in observations.values())
        different_features = sum(value == "Tidak mirip" for value in observations.values())
        unclear_features = sum(value == "Tidak jelas" for value in observations.values())
        st.markdown("### Penjelasan tinjauan visual")
        if similar_features == len(observations):
            st.info(
                "Semua ciri pada checklist ditandai mirip. Ini merangkum pengamatan manual Anda saja; "
                "kemiripan ciri visual tidak memastikan kedua foto menampilkan orang yang sama."
            )
        elif different_features:
            st.warning(
                f"{different_features} ciri ditandai tidak mirip, {similar_features} ditandai mirip, "
                f"dan {unclear_features} tidak jelas. Pengamatan bercampur; tinjau ulang bagian yang "
                "terlihat jelas sebelum menarik kesimpulan."
            )
        else:
            st.info(
                f"{similar_features} ciri ditandai mirip dan {unclear_features} tidak jelas. "
                "Bukti visual yang dicatat belum cukup untuk kesimpulan yang kuat."
            )
        st.caption(
            "Review Visual tidak menghitung hasil AI atau menetapkan identitas. Perbedaan sudut, ekspresi, "
            "pencahayaan, kualitas foto, dan usia foto dapat memengaruhi pengamatan."
        )
        if notes:
            st.write("Catatan:", notes)
    else:
        st.info("Silakan upload satu foto utama dan minimal satu foto pembanding untuk mulai meninjau.")

st.markdown("---")
st.caption("Built with Python, DeepFace, and Face Embedding")
