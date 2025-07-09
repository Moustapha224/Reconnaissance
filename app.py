import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib
import os
import urllib.request
import bz2
import time

# ========== CONFIGURATION ==========
KNOWN_FACES_DIR = "Known_faces"
SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
FACE_ENCODER_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_ENCODER_PATH = "dlib_face_recognition_resnet_model_v1.dat"
TOLERANCE = 0.6

# ========== TÉLÉCHARGEMENT DES MODÈLES DLIB ==========
def download_dlib_model(url, output_path):
    if not os.path.exists(output_path):
        zipped_path = output_path + ".bz2"
        urllib.request.urlretrieve(url, zipped_path)
        with bz2.open(zipped_path, "rb") as f_in, open(output_path, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(zipped_path)

download_dlib_model(SHAPE_PREDICTOR_URL, SHAPE_PREDICTOR_PATH)
download_dlib_model(FACE_ENCODER_URL, FACE_ENCODER_PATH)

# ========== CHARGEMENT DES MODÈLES ==========
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_ENCODER_PATH)

# ========== ENCODAGE D'UN VISAGE ==========
def get_face_encoding(image):
    if image is None:
        return None

    # Vérification et conversion
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return None
    except:
        return None

    dets = face_detector(image, 1)
    if len(dets) == 0:
        return None
    shape = shape_predictor(image, dets[0])
    return face_rec_model.compute_face_descriptor(image, shape)

# ========== CHARGEMENT DES VISAGES CONNUS ==========
known_encodings = []
known_names = []

if os.path.exists(KNOWN_FACES_DIR) and os.path.isdir(KNOWN_FACES_DIR):
    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(person_dir, filename)

            try:
                pil_img = Image.open(img_path).convert("RGB")
                img = np.array(pil_img)
                encoding = get_face_encoding(img)

                if encoding is not None:
                    known_encodings.append(encoding)
                    known_names.append(name)
                else:
                    st.warning(f"⚠️ Visage non détecté : {img_path}")
            except Exception as e:
                st.warning(f"❌ Erreur de lecture image : {img_path} - {e}")
else:
    st.warning(f"📂 Le dossier {KNOWN_FACES_DIR} est introuvable ou vide.")

# ========== INTERFACE UTILISATEUR ==========
st.title("📸 Application de Reconnaissance Faciale en Temps Réel")
st.markdown("Ce système détecte et identifie les visages connus via la webcam.")

start_cam = st.button("📷 Démarrer la webcam")

# ========== BOUCLE DE DÉTECTION ==========
if start_cam:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Impossible d'accéder à la webcam.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Problème de lecture vidéo.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_detector(rgb_small_frame)
            for face_rect in face_locations:
                shape = shape_predictor(rgb_small_frame, face_rect)
                encoding = face_rec_model.compute_face_descriptor(rgb_small_frame, shape)

                matches = [np.linalg.norm(np.array(enc) - np.array(encoding)) < TOLERANCE for enc in known_encodings]
                name = "Inconnu"
                if any(matches):
                    match_index = matches.index(True)
                    name = known_names[match_index]

                # Mise à l'échelle
                top, right, bottom, left = (face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR")
            time.sleep(0.03)

        cap.release()
        cv2.destroyAllWindows()
