import cv2
import os
import pickle
import numpy as np
import mediapipe as mp

# PARAMS: Dicionário de parâmetros de configuração do sistema de reconhecimento facial
PARAMS = {
    "DATABASE": "faces_database.pkl",         # Caminho do arquivo onde os rostos cadastrados são salvos
    "FACE_SIZE": (200, 200),                  # Tamanho padrão para redimensionar imagens de rosto
    "SAMPLES_PER_PERSON": 150,                 # Quantidade de amostras coletadas por pessoa no cadastro
    "HAAR_SCALE_FACTOR": 1.1,                 # Fator de escala para detecção de faces (Haar Cascade)
    "HAAR_MIN_NEIGHBORS": 7,                  # Número mínimo de vizinhos para considerar uma detecção válida (Haar Cascade)
    "HAAR_MIN_SIZE": (80, 80),                # Tamanho mínimo da face detectada (Haar Cascade)
    "HAAR_FLAGS": cv2.CASCADE_SCALE_IMAGE     # Flags para o classificador Haar Cascade
}

# Carrega o banco de rostos
if os.path.exists(PARAMS["DATABASE"]):
    with open(PARAMS["DATABASE"], "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {"faces": [], "names": []}


# Classificador Haar Cascade (usa arquivo local)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Erro: Não foi possível carregar o classificador haarcascade_frontalface_default.xml. Verifique se o arquivo está na mesma pasta do app.py.")
    exit(1)

def estimate_head_direction(landmarks, img_w, img_h):
    # Pontos principais: nariz (1), olho esquerdo (33), olho direito (263), canto da boca esquerda (61), canto da boca direita (291)
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    # Conversão para coordenadas de pixel
    nx, ny = int(nose.x * img_w), int(nose.y * img_h)
    lex, ley = int(left_eye.x * img_w), int(left_eye.y * img_h)
    rex, rey = int(right_eye.x * img_w), int(right_eye.y * img_h)
    lmx, lmy = int(left_mouth.x * img_w), int(left_mouth.y * img_h)
    rmx, rmy = int(right_mouth.x * img_w), int(right_mouth.y * img_h)

    # Cálculo dos centros dos olhos e da boca
    eye_cx, eye_cy = (lex + rex) // 2, (ley + rey) // 2
    mouth_cx, mouth_cy = (lmx + rmx) // 2, (lmy + rmy) // 2

    # Critérios horizontais (esquerda/direita)
    nose_eye_dx = nx - eye_cx
    nose_mouth_dx = nx - mouth_cx

    # Critérios verticais (cima/baixo) usando ângulo e distância relativa
    # Distância vertical do nariz para olhos e boca
    eye_to_nose = ny - eye_cy
    nose_to_mouth = mouth_cy - ny
    # Distância entre olhos e boca (altura do rosto)
    face_height = mouth_cy - eye_cy
    # Normalização para limiares proporcionais ao tamanho do rosto
    up_thresh = int(0.28 * face_height)  # mais sensível para cima
    down_thresh = int(0.32 * face_height)  # mais tolerante para baixo

    # Cálculo do ângulo do eixo olhos-boca
    import math
    angle = math.degrees(math.atan2(mouth_cy - eye_cy, mouth_cx - eye_cx))

    # Decisão
    if abs(nose_eye_dx) < 12 and abs(nose_mouth_dx) < 12:
        # Critério de frente
        if eye_to_nose < -up_thresh:
            return 'cima'
        elif nose_to_mouth < -down_thresh:
            return 'baixo'
        else:
            return 'frente'
    elif nose_eye_dx < -18 and nose_mouth_dx < -18:
        return 'direita'
    elif nose_eye_dx > 18 and nose_mouth_dx > 18:
        return 'esquerda'
    elif eye_to_nose < -up_thresh:
        return 'cima'
    elif nose_to_mouth < -down_thresh:
        return 'baixo'
    else:
        return 'indefinido'

def cadastrar_rosto(camera_index):
    mp_face = mp.solutions.face_mesh
    nome = input("Digite seu nome: ")
    cap = cv2.VideoCapture(camera_index)
    total_amostras = PARAMS["SAMPLES_PER_PERSON"]
    print(f"Será feito o cadastro automático. Movimente levemente a cabeça quando solicitado.")
    tempo_preparacao = 3  # segundos para a pessoa se preparar
    amostras = 0
    # Mostra contagem regressiva na tela
    window_name = "Cadastro de Rosto"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Garante que a janela fique em primeiro plano
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    for t in range(tempo_preparacao, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]
        msg = f"Prepare-se! Movimente a cabeça em {t}..."
        cv2.putText(frame, msg, (30, img_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1000)
    print("Iniciando coleta de amostras!")
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while amostras < total_amostras:
            ret, frame = cap.read()
            if not ret:
                break
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                # Só captura se detectar face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=PARAMS["HAAR_SCALE_FACTOR"],
                    minNeighbors=PARAMS["HAAR_MIN_NEIGHBORS"],
                    minSize=PARAMS["HAAR_MIN_SIZE"],
                    flags=PARAMS["HAAR_FLAGS"]
                )
                for (x, y, w, h) in faces:
                    rosto = gray[y:y+h, x:x+w]
                    rosto = cv2.resize(rosto, PARAMS["FACE_SIZE"])
                    known_faces["faces"].append(rosto)
                    known_faces["names"].append(nome)
                    amostras += 1
                    print(f"Amostra {amostras}/{total_amostras} capturada.")
                    break
            cv2.putText(frame, f"Amostras: {amostras}/{total_amostras}", (10, img_h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Cadastro interrompido pelo usuário.")
                cap.release()
                cv2.destroyAllWindows()
                return
    with open(PARAMS["DATABASE"], "wb") as f:
        pickle.dump(known_faces, f)
    print(f"✅ Cadastro de {nome} finalizado!")
    cap.release()
    cv2.destroyAllWindows()

def reconhecer_rosto(camera_index):
    if len(known_faces["faces"]) == 0:
        print("Nenhum rosto cadastrado ainda!")
        return

    faces = [cv2.resize(face, PARAMS["FACE_SIZE"]) for face in known_faces["faces"]]
    labels = []
    label_map = {}
    label_counter = 0
    for name in known_faces["names"]:
        if name not in label_map:
            label_map[name] = label_counter
            label_counter += 1
        labels.append(label_map[name])

    if len(faces) == 0:
        print("Nenhum rosto cadastrado ainda!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    cap = cv2.VideoCapture(camera_index)
    print(f"Pressione 'q' para sair. Usando câmera {camera_index}.")
    window_name = "Reconhecimento de Rosto"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Garante que a janela fique em primeiro plano
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=PARAMS["HAAR_SCALE_FACTOR"],
            minNeighbors=PARAMS["HAAR_MIN_NEIGHBORS"],
            minSize=PARAMS["HAAR_MIN_SIZE"],
            flags=PARAMS["HAAR_FLAGS"]
        )

        for (x, y, w, h) in faces_detected:
            rosto_atual = gray[y:y+h, x:x+w]
            rosto_atual = cv2.resize(rosto_atual, PARAMS["FACE_SIZE"])

            # Pré-processamento
            rosto_atual = cv2.equalizeHist(rosto_atual)  # Equalização de histograma
            rosto_atual = cv2.GaussianBlur(rosto_atual, (3, 3), 0)  # Suavização

            # Normalização (opcional)
            rosto_atual = cv2.normalize(rosto_atual, None, 0, 255, cv2.NORM_MINMAX)

            label_pred, confidence = recognizer.predict(rosto_atual)
            # Aumentando o threshold para tornar o reconhecimento mais rigoroso
            if confidence < 60:
                name = [k for k, v in label_map.items() if v == label_pred][0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def menu():
    while True:
        print("\n--- MENU ---")
        print("1 - Cadastrar rosto")
        print("2 - Reconhecer rosto")
        print("0 - Sair")
        opcao = input("Escolha: ")

        if opcao in ["1", "2"]:
            try:
                camera_index = int(input("Digite o número da câmera (normalmente 0 ou 1): "))
            except ValueError:
                print("Índice inválido. Usando câmera 0.")
                camera_index = 0
            if opcao == "1":
                cadastrar_rosto(camera_index)
            elif opcao == "2":
                reconhecer_rosto(camera_index)
        elif opcao == "0":
            print("Encerrando...")
            break
        else:
            print("Opção inválida")

if __name__ == "__main__":
    menu()
