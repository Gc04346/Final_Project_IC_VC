import os
import csv
import cv2 as cv
import time
import argparse
import numpy as np


# adicionando argumentos
from utils import load_images_from_folder

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# carregando as labels do COCO onde o modelo YOLO foi treinado
labels_path = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labels_path).read().strip().split("\n")
# labels já presentes testadas no dataset
OFFICE_LABELS = ['apple', 'backpack', 'bed', 'book', 'bottle', 'bowl', 'car', 'cat', 'cell phone', 'chair', 'clock',
                 'cup', 'diningtable', 'handbag', 'keyboard', 'knife', 'laptop', 'microwave', 'mouse', 'orange', 'oven',
                 'person', 'pizza', 'pottedplant', 'remote', 'scissors', 'sofa', 'spoon', 'teddy bear', 'tie',
                 'tvmonitor', 'vase', 'wine glass', 'mean', 'variance', 'std', 'office']

# inicializando uma lista de cores para representar cada label possível
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# carregando os paths do arquivo de pesos e configuração do modelo
weights_path = os.path.sep.join([args["yolo"], "yolov3.weights"])
config_path = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# carregando o YOLO object detector
print('loading YOLO...')
net = cv.dnn.readNetFromDarknet(config_path, weights_path)

# carregando as imagens
images = load_images_from_folder(args["image"])
# office_labels = set()
images_labels = []

# iterando sobre as imagens do diretório
for image in images:
    (h, w) = image.shape[:2]

    # definindo somente as camadas de saída
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # separando a partir da image inserida e então performando a rede FP do YOLO detector
    # gerando os retângulos e as probabilidades
    blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()

    # exibindo tal informação
    print(f'Processamento em {end - start}')

    # inicializando as listas de detected bounding boxes, confidences and class Ids
    boxes = []
    confidences = []
    class_ids = []

    # iterando sobre cada camada de saída
    for output in layer_outputs:
        # iterando sobre cada detecção
        for detection in output:
            # extraindo o ID da classe, probabilidade e a detecção do objeto
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filtrando as predições "fracas"
            if confidence > args["confidence"]:
                # escalando a box para o tamanho original da imagem
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                # usando as coordenadas do centro (x, y) para derivar o canto superior e esquerdo
                x = int(center_x - (width/2))
                y = int(center_y - (height/2))

                # atualizando nossas listas
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # aplicando o non-maxima suppression para suprimir predições fracas
    idxs = cv.dnn.NMSBoxes(boxes, confidences, args["confidence"], args['threshold'])
    labels = set()

    # conferindo se temos pelo menos uma detecção
    if len(idxs) > 0:
        # iterando por cada índice
        for i in idxs.flatten():
            # extraindo a caixa de marcação do objeto
            # (x, y) = (boxes[i][0], boxes[i][1])
            # (w, h) = (boxes[i][2], boxes[i][3])

            # desenhando a caixa de marcação do objeto
            # color = [int(c) for c in COLORS[class_ids[i]]]
            # cv.rectangle(image, (x, y), (x+w, y+h), color, 2)
            # text = f'{LABELS[class_ids[i]]}: {confidences[i]:.4f}'
            text = f'{LABELS[class_ids[i]]}'
            labels.add(text)
            # cv.putText(image, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # preparando o conjunto e a lista de todas as features de todas as imagens lidas
    # office_labels = office_labels.union(labels)
    labels = list(sorted(labels))
    labels.append(image.mean())
    image_std = image.std()
    labels.append(image_std**2)
    labels.append(image_std)
    images_labels.append(labels)

# gerando o arquivo csv com as features
# office_labels = sorted(office_labels)
# images_labels = [sorted(labels) for labels in images_labels]
# headers = [label for label in OFFICE_LABELS]
# headers.append('office')

with open('dataset_metrics.csv', 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # gravando o cabeçalho
    # writer.writerow(OFFICE_LABELS)
    # para cada conjunto de labels entre as labels das imagens
    for labels in images_labels:
        content = []
        # conferindo se a label encontrada está entre as labels encontradas em escritórios
        for office_label in OFFICE_LABELS:
            # registrando a presença ou não da feature
            if office_label in labels:
                content.append(True)
            else:
                content.append(False)
        # adicionando ao final as métricas de média, variância e desvio padrão
        content.append(labels[-3])
        content.append(labels[-2])
        content.append(labels[-1])
        # adicionando a marcação de escritório e inserindo o conteúdo no arquivo
        content.append(False)
        writer.writerow(content)
