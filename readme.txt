# Instruções de Execução do Script Yolo
    - Instale os seguintes pacotes python utilizando o próprio PIP
        - python-opencv
        - numpy
    - Após a instalação para executar o script yolo basta digitar
        - python yolo.py --image ~/<image_dir> --yolo ~/<yolo_coco_dir>
            - image_dir é o diretório que contém as imagens
            - yolo_coco_dir é o diretório que contém os arquivos do modelo yolo treinado

# Instruções de Execução da CNN
    - A implementação se encontra em um arquivo .ipynb
    - Recomendamos que o arquivo seja executado no Google Colab para facilitar todo o processo de instalação
     de dependências

# Links utilizados
    - Arquivo com os pesos do Yolo: https://pjreddie.com/media/files/yolov3.weights
        - Este arquivo deve ser baixado e colocado na pasta yolo_coco
    - Dataset Utilizado: https://drive.google.com/file/d/1bh9uf1Q6YdT7roE8A-ZQK9pypzmcaX7l/view?usp=sharing
    - Weka: https://www.cs.waikato.ac.nz/ml/weka/
