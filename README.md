# Face-ID App

## Objetivo
Este projeto é uma aplicação local para reconhecimento e identificação facial de usuários utilizando OpenCV, MediaPipe e Haar Cascade. O sistema permite cadastrar rostos e realizar reconhecimento em tempo real, demonstrando parâmetros ajustáveis e exibindo os resultados na tela.

## Tecnologias Utilizadas
- **OpenCV**: Detecção de faces e manipulação de imagens.
- **MediaPipe**: Extração de landmarks faciais e estimativa de direção da cabeça.
- **Haar Cascade**: Algoritmo clássico para detecção de faces.
- **LBPH (Local Binary Patterns Histograms)**: Reconhecimento facial.
- **Python**: Linguagem principal do projeto.

## Funcionamento
1. **Cadastro de rosto**: O usuário informa seu nome e o sistema coleta múltiplas amostras do rosto usando a câmera. As imagens são processadas e salvas em um banco local.
2. **Reconhecimento facial**: O sistema utiliza o banco de rostos cadastrados para identificar pessoas em tempo real pela câmera, mostrando o nome e a confiança na tela.
3. **Estimativa de direção da cabeça**: Utiliza landmarks do MediaPipe para indicar se o usuário está olhando para frente, cima, baixo, esquerda ou direita.
4. **Interface**: O menu permite escolher entre cadastrar rosto, reconhecer rosto ou sair. O usuário também pode selecionar o índice da câmera.

## Parâmetros Ajustáveis
Todos os parâmetros principais estão centralizados no dicionário `PARAMS` no início do arquivo `app.py`:

```python
PARAMS = {
    "DATABASE": "faces_database.pkl",         # Caminho do arquivo onde os rostos cadastrados são salvos
    "FACE_SIZE": (200, 200),                  # Tamanho padrão para redimensionar imagens de rosto
    "SAMPLES_PER_PERSON": 30,                 # Quantidade de amostras coletadas por pessoa no cadastro
    "HAAR_SCALE_FACTOR": 1.1,                 # Fator de escala para detecção de faces (Haar Cascade)
    "HAAR_MIN_NEIGHBORS": 7,                  # Número mínimo de vizinhos para considerar uma detecção válida (Haar Cascade)
    "HAAR_MIN_SIZE": (80, 80),                # Tamanho mínimo da face detectada (Haar Cascade)
    "HAAR_FLAGS": cv2.CASCADE_SCALE_IMAGE     # Flags para o classificador Haar Cascade
}
```

### Explicação dos Parâmetros
- **DATABASE**: Nome do arquivo onde os rostos cadastrados são armazenados.
- **FACE_SIZE**: Tamanho para o qual cada imagem de rosto é redimensionada antes do processamento.
- **SAMPLES_PER_PERSON**: Quantidade de imagens capturadas por pessoa durante o cadastro. Quanto maior, melhor o reconhecimento.
- **HAAR_SCALE_FACTOR**: Fator de escala usado pelo Haar Cascade para buscar faces em diferentes tamanhos. Valores maiores tornam a busca mais rápida, mas menos precisa.
- **HAAR_MIN_NEIGHBORS**: Número mínimo de vizinhos para considerar uma detecção como válida. Valores maiores reduzem falsos positivos.
- **HAAR_MIN_SIZE**: Tamanho mínimo (em pixels) para que uma região seja considerada uma face.
- **HAAR_FLAGS**: Flags de configuração do Haar Cascade.

## Como Executar
1. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Certifique-se de que o arquivo `haarcascade_frontalface_default.xml` está na mesma pasta do `app.py`.**
3. **Execute o programa**:
   ```bash
   python app.py
   ```
4. **Siga o menu interativo** para cadastrar rostos ou realizar reconhecimento.

## Ajustando Parâmetros
Para alterar qualquer parâmetro, edite o dicionário `PARAMS` no início do arquivo `app.py`. Por exemplo, para aumentar o número de amostras por pessoa:
```python
"SAMPLES_PER_PERSON": 50
```

## Limitações
- O reconhecimento depende da qualidade da câmera e da iluminação.
- O sistema não faz validação de múltiplos rostos simultâneos.
- O banco de dados é local e não criptografado.
- O LBPH é simples e pode não funcionar bem em ambientes com muita variação de luz ou expressões.

## Próximos Passos
- Adicionar suporte a múltiplos rostos simultâneos.
- Implementar métodos de reconhecimento mais avançados (ex: deep learning).
- Criar interface gráfica (GUI).
- Adicionar criptografia ao banco de dados.

## Nota Ética
Este projeto é apenas para fins educacionais e demonstração técnica. O uso de dados faciais deve respeitar a privacidade dos usuários e seguir as leis de proteção de dados. Não utilize este sistema para fins comerciais ou sem consentimento explícito dos envolvidos.

## Estrutura do Projeto
```
face-id-app/
├── app.py
├── faces_database.pkl
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

## Dependências
- opencv-python
- mediapipe
- numpy

(Instale todas via `pip install -r requirements.txt`)
