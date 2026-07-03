import struct
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cv2
import glob

def empacotar_indices(indices: np.ndarray, bits_por_indice: int) -> bytes:
    """
    Empacota um array de inteiros de tamanhos sub-byte (1, 2, 4 bits)
    em um array de bytes nativos (uint8) usando NumPy vetorizado.
    """
    # Garante que o array está plano e é do tipo correto para bitwise
    indices = indices.flatten().astype(np.uint8)
    N = len(indices)
    
    if bits_por_indice == 8:
        return indices.tobytes()
        
    elif bits_por_indice == 4:
        # Tratamento de padding: o número de índices precisa ser par
        if N % 2 != 0:
            indices = np.append(indices, 0)
        
        # Reshape em duas colunas (par de índices por byte)
        matriz = indices.reshape(-1, 2)
        # Desloca o primeiro índice para a esquerda e junta com o segundo
        bytes_empacotados = (matriz[:, 0] << 4) | matriz[:, 1]
        return bytes_empacotados.tobytes()
        
    elif bits_por_indice == 2:
        # Tratamento de padding: precisa ser múltiplo de 4
        resto = N % 4
        if resto != 0:
            indices = np.append(indices, np.zeros(4 - resto, dtype=np.uint8))
            
        matriz = indices.reshape(-1, 4)
        bytes_empacotados = (matriz[:, 0] << 6) | (matriz[:, 1] << 4) | (matriz[:, 2] << 2) | matriz[:, 3]
        return bytes_empacotados.tobytes()
        
    elif bits_por_indice == 1:
        # Para 1 bit (K=2), o np.packbits nativo do NumPy funciona
        return np.packbits(indices).tobytes()
        
    else:
        raise ValueError("Este compressor exige potências de 2 exatas (1, 2, 4 ou 8 bits).")

def buscar_melhor_k_kmeans(vetores: np.ndarray, mse_maximo: float, max_bits: int = 8):
    """
    Busca a menor potência de 2 para K tal que o MSE seja inferior a mse_maximo.
    
    Parâmetros:
    vetores: Array NumPy (N, D) contendo os dados originais (ex: pixels RGB float32 ou uint8).
    mse_maximo: Tolerância máxima de distorção aceitável.
    max_bits: Teto máximo de bits por índice (padrão 8 bits -> K = 256).
    """
    N, D = vetores.shape
    
    # Converter para float32 para cálculos precisos de distância no sklearn
    vetores_float = vetores.astype(np.float32)
    
    melhor_modelo = None
    melhor_bits = None
    melhor_indices = None
    
    # Restringimos estritamente aos bits que cabem exatos num byte: 1, 2, 4 ou 8.
    for bits in [1, 2, 4, 8]:
        k = 2 ** bits
        
        kmeans = MiniBatchKMeans(
            n_clusters=k, 
            batch_size=min(10000, N), 
            random_state=42, 
            n_init="auto"
        )
        
        indices = kmeans.fit_predict(vetores_float)
        centroides = kmeans.cluster_centers_
        
        vetores_reconstruidos = centroides[indices]
        mse_atual = np.mean((vetores_float - vetores_reconstruidos) ** 2)
        
        print(f"[{bits} bit/vetor | K={k:3d}] -> MSE: {mse_atual:.2f} (Alvo: <= {mse_maximo})")
        
        melhor_bits = bits
        melhor_modelo = kmeans
        melhor_indices = indices
        
        if mse_atual <= mse_maximo:
            print(f"Meta atingida com {bits} bits/pixel!")
            break
            
    if mse_atual > mse_maximo:
        print(f"⚠️ Alerta: Limite de bits ({max_bits}) atingido antes da meta de MSE.")
            
    # Retornamos melhor_bits e não k!
    return melhor_bits, melhor_modelo.cluster_centers_, melhor_indices

def compactarImagem(caminho_entrada: str, caminho_saida: str, mse_maximo: float = 50.0):
    """
    Lê a imagem, quantiza vetorialmente e guarda no formato binário .kmc1.
    """
    # OpenCV carrega em BGR
    img = cv2.imread(caminho_entrada, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Falha ao carregar a imagem. Verifique o caminho: {caminho_entrada}")
        
    # Defesa contra canal Alpha e padronização para espaço RGB (vital no OpenCV)
    if len(img.shape) == 3 and img.shape[2] == 4:
        print("Convertendo imagem BGRA para RGB puro...")
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2: # Escala de cinza
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    matriz_img = img
    altura, largura, canais = matriz_img.shape
    total_pixels = largura * altura
    
    # Reshape para N vetores de dimensão C
    vetores = matriz_img.reshape(-1, canais)
    
    print(f"Iniciando compressão de {largura}x{altura} píxeis...")
    
    # Treino
    bits, centroides, indices = buscar_melhor_k_kmeans(vetores, mse_maximo)
    
    # Bit-packing
    payload_bytes = empacotar_indices(indices, bits)
    
    # Construção do Cabeçalho e Escrita Binária
    with open(caminho_saida, 'wb') as f:
        # Assinatura (Magic Number)
        f.write(struct.pack('4s', b'KMC1'))
        
        # Metadados (Width [H], Height [H], Canais [B], Bits [B])
        # Total de 6 bytes
        f.write(struct.pack('H H B B', largura, altura, canais, bits))
        
        # Dicionário (Centroides)
        # Matriz plana como floats
        k_centroides = 2 ** bits
        tamanho_centroides = k_centroides * canais
        formato_centroides = f'{tamanho_centroides}f'
        f.write(struct.pack(formato_centroides, *centroides.flatten()))
        
        # Payload
        f.write(payload_bytes)

    # Calculando a taxa de compressão
    tamanho_original = total_pixels * canais  # Pixels brutos
    tamanho_final = os.path.getsize(caminho_saida)
    taxa = (1 - (tamanho_final / tamanho_original)) * 100
    
    print("-" * 30)
    print("Resumo da Compressão:")
    print(f"Tamanho Original Bruto: {tamanho_original / 1024:.2f} KB")
    print(f"Tamanho Final (.kmc1):  {tamanho_final / 1024:.2f} KB")
    print(f"Taxa de Redução:        {taxa:.2f}%")
    
    if tamanho_final > tamanho_original:
        print("AVISO: O overhead do dicionário tornou o ficheiro maior que o original.")

def deletarArquivosNoDiretorio(path):
    for arquivo in os.listdir(path):
        caminhoCompleto = os.path.join(path, arquivo)
        
        if(os.path.isfile(caminhoCompleto)):
            os.remove(caminhoCompleto)

if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    deletarArquivosNoDiretorio('output')
    
    arquivos_input = [f for f in glob.glob("input/*") if os.path.isfile(f)]
    
    # Se estiver vazia, gera o arquivo para teste
    if len(arquivos_input) == 0:
        print("Pasta 'input/' vazia. Gerando imagem sintética de teste...")
        img_teste = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        caminho_entrada = "input/teste.bmp"
        cv2.imwrite(caminho_entrada, cv2.cvtColor(img_teste, cv2.COLOR_RGB2BGR))
    
    # Se houver mais de 1 arquivo, avisa e impõe a regra de 1 único arquivo
    elif len(arquivos_input) > 1:
        print(f"Atenção: A lógica do sistema exige apenas 1 arquivo. Pegando o primeiro e limpando o resto.")
        caminho_entrada = arquivos_input[0]
        for f in arquivos_input[1:]:
            os.remove(f)
    else:
        caminho_entrada = arquivos_input[0]
    
    compactarImagem(caminho_entrada, "output/compressedFile.kmc1", mse_maximo=15.0)