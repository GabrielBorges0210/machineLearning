import struct
import numpy as np
import cv2
import os

def desempacotarIndices(bytes_empacotados: bytes, bits_por_indice: int, total_pixels: int) -> np.ndarray:
    """
    Desfaz o bit-packing, transformando um array de bytes nativos (uint8)
    de volta em um array expandido de índices (uint8).
    """
    # Converte os bytes lidos do disco em um array NumPy
    array_bytes = np.frombuffer(bytes_empacotados, dtype=np.uint8)
    
    if bits_por_indice == 8:
        indices = array_bytes
        
    elif bits_por_indice == 4:
        # Cada byte contém 2 índices.
        # Deslocamos 4 bits para a direita para pegar o índice superior (esquerda)
        indices_superiores = (array_bytes >> 4) & 0x0F
        # Usamos uma máscara bitwise AND (0x0F = 00001111) para pegar o índice inferior (direita)
        indices_inferiores = array_bytes & 0x0F
        
        # Juntamos as duas colunas
        indices = np.column_stack((indices_superiores, indices_inferiores)).flatten()
        
    elif bits_por_indice == 2:
        # Cada byte contém 4 índices.
        idx_0 = (array_bytes >> 6) & 0x03
        idx_1 = (array_bytes >> 4) & 0x03
        idx_2 = (array_bytes >> 2) & 0x03
        idx_3 = array_bytes & 0x03
        
        indices = np.column_stack((idx_0, idx_1, idx_2, idx_3)).flatten()
        
    elif bits_por_indice == 1:
        # Usamos a função nativa do NumPy para desfazer arrays de 1 bit
        indices = np.unpackbits(array_bytes)
        
    else:
        raise ValueError(f"Bits por índice não suportado: {bits_por_indice}")
    
    # O array desempacotado pode conter padding no final.
    # Truncamos o array para o tamanho exato de pixels da imagem original.
    return indices[:total_pixels]

def descompactarArquivo(caminho_arquivo_entrada: str) -> np.ndarray:
    """
    Lê o arquivo binário customizado, extrai o cabeçalho, os centroides
    e reconstrói a matriz original da imagem.
    """
    with open(caminho_arquivo_entrada, 'rb') as f:
        # Leitura e Validação da Assinatura (Magic Number)
        # 4s = string de 4 bytes
        assinatura = struct.unpack('4s', f.read(4))[0]
        if assinatura != b'KMC1':
            raise ValueError("Formato de arquivo inválido ou corrompido. Assinatura 'KMC1' não encontrada.")
            
        # Formato: Width, Height, Canais, Bits
        # H = unsigned short (2 bytes, max 65535) para dimensões
        # B = unsigned char (1 byte, max 255) para canais e bits
        formato_metadados = 'H H B B'
        tamanho_metadados = struct.calcsize(formato_metadados)
        dados_metadados = f.read(tamanho_metadados)
        
        largura, altura, canais, bits_por_indice = struct.unpack(formato_metadados, dados_metadados)
        
        # Cálculo de variáveis derivadas
        total_pixels = largura * altura
        k_centroides = 2 ** bits_por_indice
        
        print(f"Cabeçalho: {largura}x{altura}, Canais: {canais}, Bits/Pixel: {bits_por_indice} (K={k_centroides})")
        
        # Leitura do Dicionário (Centroides)
        # Centroides são floats (4 bytes por valor)
        # A quantidade de valores é K * canais
        tamanho_centroides = k_centroides * canais
        formato_centroides = f'{tamanho_centroides}f'
        bytes_centroides = f.read(struct.calcsize(formato_centroides))
        
        centroides_flat = struct.unpack(formato_centroides, bytes_centroides)
        # Remolda a tupla lida em uma matriz numpy K x C
        centroides = np.array(centroides_flat, dtype=np.float32).reshape((k_centroides, canais))
        
        # Leitura do Payload (Dados compactados)
        # O resto do arquivo são os bytes empacotados
        payload_bytes = f.read()
        
        # Desempacotamento e Reconstrução
        indices_recuperados = desempacotarIndices(payload_bytes, bits_por_indice, total_pixels)
        # Usa indexação avançada do NumPy para mapear cada índice para seu centroide correspondente
        vetores_reconstruidos = centroides[indices_recuperados]
        
        # Remolda a matriz de volta para as dimensões da imagem (Height, Width, Canais)
        imagem_reconstruida = vetores_reconstruidos.reshape((altura, largura, canais))
        
        # Converte de volta para inteiros sem sinal de 8 bits para visualização (RGB normal)
        return np.clip(imagem_reconstruida, 0, 255).astype(np.uint8)

def deletarArquivosNoDiretorio(path):
    for arquivo in os.listdir(path):
        caminhoCompleto = os.path.join(path, arquivo)
        
        if(os.path.isfile(caminhoCompleto)):
            os.remove(caminhoCompleto)

if __name__ == "__main__":
    try:
        imagem = descompactarArquivo("output/compressedFile.kmc1")
        print(f"Imagem reconstruída com shape: {imagem.shape}")
        
        deletarArquivosNoDiretorio("output")
        
        # Salvar para visualizar usando OpenCV (convertendo de volta para BGR)
        cv2.imwrite("output/imagem_restaurada.bmp", cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"Erro na descompactação: {e}")