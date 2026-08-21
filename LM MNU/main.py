import numpy as np
import soundfile as sf
import sounddevice as sd
import ollama
from faster_whisper import WhisperModel
from kokoro import KPipeline


################# Código gerado em parte com IA #################

def get_context():
    with open("context.txt", "r", encoding="utf-8") as op:
        return op.read()

def record_until_silence(sample_rate=16000, silence_limit=5.0, calib_duration=1.5, noise_factor=2.5):
    """
    Grava o áudio do microfone até detectar silêncio contínuo após a fala.
    
    :param sample_rate: Taxa de amostragem (padrão 16000Hz para o Whisper).
    :param silence_limit: Segundos de pausa necessários para encerrar a gravação.
    :param calib_duration: Segundos iniciais para calcular a média do ruído ambiente.
    :param noise_factor: Multiplicador sobre o ruído médio para definir o limiar de voz.
    """
    chunk_duration = 0.1  # Blocos de 100ms
    chunk_samples = int(sample_rate * chunk_duration)
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        # 1. Calibração do Ruído Ambiente
        print(f"[Microfone] Calibrando ruído ambiente ({calib_duration}s)... Mantenha silêncio.")
        calib_chunks = int((sample_rate * calib_duration) / chunk_samples)
        ambient_rms = []

        for _ in range(max(1, calib_chunks)):
            chunk, _ = stream.read(chunk_samples)
            ambient_rms.append(np.sqrt(np.mean(chunk**2)))
        
        # Média do ruído ambiente com um piso mínimo de segurança
        avg_noise = np.mean(ambient_rms)
        dynamic_threshold = max(avg_noise * noise_factor, 0.003)
        
        print(f"[Microfone] Calibrado (Ruído médio: {avg_noise:.4f} | Limiar: {dynamic_threshold:.4f}). Pode falar!")

        # 2. Gravação do Áudio
        frames = []
        silence_counter = 0.0
        has_spoken = False

        while True:
            chunk, _ = stream.read(chunk_samples)
            frames.append(chunk)
            
            rms = np.sqrt(np.mean(chunk**2))
            
            if rms > dynamic_threshold:
                has_spoken = True
                silence_counter = 0.0
                print('>FALANDO<')
            elif has_spoken:
                silence_counter += chunk_duration
                if silence_counter >= silence_limit:
                    print(f"[Microfone] Pausa de {silence_limit}s detectada. Finalizando.")
                    break

    print("acabou de falar!")
    return np.concatenate(frames, axis=0).flatten()

print("1. Carregando Faster-Whisper na GPU...")
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

print("2. Carregando Kokoro-82M (PT-BR)...")
tts_pipeline = KPipeline(lang_code='p')

#Captura o áudio e espera um silêncio de 5s
audio_input = record_until_silence(sample_rate=16000, silence_limit=3.0, noise_factor=1.5)

print("Transcrevendo áudio do microfone...")
segments, _ = stt_model.transcribe(audio_input, language="pt", beam_size=5)
user_prompt = "".join([s.text for s in segments]).strip()

print(f"O usuário disse:\n'{user_prompt}'\n")

#Pega o contexto no arquivo de texto correto.
context = get_context()


# Atualmente está no modo 'stream', mas está juntando a mensagem inteira.
# No futuro o ideal é fazer o Kokoro processar uma frase inteira conforme o qwen a gera.

# O modo 'think' foi desativado para evitar um raciocínio longo que quase não impacta na qualidade da resposta.

stream = ollama.chat(
    model = 'qwen3.5:9b',
    messages=[
        {
            'role' : 'system',
            'content' :  context,
        },

        {
            'role' : 'user',
            'content' : user_prompt,
        },
    ],
    think = False,
    stream = True
)

audio_text = ''
print("3. Resposta do qwen3.5:9b")
for chunk in stream:
    audio_text+=chunk['message']['content']
    print(chunk['message']['content'], end='', flush=True)
print()

print("4. Sintetizando voz com Kokoro...")
generator = tts_pipeline(audio_text, voice='pm_alex', speed=1.0)

# Lista para armazenar as partes do áudio
audio_chunks = []

for _, _, audio in generator:
    # Converte o tensor do Kokoro para array NumPy e adiciona à lista
    chunk = audio.numpy() if hasattr(audio, 'numpy') else audio
    audio_chunks.append(chunk)

# Junta todos os trechos e salva um único arquivo contínuo
if audio_chunks:
    full_audio = np.concatenate(audio_chunks)
    sf.write("audio_teste_3.wav", full_audio, 24000)
    print("\n--- TUDO PRONTO! O áudio unificado foi gerado como 'teste_saida_completo.wav' ---")
