import numpy as np
import soundfile as sf
import ollama
from faster_whisper import WhisperModel
from kokoro import KPipeline


################# Código gerado em parte com IA #################

def get_context():
    with open("context.txt", "r", encoding="utf-8") as op:
        return op.read()

print("1. Carregando Faster-Whisper na GPU...")
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

print("2. Carregando Kokoro-82M (PT-BR)...")
tts_pipeline = KPipeline(lang_code='p')


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
            'content' : 'Agora você, agente da França, deve fazer uma pergunta para a comissão dos Estados Unidos.',
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
    sf.write("teste_saida_completo.wav", full_audio, 24000)
    print("\n--- TUDO PRONTO! O áudio unificado foi gerado como 'teste_saida_completo.wav' ---")
