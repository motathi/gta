# GTA San Andreas → vídeo por código

Gera um vídeo de 15 s **por código** com assets reais do GTA San Andreas: um ped (Woozie) caminhando e depois
correndo com as animações originais do jogo, um Infernus estacionado, postes/hidrante/semáforo do jogo, numa rua
procedural ao pôr do sol. Blender (headless) renderiza os quadros e o ffmpeg monta o MP4.

> Os arquivos do jogo são da Rockstar/Take-Two e **não** vêm neste repositório (nem devem ser publicados nele:
> o `.gitignore` bloqueia `.dff/.txd/.ifp/.img`). Use cópias do **seu** jogo, para fins pessoais e não comerciais.

## Requisitos
- Blender 4.2+ com o add-on **DragonFF** (Edit › Preferences › Get Extensions › DragonFF)
  ou, via linha de comando: `blender -b --command extension install --enable dragonff`
- ffmpeg e Python 3.8+ (numpy só para a trilha sonora; sem ele o vídeo sai mudo)

## Uso
```bash
# 1) extrai do SEU San Andreas (PC clássico) só o que é preciso -> ./assets e ./assets_starter.zip
python tools/sa_extract.py "C:\Program Files (x86)\Steam\steamapps\common\Grand Theft Auto San Andreas"

# 2) renderiza e monta o vídeo -> out/video.mp4   (retoma de onde parou se for interrompido)
python render/make_video.py --assets assets --out out/video.mp4
```
**Com placa de vídeo (recomendado, minutos em vez de ~1 h):**
```bash
# Cycles na GPU (OptiX/CUDA/HIP/Metal detectados automaticamente), mais amostras e resolução maior
python render/make_video.py --assets assets --out out/video.mp4 --device GPU --samples 32 --res 1280x720
# ou Eevee (tempo real, o mais rápido; visual um pouco menos fiel)
python render/make_video.py --assets assets --out out/video.mp4 --engine eevee --res 1920x1080 --final-res 1920x1080
```
Se o Blender não for achado sozinho: `--blender "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"`.

Útil: `python render/ifp_anp3.py assets/anim/ped.ifp` lista as animações do IFP; `tools/sa_extract.py --list <nome>`
procura arquivos dentro dos `.img`. Qualquer pasta com os arquivos serve em `--assets` (a busca é recursiva por nome).

## Como funciona
| Arquivo | Papel |
|---|---|
| `tools/sa_extract.py` | lê os `.img` (IMG v2) do jogo e copia só peds, veículos, `ped.ifp` e `models/generic` |
| `render/ifp_anp3.py` | leitor do formato de animação ANP3 (`ped.ifp`) em Python puro |
| `render/sa_blender.py` | import via DragonFF, `PoseSampler` (aplica o IFP aos ossos pelo `bone_id` HAnim), limpeza do carro (sem colisão/LOD/peças danificadas, rodas clonadas, pintura recolorida), tratamento de texturas ausentes |
| `render/scene_video.py` | rua, prédios, céu, props, coreografia e 3 planos de câmera; renderiza os PNGs (Cycles, CPU) |
| `render/make_video.py` | orquestra Blender + ffmpeg (+ trilha de `synth_audio.py`) |

Detalhes que importam:
- O osso raiz das animações de andar/correr traz deslocamento para a frente; ele é retirado e o objeto anda na
  velocidade do ciclo (1,56 m/s caminhando, 5,7 m/s correndo), o que evita o ped "patinar".
- Sem o `.txd` do ped, ele é pintado por pesos dos ossos (pele/terno/sapato/cabelo); com o `.txd` na pasta, as
  texturas do jogo são usadas. Texturas ausentes do carro (ex.: `vehicle.txd`) caem para cores planas.
- Tempo de render: ~10 s/quadro em 4 núcleos (960×540, 10 amostras + denoise) → ~1 h para 360 quadros.

## Testes
`python -m unittest -v render/test_ifp_anp3.py` (arquivo ANP3 sintético). Com `SA_PED_IFP=caminho/ped.ifp` também
valida que os quaternions do arquivo real têm norma ≈ 1.

## Créditos e licenças
Blender e DragonFF (GPL-3.0). A conta de retarget do IFP segue a abordagem do add-on `io_scene_gta_ifp`
(Psycrow101), reimplementada aqui. Arquivos de amostra usados no desenvolvimento vieram dos testes do
`DepsCian/rw-parser-ng` (GPL-3.0, só o código); os assets em si são propriedade da Rockstar/Take-Two.
