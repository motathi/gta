#!/usr/bin/env python3
"""Gera o video completo: roda o Blender (scene_video.py) para renderizar os frames e o ffmpeg para montar o MP4.

Exemplo:
  python render/make_video.py --assets assets --out out/video.mp4
Requisitos: Blender 4.2+ com o add-on DragonFF, ffmpeg e Python 3.8+ (numpy so para a trilha; sem ele o video sai mudo).
Os arquivos do jogo NAO vem no repositorio: gere a pasta `assets` com tools/sa_extract.py a partir do SEU jogo.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find_blender(explicit):
    cands = [explicit, os.environ.get("BLENDER"), shutil.which("blender"),
             "/Applications/Blender.app/Contents/MacOS/Blender"]
    cands += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe"), reverse=True)
    cands += sorted(glob.glob(os.path.expanduser("~/blender*/blender")), reverse=True)
    for c in cands:
        if c and os.path.exists(c):
            return c
    sys.exit("Blender nao encontrado: passe --blender CAMINHO ou defina a variavel BLENDER.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="pasta com os arquivos do jogo (saida de tools/sa_extract.py)")
    ap.add_argument("--out", default="out/video.mp4")
    ap.add_argument("--blender", default=None)
    ap.add_argument("--res", default="960x540", help="resolucao do render (o MP4 final e ampliado para --final-res)")
    ap.add_argument("--final-res", default="1280x720")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--ped", default="wuzimu")
    ap.add_argument("--car", default="infernus")
    ap.add_argument("--frames-dir", default=None, help="padrao: <pasta do video>/frames")
    ap.add_argument("--skip-render", action="store_true", help="so codifica os frames ja renderizados")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    frames = os.path.abspath(a.frames_dir or os.path.join(os.path.dirname(out), "frames"))
    os.makedirs(os.path.dirname(out), exist_ok=True)  # a trilha e o MP4 vao para a pasta do video
    os.makedirs(frames, exist_ok=True)
    total = int(round(a.seconds * a.fps))

    if not a.skip_render:
        blender = find_blender(a.blender)
        cmd = [blender, "-b", "-P", os.path.join(HERE, "scene_video.py"), "--",
               "--assets", os.path.abspath(a.assets), "--out", frames, "--res", a.res, "--samples", str(a.samples),
               "--fps", str(a.fps), "--ped", a.ped, "--car", a.car]
        print("render:", " ".join(cmd))
        subprocess.run(cmd, check=True)  # retoma de onde parou: frames ja existentes sao pulados

    got = len(glob.glob(os.path.join(frames, "f_*.png")))
    if got < total:
        sys.exit("faltam frames: %d de %d em %s (rode de novo para continuar o render)" % (got, total, frames))

    audio = None
    if not a.no_audio:
        try:
            sys.path.insert(0, HERE)
            import synth_audio
            audio = os.path.join(os.path.dirname(out), "trilha.wav")
            synth_audio.write_wav(audio, a.seconds)
        except ImportError:
            print("numpy ausente: video sem trilha")

    w, h = a.final_res.lower().split("x")
    vf = ("scale=%s:%s:flags=lanczos,fade=t=in:st=0:d=0.5,fade=t=out:st=%.2f:d=0.7,format=yuv420p"
          % (w, h, a.seconds - 0.7))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(frames, "f_%04d.png")]
    if audio:
        cmd += ["-i", audio]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-movflags", "+faststart"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    cmd += ["-t", str(a.seconds), out]
    subprocess.run(cmd, check=True)
    print("pronto:", out)


if __name__ == "__main__":
    main()
