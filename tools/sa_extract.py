#!/usr/bin/env python3
"""
sa_extract.py - extrai da SUA instalacao do GTA San Andreas (PC classico) apenas
os arquivos necessarios para renderizar videos por codigo: peds, veiculos e
animacoes. Nao baixa nem envia nada: so le a pasta do jogo e copia alguns arquivos.

Os arquivos extraidos pertencem a Rockstar/Take-Two. Use para fins pessoais e
NAO publique em repositorio publico.

Uso:
  python sa_extract.py "C:\\Games\\GTA San Andreas"              # preset 'starter'
  python sa_extract.py "<pasta do jogo>" --list sweet            # procura nomes nos .img
  python sa_extract.py "<pasta do jogo>" --get "ryder.*" "bmycg.*"   # extras

Saida: ./assets/ e ./assets_starter.zip (um arquivo unico para enviar).
Requer so Python 3.8+ (sem bibliotecas externas).
"""
import argparse
import fnmatch
import json
import os
import struct
import sys
import zipfile

SECTOR = 2048
# onde procurar modelos de peds/veiculos (ordem de prioridade)
IMG_PATHS = ["models/gta3.img", "models/player.img", "models/gta_int.img", "anim/anim.img"]

# nomes de modelo (sem extensao). O que nao existir no seu jogo e apenas ignorado.
PEDS = ["sweet", "ryder", "smoke", "cesar", "truth", "maccer",
        "fam1", "fam2", "fam3", "ballas1", "ballas2", "ballas3"]
VEHICLES = ["infernus", "bullet", "cheetah", "banshee", "sultan", "elegy", "landstal",
            "greenwoo", "voodoo", "tornado", "savanna", "blade", "buccanee", "remingtn", "sabre"]
PLAYER = ["player"]  # CJ: player.dff / player.txd em models/player.img

# (subpasta de saida, lista de nomes, extensoes)
PRESET_STARTER = [
    ("player", PLAYER, ("dff", "txd")),
    ("peds", PEDS, ("dff", "txd")),
    ("vehicles", VEHICLES, ("dff", "txd")),
]
# arquivos soltos (relativos a pasta do jogo) copiados inteiros
LOOSE_STARTER = [("anim", "anim/ped.ifp"), ("generic", "models/generic")]


def find_ci(root, rel):
    """Resolve um caminho relativo ignorando maiusculas/minusculas (Linux/macOS)."""
    cur = root
    for part in rel.replace("\\", "/").split("/"):
        if not os.path.isdir(cur):
            return None
        hit = next((n for n in os.listdir(cur) if n.lower() == part.lower()), None)
        if hit is None:
            return None
        cur = os.path.join(cur, hit)
    return cur


class Img:
    """Arquivo IMG versao 2 (San Andreas): 'VER2', n entradas de 32 bytes, dados em setores de 2048."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        head = self.f.read(8)
        if head[:4] != b"VER2":
            raise ValueError(f"{path}: nao e um IMG v2 do San Andreas")
        (count,) = struct.unpack("<I", head[4:8])
        raw = self.f.read(32 * count)
        if len(raw) != 32 * count:
            raise ValueError(f"{path}: tabela de entradas truncada")
        self.entries = {}
        for i in range(count):
            off, size_stream, size_arch, name = struct.unpack("<IHH24s", raw[i * 32:(i + 1) * 32])
            name = name.split(b"\0", 1)[0].decode("latin-1")
            sectors = size_stream or size_arch
            self.entries[name.lower()] = (off * SECTOR, sectors * SECTOR)

    def read(self, name):
        off, size = self.entries[name.lower()]
        self.f.seek(off)
        return self.f.read(size)


def trim_rw(name, data):
    """DFF/TXD sao um unico chunk RenderWare: o tamanho real e 12 + tamanho do chunk raiz."""
    if name.lower().endswith((".dff", ".txd")) and len(data) >= 12:
        ctype, csize = struct.unpack("<II", data[:8])
        if ctype in (0x10, 0x16) and 12 + csize <= len(data):
            return data[:12 + csize]
    return data


def open_archives(game):
    arcs = {}
    for rel in IMG_PATHS:
        p = find_ci(game, rel)
        if p and os.path.isfile(p):
            try:
                arcs[rel] = Img(p)
            except ValueError as e:
                print("aviso:", e)
    return arcs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", help="pasta do jogo (a que contem gta_sa.exe e a pasta models)")
    ap.add_argument("--out", default="assets", help="pasta de saida (padrao: assets)")
    ap.add_argument("--preset", choices=["starter", "none"], default="starter")
    ap.add_argument("--get", nargs="*", default=[], metavar="PADRAO",
                    help="padroes extras de nome, ex.: 'ryder.*' (vao para assets/extra)")
    ap.add_argument("--list", nargs="?", const="*", metavar="PADRAO",
                    help="so lista o conteudo dos .img que casa com o padrao e sai")
    ap.add_argument("--no-zip", action="store_true", help="nao cria o .zip")
    a = ap.parse_args()

    game = os.path.abspath(a.game)
    arcs = open_archives(game)
    if "models/gta3.img" not in arcs:
        sys.exit("Nao encontrei models/gta3.img (IMG v2) em %s.\n"
                 "Use a pasta do GTA San Andreas classico para PC (a Definitive Edition nao e suportada)." % game)

    if a.list is not None:
        pat = a.list.lower()
        shown = 0
        for rel, img in arcs.items():
            for name, (off, size) in sorted(img.entries.items()):
                if fnmatch.fnmatch(name, pat if any(c in pat for c in "*?[") else f"*{pat}*"):
                    print(f"{rel:18s} {name:28s} {size // 1024:6d} KB")
                    shown += 1
        print(f"{shown} entradas")
        return

    out = os.path.abspath(a.out)
    manifest, missing = [], []

    def save(sub, name, data, source):
        d = os.path.join(out, sub)
        os.makedirs(d, exist_ok=True)
        data = trim_rw(name, data)
        with open(os.path.join(d, name.lower()), "wb") as fh:
            fh.write(data)
        manifest.append({"file": f"{sub}/{name.lower()}", "bytes": len(data), "from": source})

    def from_imgs(name):
        for rel, img in arcs.items():
            if name.lower() in img.entries:
                return rel, img.read(name)
        return None, None

    if a.preset == "starter":
        for sub, names, exts in PRESET_STARTER:
            for base in names:
                for ext in exts:
                    rel, data = from_imgs(f"{base}.{ext}")
                    if data is None:
                        missing.append(f"{base}.{ext}")
                    else:
                        save(sub, f"{base}.{ext}", data, rel)
        for sub, rel in LOOSE_STARTER:
            p = find_ci(game, rel)
            if p is None:
                missing.append(rel)
            elif os.path.isdir(p):
                for fn in sorted(os.listdir(p)):
                    fp = os.path.join(p, fn)
                    if os.path.isfile(fp):
                        with open(fp, "rb") as fh:
                            save(sub, fn, fh.read(), rel)
            else:
                with open(p, "rb") as fh:
                    save(sub, os.path.basename(p), fh.read(), rel)

    for pat in a.get:
        found = False
        for rel, img in arcs.items():
            for name in sorted(img.entries):
                if fnmatch.fnmatch(name, pat.lower()):
                    save("extra", name, img.read(name), rel)
                    found = True
        if not found:
            missing.append(pat)

    with open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    total = sum(m["bytes"] for m in manifest)
    print(f"{len(manifest)} arquivos extraidos para {out} ({total / 1e6:.1f} MB)")
    if missing:
        print("nao encontrados (ok se nao existirem na sua versao):", ", ".join(missing))

    if not a.no_zip and manifest:
        zpath = out.rstrip("/\\") + "_starter.zip"
        base = os.path.dirname(out)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for root, _, files in os.walk(out):
                for fn in files:
                    fp = os.path.join(root, fn)
                    z.write(fp, os.path.relpath(fp, base))
        print(f"zip: {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
