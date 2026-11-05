#!/usr/bin/env python3
"""Leitor de animacoes .ifp do GTA San Andreas (formato ANP3), em Python puro (sem bpy).

Formato (little-endian), como validado em ped.ifp:
  "ANP3", uint32 tamanho (sem contar os 8 primeiros bytes), char[24] nome, uint32 n_animacoes
  animacao: char[24] nome, uint32 n_ossos, uint32 bytes_de_frames, uint32 unk
    osso:   char[24] nome, uint32 tipo (3 = so rotacao, 4 = rotacao + posicao),
            uint32 n_frames, int32 bone_id (ID HAnim do esqueleto do ped)
      frame: int16 qx,qy,qz,qw (valor/4096), int16 tempo (ticks de 1/60 s),
             [int16 px,py,pz (valor/1024, metros) se tipo 4]

Uso:  python ifp_anp3.py caminho/ped.ifp        # lista as animacoes
"""
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

TICKS_PER_SECOND = 60.0


@dataclass
class Key:
    tick: int
    rot: Tuple[float, float, float, float]            # quaternion (w, x, y, z)
    pos: Optional[Tuple[float, float, float]] = None  # metros; so no osso raiz


@dataclass
class Track:
    name: str
    bone_id: int
    keys: List[Key] = field(default_factory=list)


@dataclass
class Animation:
    name: str
    tracks: List[Track]

    @property
    def last_tick(self) -> int:
        return max((t.keys[-1].tick for t in self.tracks if t.keys), default=0)

    @property
    def duration(self) -> float:
        return self.last_tick / TICKS_PER_SECOND

    def root(self) -> Optional[Track]:
        return next((t for t in self.tracks if t.keys and t.keys[0].pos is not None), None)

    def root_displacement(self) -> Tuple[float, float, float]:
        """Deslocamento do osso raiz entre o primeiro e o ultimo frame (metros)."""
        r = self.root()
        if r is None:
            return (0.0, 0.0, 0.0)
        a, b = r.keys[0].pos, r.keys[-1].pos
        return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _str(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1")


def parse(data: bytes):
    """Retorna (nome_do_ifp, [Animation]). Levanta ValueError se nao for ANP3 valido."""
    if data[:4] != b"ANP3":
        raise ValueError("formato IFP nao suportado: esperado ANP3 (San Andreas), obtido %r" % data[:4])
    (size,) = struct.unpack_from("<I", data, 4)
    end = min(len(data), size + 8)  # pode haver padding/lixo depois do fim declarado
    name = _str(data[8:32])
    (count,) = struct.unpack_from("<I", data, 32)
    off, anims = 36, []
    for _ in range(count):
        if off + 36 > end:
            raise ValueError("IFP truncado (cabecalho de animacao)")
        aname = _str(data[off:off + 24])
        n_bones = struct.unpack_from("<I", data, off + 24)[0]
        off += 36
        tracks = []
        for _b in range(n_bones):
            if off + 36 > end:
                raise ValueError("IFP truncado (cabecalho de osso)")
            bname = _str(data[off:off + 24])
            ftype, n_keys, bone_id = struct.unpack_from("<IIi", data, off + 24)
            off += 36
            has_pos = ftype == 4
            stride = 16 if has_pos else 10
            if off + stride * n_keys > end:
                raise ValueError("IFP truncado (frames de '%s')" % aname)
            keys = []
            for _k in range(n_keys):
                qx, qy, qz, qw, tick = struct.unpack_from("<hhhhh", data, off)
                pos = None
                if has_pos:
                    px, py, pz = struct.unpack_from("<hhh", data, off + 10)
                    pos = (px / 1024.0, py / 1024.0, pz / 1024.0)
                keys.append(Key(tick, (qw / 4096.0, qx / 4096.0, qy / 4096.0, qz / 4096.0), pos))
                off += stride
            tracks.append(Track(bname.strip(), bone_id, keys))
        anims.append(Animation(aname, tracks))
    return name, anims


def load(path: str):
    with open(path, "rb") as fh:
        return parse(fh.read())


def find(anims: List[Animation], name: str) -> Animation:
    """Busca animacao pelo nome, sem diferenciar maiusculas de minusculas."""
    for a in anims:
        if a.name.lower() == name.lower():
            return a
    raise KeyError("animacao %r nao encontrada; disponiveis: %s" % (name, ", ".join(a.name for a in anims[:30])))


if __name__ == "__main__":
    ifp_name, anims = load(sys.argv[1])
    print("IFP '%s': %d animacoes" % (ifp_name, len(anims)))
    for a in anims:
        dx, dy, dz = a.root_displacement()
        print("  %-24s %2d ossos  %6.2f s  deslocamento da raiz (%.2f, %.2f, %.2f) m" %
              (a.name, len(a.tracks), a.duration, dx, dy, dz))
