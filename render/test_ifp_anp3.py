"""Testes do leitor ANP3 com um arquivo sintetico (nenhum dado da Rockstar).

  python -m unittest -v render/test_ifp_anp3.py
Se SA_PED_IFP apontar para um ped.ifp real, tambem valida que todos os quaternions tem norma ~1.
"""
import math
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ifp_anp3  # noqa: E402


def _name(s, n=24):
    return s.encode().ljust(n, b"\0")


def build_anp3(anims):
    """anims: [(nome, [(osso, bone_id, [(tick, (w, x, y, z), pos_ou_None)])])]"""
    body = b""
    for name, bones in anims:
        body += _name(name) + struct.pack("<III", len(bones), 0, 1)
        for bname, bid, keys in bones:
            has_pos = keys[0][2] is not None
            body += _name(bname) + struct.pack("<IIi", 4 if has_pos else 3, len(keys), bid)
            for tick, (w, x, y, z), pos in keys:
                body += struct.pack("<hhhhh", round(x * 4096), round(y * 4096), round(z * 4096), round(w * 4096), tick)
                if has_pos:
                    body += struct.pack("<hhh", *(round(p * 1024) for p in pos))
    header = _name("teste") + struct.pack("<I", len(anims))
    return b"ANP3" + struct.pack("<I", len(header) + len(body)) + header + body


S = math.sqrt(0.5)
SAMPLE = [
    ("WALK_test", [
        ("Normal", 0, [(0, (1, 0, 0, 0), (0.0, 0.0, -0.03)), (36, (1, 0, 0, 0), (0.0, 0.9, 0.02)), (72, (1, 0, 0, 0), (0.0, 1.5, -0.03))]),
        (" Pelvis", 1, [(0, (1, 0, 0, 0), None), (72, (S, S, 0, 0), None)]),
    ]),
    ("idle_test", [("Normal", 0, [(0, (1, 0, 0, 0), (0, 0, 0)), (120, (1, 0, 0, 0), (0, 0, 0))])]),
]


class Anp3Tests(unittest.TestCase):
    def test_roundtrip(self):
        name, anims = ifp_anp3.parse(build_anp3(SAMPLE))
        self.assertEqual(name, "teste")
        self.assertEqual([a.name for a in anims], ["WALK_test", "idle_test"])
        walk = ifp_anp3.find(anims, "walk_TEST")  # sem diferenciar maiusculas
        self.assertEqual([(t.name, t.bone_id) for t in walk.tracks], [("Normal", 0), ("Pelvis", 1)])
        self.assertEqual([k.tick for k in walk.tracks[0].keys], [0, 36, 72])
        self.assertAlmostEqual(walk.duration, 1.2)
        dx, dy, dz = walk.root_displacement()
        self.assertAlmostEqual(dy, 1.5, delta=1 / 1024)
        self.assertIsNone(walk.tracks[1].keys[0].pos)
        w, x, y, z = walk.tracks[1].keys[1].rot
        self.assertAlmostEqual(w, S, delta=1 / 4096)
        self.assertAlmostEqual(x, S, delta=1 / 4096)

    def test_ignores_trailing_garbage(self):
        data = build_anp3(SAMPLE) + b"\xff" * 5000  # o ped.ifp real tem lixo/padding depois do fim declarado
        _, anims = ifp_anp3.parse(data)
        self.assertEqual(len(anims), 2)

    def test_rejects_other_formats(self):
        with self.assertRaises(ValueError):
            ifp_anp3.parse(b"ANPK" + b"\0" * 64)

    def test_truncated_file(self):
        data = build_anp3(SAMPLE)
        for cut in (60, len(data) - 20):  # no cabecalho da 1a animacao / no meio dos frames da ultima
            with self.assertRaises(ValueError, msg="corte em %d" % cut):
                ifp_anp3.parse(data[:cut])

    def test_find_missing_animation(self):
        _, anims = ifp_anp3.parse(build_anp3(SAMPLE))
        with self.assertRaises(KeyError):
            ifp_anp3.find(anims, "nao_existe")

    @unittest.skipUnless(os.environ.get("SA_PED_IFP"), "defina SA_PED_IFP=caminho/do/ped.ifp para o teste com arquivo real")
    def test_real_file_quaternions_are_unit(self):
        _, anims = ifp_anp3.load(os.environ["SA_PED_IFP"])
        self.assertTrue(anims)
        for a in anims:
            for t in a.tracks:
                for k in t.keys:
                    self.assertAlmostEqual(sum(c * c for c in k.rot) ** 0.5, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
