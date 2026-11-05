#!/usr/bin/env python3
"""Cena do video: rua ao por do sol com um ped do San Andreas caminhando e correndo e um carro estacionado.

Roda DENTRO do Blender (headless):
  blender -b -P scene_video.py -- --assets <pasta> --out <pasta_de_frames> [--res 1280x720] [--samples 24]
                                  [--fps 30] [--frames 0:450 | 45,105,180] [--ped wuzimu] [--car infernus]

Usa dos assets do jogo: ped .dff (+ .txd se houver), carro .dff/.txd (+ vehicle.txd se houver), ped.ifp
(animacoes WALK_player / run_player) e props (lamppost1, fire_hydrant, trafficlight1). O resto (rua,
predios, ceu) e procedural. Os frames saem em PNG; o ffmpeg monta o MP4 (ver make_video.py).
"""
import argparse
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import ifp_anp3  # noqa: E402
import sa_blender as sa  # noqa: E402

# ------------------------------------------------------------------ parametros da coreografia
SUN_AZIMUTH, SUN_ELEVATION = 135.0, 8.0     # azimute em graus (0 = +X, 90 = +Y); sol baixo, para -X/+Y
ROAD_HALF, SIDEWALK_OUT = 7.0, 11.0          # meia largura da pista e borda externa da calcada (m)
PED_X, CAR_X, CAR_Y = 9.4, 5.0, 4.0          # onde o ped anda (calcada) e onde o carro fica (m)
CUT_AB, CUT_BC, TOTAL = 4.5, 9.5, 15.0       # cortes entre planos (s) e duracao total
WALK_START_Y, RUN_START_Y = -9.0, -12.0


def lerp(a, b, u):
    return a + (b - a) * u


def ease(u):
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------ argumentos / ativos
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, help="pasta com os arquivos do jogo (busca recursiva por nome)")
    ap.add_argument("--out", required=True, help="pasta de saida dos frames PNG")
    ap.add_argument("--ped", default="wuzimu")
    ap.add_argument("--car", default="infernus")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", default="", help="'a:b' (intervalo) ou lista 'f1,f2,...'; vazio = todos")
    ap.add_argument("--paint", default="0.55,0.02,0.02", help="cor da pintura do carro (RGB linear)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--engine", choices=("cycles", "eevee"), default="cycles",
                    help="cycles = path tracing (mais fiel); eevee = tempo real na GPU (bem mais rapido)")
    ap.add_argument("--device", choices=("CPU", "GPU"), default="CPU", help="GPU so vale para o Cycles")
    return ap.parse_args(argv)


def setup_engine(sc, args):
    if args.engine == "eevee":
        names = [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        sc.render.engine = next(n for n in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if n in names)  # 4.2-4.5 / 5.x
        for attr, val in (("taa_render_samples", args.samples), ("use_raytracing", True), ("use_shadows", True)):
            try:
                setattr(sc.eevee, attr, val)
            except (AttributeError, TypeError):
                pass
        print("motor:", sc.render.engine)
        return
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    if args.device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
                prefs.refresh_devices() if hasattr(prefs, "refresh_devices") else prefs.get_devices()
            except (TypeError, ValueError, RuntimeError):
                continue
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type != "CPU"
                sc.cycles.device = "GPU"
                print("Cycles GPU (%s): %s" % (backend, ", ".join(d.name for d in gpus)))
                return
        print("AVISO: nenhuma GPU encontrada para o Cycles; usando CPU")
    else:
        print("Cycles CPU")


def find_asset(root, name):
    name = name.lower()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower() == name:
                return os.path.join(dirpath, f)
    return None


def need_asset(root, name):
    p = find_asset(root, name)
    if p is None:
        sys.exit("arquivo %r nao encontrado em %s (use tools/sa_extract.py ou aponte --assets para a pasta certa)" % (name, root))
    return p


# ------------------------------------------------------------------ geometria / materiais auxiliares
def make_box(sc, name, center, size, mat):
    hx, hy, hz = (s / 2 for s in size)
    verts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz), (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    o = bpy.data.objects.new(name, me)
    o.location = center
    sc.collection.objects.link(o)
    o.data.materials.append(mat)
    return o


def node(nt, kind, **props):
    n = nt.nodes.new(kind)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def math_node(nt, op, a=None, b=None):
    n = node(nt, "ShaderNodeMath", operation=op)
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            nt.links.new(x, n.inputs[i])
    return n


def no_light_sampling(mat):
    """Emissao visivel, mas sem amostragem como luz (muito mais rapido no Cycles)."""
    for attr, val in (("emission_sampling", "NONE"), ("sample_as_light", False)):
        try:
            setattr(mat.cycles, attr, val)
            return
        except (AttributeError, TypeError):
            continue


def asphalt_material():
    mat = bpy.data.materials.new("Asphalt")
    mat.use_nodes = True
    nt = mat.node_tree
    noise = node(nt, "ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 60.0
    ramp = node(nt, "ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.028, 0.028, 0.031, 1)
    ramp.color_ramp.elements[1].color = (0.06, 0.058, 0.06, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], sa.principled(mat).inputs["Base Color"])
    sa.set_principled(mat, roughness=0.88)
    return mat


def concrete_material():
    mat = bpy.data.materials.new("Concrete")
    mat.use_nodes = True
    nt = mat.node_tree
    coord = node(nt, "ShaderNodeTexCoord")
    brick = node(nt, "ShaderNodeTexBrick")
    brick.inputs["Color1"].default_value = (0.36, 0.34, 0.31, 1)
    brick.inputs["Color2"].default_value = (0.33, 0.31, 0.29, 1)
    brick.inputs["Mortar"].default_value = (0.2, 0.19, 0.18, 1)
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Mortar Size"].default_value = 0.012
    brick.inputs["Brick Width"].default_value = 1.6
    brick.inputs["Row Height"].default_value = 1.6
    brick.offset = 0.0
    nt.links.new(coord.outputs["Object"], brick.inputs["Vector"])
    nt.links.new(brick.outputs["Color"], sa.principled(mat).inputs["Base Color"])
    sa.set_principled(mat, roughness=0.9)
    return mat


def building_material():
    """Parede + janelas por coordenadas de mundo (grade 3 x 3,3 m) com luzes acesas aleatorias e neblina de distancia."""
    mat = bpy.data.materials.new("Building")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = node(nt, "ShaderNodeOutputMaterial")
    bsdf = node(nt, "ShaderNodeBsdfPrincipled")
    geo = node(nt, "ShaderNodeNewGeometry")
    sep = node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs["Vector"])
    u = math_node(nt, "DIVIDE", math_node(nt, "ADD", sep.outputs["X"], sep.outputs["Y"]).outputs[0], 3.0)
    v = math_node(nt, "DIVIDE", sep.outputs["Z"], 3.3)
    fu, fv = math_node(nt, "FRACT", u.outputs[0]), math_node(nt, "FRACT", v.outputs[0])

    def window(fr, lo, hi):
        return math_node(nt, "MULTIPLY", math_node(nt, "GREATER_THAN", fr.outputs[0], lo).outputs[0],
                         math_node(nt, "LESS_THAN", fr.outputs[0], hi).outputs[0])

    mask = math_node(nt, "MULTIPLY", window(fu, 0.18, 0.82).outputs[0], window(fv, 0.22, 0.78).outputs[0])
    above = math_node(nt, "GREATER_THAN", sep.outputs["Z"], 0.5)          # janelas desde o terreo (vitrines)
    mask = math_node(nt, "MULTIPLY", mask.outputs[0], above.outputs[0])

    info = node(nt, "ShaderNodeObjectInfo")
    seed = math_node(nt, "MULTIPLY", info.outputs["Random"], 97.0)
    comb = node(nt, "ShaderNodeCombineXYZ")
    nt.links.new(math_node(nt, "FLOOR", u.outputs[0]).outputs[0], comb.inputs["X"])
    nt.links.new(math_node(nt, "FLOOR", v.outputs[0]).outputs[0], comb.inputs["Y"])
    nt.links.new(seed.outputs[0], comb.inputs["Z"])
    wn = node(nt, "ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    nt.links.new(comb.outputs["Vector"], wn.inputs["Vector"])
    shops = math_node(nt, "MULTIPLY", math_node(nt, "LESS_THAN", sep.outputs["Z"], 3.3).outputs[0], 0.3)  # terreo acende mais
    lit = math_node(nt, "GREATER_THAN", math_node(nt, "ADD", wn.outputs["Value"], shops.outputs[0]).outputs[0], 0.76)
    emit = math_node(nt, "MULTIPLY", mask.outputs[0], lit.outputs[0])

    ramp = node(nt, "ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    palette = [(0.0, (0.30, 0.20, 0.14)), (0.2, (0.42, 0.36, 0.28)), (0.4, (0.17, 0.17, 0.21)),
               (0.6, (0.38, 0.12, 0.09)), (0.8, (0.27, 0.32, 0.29))]
    els = ramp.color_ramp.elements
    for el, (pos, col) in zip((els[0], els[1]), palette[:2]):
        el.position, el.color = pos, (*col, 1)
    for pos, col in palette[2:]:
        el = ramp.color_ramp.elements.new(pos)
        el.color = (*col, 1)
    nt.links.new(info.outputs["Random"], ramp.inputs["Fac"])
    # janelas = vidro (rugosidade baixa reflete o ceu); parede = fosca
    rough = math_node(nt, "SUBTRACT", 0.85, math_node(nt, "MULTIPLY", mask.outputs[0], 0.77).outputs[0])
    nt.links.new(rough.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.72, 0.38, 1)
    glow = math_node(nt, "MULTIPLY", emit.outputs[0],
                     math_node(nt, "ADD", 0.5, math_node(nt, "MULTIPLY", wn.outputs["Value"], 1.2).outputs[0]).outputs[0])
    nt.links.new(glow.outputs[0], bsdf.inputs["Emission Strength"])

    # relevo sutil de tijolo (plano XZ/YZ unificado) multiplicando a cor da parede
    uvz = node(nt, "ShaderNodeCombineXYZ")
    nt.links.new(math_node(nt, "ADD", sep.outputs["X"], sep.outputs["Y"]).outputs[0], uvz.inputs["X"])
    nt.links.new(sep.outputs["Z"], uvz.inputs["Y"])
    brick = node(nt, "ShaderNodeTexBrick")
    brick.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1)
    brick.inputs["Color2"].default_value = (0.9, 0.9, 0.9, 1)
    brick.inputs["Mortar"].default_value = (0.7, 0.7, 0.7, 1)
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Mortar Size"].default_value = 0.03
    brick.inputs["Brick Width"].default_value = 0.42
    brick.inputs["Row Height"].default_value = 0.16
    nt.links.new(uvz.outputs["Vector"], brick.inputs["Vector"])
    wall = node(nt, "ShaderNodeMixRGB", blend_type="MULTIPLY")
    wall.inputs["Fac"].default_value = 1.0
    nt.links.new(ramp.outputs["Color"], wall.inputs["Color1"])
    nt.links.new(brick.outputs["Color"], wall.inputs["Color2"])
    mix = node(nt, "ShaderNodeMixRGB")                       # parede (com tijolo) <-> vidro escuro da janela
    mix.inputs["Color2"].default_value = (0.015, 0.02, 0.03, 1)
    nt.links.new(mask.outputs[0], mix.inputs["Fac"])
    nt.links.new(wall.outputs["Color"], mix.inputs["Color1"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    # neblina de distancia: mistura com a cor do horizonte
    cam = node(nt, "ShaderNodeCameraData")
    fog = math_node(nt, "SUBTRACT", 1.0,
                    math_node(nt, "EXPONENT", math_node(nt, "MULTIPLY", cam.outputs["View Distance"], -0.0042).outputs[0]).outputs[0])
    haze = node(nt, "ShaderNodeEmission")
    haze.inputs["Color"].default_value = (1.0, 0.55, 0.3, 1)
    haze.inputs["Strength"].default_value = 2.2
    mixs = node(nt, "ShaderNodeMixShader")
    nt.links.new(fog.outputs[0], mixs.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mixs.inputs[1])
    nt.links.new(haze.outputs["Emission"], mixs.inputs[2])
    nt.links.new(mixs.outputs["Shader"], out.inputs["Surface"])
    no_light_sampling(mat)  # janelas so sao visuais: nao amostrar milhares de triangulos como luzes
    return mat


# ------------------------------------------------------------------ cenario
def build_world(sc):
    world = bpy.data.worlds.new("SunsetSky")
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    sky = node(nt, "ShaderNodeTexSky", sky_type="NISHITA")
    sky.sun_elevation = math.radians(SUN_ELEVATION)
    sky.sun_rotation = math.radians(90.0 - SUN_AZIMUTH)     # calibrado: azimute = 90 graus - rotacao
    for prop, val in (("sun_size", math.radians(1.6)), ("sun_intensity", 1.0), ("air_density", 1.4),
                      ("dust_density", 2.2), ("ozone_density", 1.0)):
        try:
            setattr(sky, prop, val)
        except (AttributeError, TypeError):
            pass
    bg = node(nt, "ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    out = node(nt, "ShaderNodeOutputWorld")
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    sc.world = world


def build_street(sc, rng):
    asphalt, concrete, bmat = asphalt_material(), concrete_material(), building_material()
    yellow = sa.flat_material("PaintYellow", (0.8, 0.52, 0.03), 0.6)
    white = sa.flat_material("PaintWhite", (0.75, 0.75, 0.72), 0.6)
    y0, y1 = -75.0, 260.0
    length, yc = y1 - y0, (y0 + y1) / 2
    make_box(sc, "ground", (0, yc, -0.07), (900, length + 700, 0.1), concrete)
    make_box(sc, "road", (0, yc, -0.05), (2 * ROAD_HALF, length, 0.1), asphalt)
    for side in (-1, 1):
        w = SIDEWALK_OUT - ROAD_HALF
        make_box(sc, "sidewalk", (side * (ROAD_HALF + w / 2), yc, 0.08), (w, length, 0.16), concrete)
        make_box(sc, "line_edge", (side * 6.5, yc, 0.006), (0.14, length, 0.01), white)
        make_box(sc, "line_yellow", (side * 0.13, yc, 0.006), (0.1, length, 0.01), yellow)
        y = y0
        while y < y1:
            make_box(sc, "line_dash", (side * 3.2, y + 1.5, 0.006), (0.12, 3.0, 0.01), white)
            y += 9.0
    # predios dos dois lados
    for side in (-1, 1):
        y = y0 - 10
        while y < y1:
            wy, depth = rng.uniform(12, 24), rng.uniform(10, 18)
            h = rng.choice((rng.uniform(8, 14), rng.uniform(12, 24), rng.uniform(20, 38)))
            make_box(sc, "bld", (side * (SIDEWALK_OUT + 1.5 + depth / 2 + rng.uniform(0, 1)), y + wy / 2, h / 2), (depth, wy, h), bmat)
            y += wy + rng.uniform(0.5, 2.5)
    for _ in range(22):  # skyline ao fundo
        w, h = rng.uniform(16, 32), rng.uniform(45, 125)
        make_box(sc, "tower", (rng.uniform(-60, 60), rng.uniform(275, 380), h / 2), (w, rng.uniform(16, 30), h), bmat)
    build_palms(sc, rng)


def make_palm(sc, name, loc, height, rng, trunk_mat, leaf_mat):
    """Palmeira procedural: tronco afunilado e levemente curvo + frondes em faixa (dupla face)."""
    verts, faces, mats = [], [], []
    segs, rings = 8, 12
    lean = (rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7))
    for i in range(rings + 1):
        u = i / rings
        cx, cy, cz, r = lean[0] * u * u, lean[1] * u * u, height * u, lerp(0.27, 0.15, u)
        for k in range(segs):
            a = 2 * math.pi * k / segs
            verts.append((cx + r * math.cos(a), cy + r * math.sin(a), cz))
    for i in range(rings):
        for k in range(segs):
            a, b = i * segs + k, i * segs + (k + 1) % segs
            faces.append((a, b, b + segs, a + segs))
            mats.append(0)
    top = (lean[0], lean[1], height)
    n_fronds, n_seg = 11, 7
    for j in range(n_fronds):
        yaw = 2 * math.pi * j / n_fronds + rng.uniform(-0.25, 0.25)
        length, droop, rise = rng.uniform(2.6, 3.5), rng.uniform(1.0, 1.7), rng.uniform(0.6, 1.2)
        dx, dy, sx, sy = math.cos(yaw), math.sin(yaw), -math.sin(yaw), math.cos(yaw)
        base = len(verts)
        for s in range(n_seg + 1):
            t = s / n_seg
            w = 0.05 + 0.6 * math.sin(math.pi * t) ** 0.7
            px, py, pz = top[0] + dx * length * t, top[1] + dy * length * t, top[2] + rise * t - droop * t * t
            verts += [(px - sx * w, py - sy * w, pz - 0.10 * w), (px, py, pz + 0.06 * w), (px + sx * w, py + sy * w, pz - 0.10 * w)]
        for s in range(n_seg):
            a, b = base + s * 3, base + (s + 1) * 3
            faces += [(a, a + 1, b + 1, b), (a + 1, a + 2, b + 2, b + 1)]
            mats += [1, 1]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    for p, m in zip(me.polygons, mats):
        p.material_index = m
    me.materials.append(trunk_mat)
    me.materials.append(leaf_mat)
    o = bpy.data.objects.new(name, me)
    o.location = loc
    sc.collection.objects.link(o)
    return o


def build_palms(sc, rng):
    trunk = sa.flat_material("PalmTrunk", (0.2, 0.14, 0.09), 0.9)
    leaf = sa.flat_material("PalmLeaf", (0.04, 0.16, 0.035), 0.55)
    for side in (-1, 1):
        y = -60.0
        while y < 250:
            make_palm(sc, "palm", (side * 10.4, y, 0.16), rng.uniform(7.5, 11.0), rng, trunk, leaf)
            y += 24.0


def ground_offset(obj):
    """Altura para o objeto (origem nao necessariamente no chao) apoiar no z = 0."""
    return -min(c[2] for c in obj.bound_box)


def place_props(sc, assets):
    """Postes, hidrante e semaforo do San Andreas. Devolve a lista de objetos criados."""
    specs = {}
    for name in ("lamppost1", "fire_hydrant", "trafficlight1"):
        p = find_asset(assets, name + ".dff")
        if p:
            before = set(bpy.data.objects.keys())
            sa.import_dff([p])
            specs[name] = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before]
    steel = sa.flat_material("PropSteel", (0.11, 0.12, 0.13), 0.45, metallic=0.6)
    made = []

    def clean(objs, mat):
        mesh = next(o for o in objs if o.type == "MESH")
        lights = [o for o in objs if o.type == "LIGHT"]
        heads = [o.matrix_world.translation.copy() for o in lights]
        sa.remove_objects([o for o in objs if o.type in ("LIGHT", "EMPTY")])
        mesh.data.materials.clear()
        mesh.data.materials.append(mat)
        return mesh, heads

    def glow(name, pos, color, strength, parent, radius=0.12):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=pos, segments=12, ring_count=8)
        s = bpy.context.object
        s.name = name
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        sa.set_principled(m, base_color=(*color, 1), emission_color=(*color, 1), emission_strength=strength)
        no_light_sampling(m)
        s.data.materials.append(m)
        s.parent = parent
        s.matrix_parent_inverse = parent.matrix_world.inverted()
        return s

    if "lamppost1" in specs:
        mesh, heads = clean(specs["lamppost1"], steel)
        zoff = ground_offset(mesh)
        arm_dir = -1 if (sum(c[0] for c in mesh.bound_box) / 8) < 0 else 1   # lado para onde o braco aponta
        mesh.hide_render = True
        for side in (-1, 1):
            y = -66.0
            while y < 250:
                o = mesh.copy()
                for c in mesh.users_collection:
                    c.objects.link(o)
                o.hide_render = False
                o.location = (side * 7.9, y, zoff)
                o.rotation_euler.z = 0 if (arm_dir * side) < 0 else math.pi
                made.append(o)
                for h in heads[:1]:
                    pos = o.matrix_world @ Vector(h)
                    glow("lamp_bulb", (pos.x, pos.y, pos.z), (1.0, 0.8, 0.5), 25.0, o, 0.14)
                y += 24.0
        mesh.hide_viewport = True
    if "fire_hydrant" in specs:
        yellow = sa.flat_material("Hydrant", (0.75, 0.45, 0.02), 0.45)
        mesh, _ = clean(specs["fire_hydrant"], yellow)
        mesh.location = (7.6, -3.5, ground_offset(mesh) + 0.16)
        made.append(mesh)
    if "trafficlight1" in specs:
        mesh, heads = clean(specs["trafficlight1"], sa.flat_material("TrafficPole", (0.09, 0.09, 0.1), 0.5, metallic=0.4))
        mesh.location = (ROAD_HALF + 0.9, 62.0, ground_offset(mesh) + 0.16)
        mesh.rotation_euler.z = 0 if (sum(c[0] for c in mesh.bound_box) / 8) < 0 else math.pi
        made.append(mesh)
        for h in heads:
            pos = mesh.matrix_world @ Vector(h)
            level = round(h.z, 1)
            top = max(x.z for x in heads)
            bottom = min(x.z for x in heads)
            col, power = ((1.0, 0.05, 0.03), 4.0) if abs(h.z - top) < 0.05 else (((0.1, 1.0, 0.2), 30.0) if abs(h.z - bottom) < 0.05 else ((1.0, 0.7, 0.05), 2.0))
            glow("tl_light_%.1f" % level, (pos.x, pos.y, pos.z), col, power, mesh, 0.13)
    return made


# ------------------------------------------------------------------ personagens
def setup_ped(assets, name):
    dff, txd = need_asset(assets, name + ".dff"), find_asset(assets, name + ".txd")
    new = sa.import_dff([p for p in (txd, dff) if p])
    arm = next(o for o in new if o.type == "ARMATURE")
    mesh = next(o for o in new if o.type == "MESH" and o.parent == arm)
    root = arm
    while root.parent:
        root = root.parent
    textured = any(n.type == "TEX_IMAGE" and n.image and n.image.has_data
                   for m in sa.materials_of([mesh]) if m.node_tree for n in m.node_tree.nodes)
    if not textured:
        sa.drop_missing_textures(sa.materials_of([mesh]))
        sa.paint_ped_by_weights(mesh, arm, skin=(0.55, 0.32, 0.2), suit=(0.025, 0.03, 0.045),
                                shoes=(0.008, 0.008, 0.008), hair=(0.004, 0.004, 0.004))
    low = min(v.co.z for v in mesh.data.vertices)
    return arm, root, -low, textured


def setup_car(assets, name, paint):
    dff = need_asset(assets, name + ".dff")
    paths = [p for p in (find_asset(assets, name + ".txd"), find_asset(assets, "vehicle.txd"), dff) if p]
    sa.import_dff(paths)
    root, wheels, radius, missing = sa.prepare_car(name, paint)
    return root, wheels, radius, missing


# ------------------------------------------------------------------ camera
def camera_pose(t, ped_y):
    """Devolve (posicao, alvo, lente_mm) para o instante t."""
    if t < CUT_AB:                                   # plano A: acompanha o ped caminhando, de lado
        u = t / CUT_AB
        return (Vector((lerp(3.6, 5.2, ease(u)), ped_y + 1.0, lerp(1.1, 1.3, u))),
                Vector((PED_X, ped_y + 1.6, 1.0)), 36.0)
    if t < CUT_BC:                                   # plano B: orbita o carro, o ped passa ao fundo
        u = (t - CUT_AB) / (CUT_BC - CUT_AB)
        a, r = math.radians(lerp(220, 150, ease(u))), lerp(5.8, 4.4, u)
        return (Vector((CAR_X + r * math.cos(a), CAR_Y + r * math.sin(a), lerp(0.85, 1.5, u))),
                Vector((CAR_X + 0.3, CAR_Y, 0.8)), 35.0)
    u = (t - CUT_BC) / (TOTAL - CUT_BC)              # plano C: persegue o ped correndo, de tras, contra o sol
    sh = Vector((math.sin(t * 9.1), math.sin(t * 7.3 + 1), math.sin(t * 5.7 + 2))) * 0.012
    return (Vector((6.6, ped_y - lerp(5.5, 3.8, u), lerp(0.95, 1.2, u))) + sh,
            Vector((PED_X - 1.5, ped_y + 2.5, 1.05)), 26.0)


def aim(cam, pos, target, lens):
    cam.location = pos
    cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens


# ------------------------------------------------------------------ principal
def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    print("DragonFF:", sa.enable_dragonff())
    rng = random.Random(7)

    build_world(sc)
    build_street(sc, rng)
    props = place_props(sc, args.assets)

    arm, ped_root, hip_z, textured = setup_ped(args.assets, args.ped)
    _, anims = ifp_anp3.load(need_asset(args.assets, "ped.ifp"))
    walk = sa.PoseSampler(arm, ifp_anp3.find(anims, "WALK_player"))
    run = sa.PoseSampler(arm, ifp_anp3.find(anims, "run_player"))
    paint = tuple(float(x) for x in args.paint.split(","))
    car, wheels, radius, missing = setup_car(args.assets, args.car, paint)
    car.location, car.rotation_euler = (CAR_X, CAR_Y, radius * 2), (0, 0, 0)
    print(f"ped texturizado={textured} hip_z={hip_z:.2f} | caminhada {walk.speed:.2f} m/s, corrida {run.speed:.2f} m/s | "
          f"props={len(props)} | texturas do carro ausentes (cor plana): {missing}")

    cam_data = bpy.data.cameras.new("cam")
    cam_data.clip_end = 3000
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    w, h = (int(x) for x in args.res.lower().split("x"))
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.fps = args.fps
    setup_engine(sc, args)
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces, sc.cycles.transmission_bounces = 2, 2, 3
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.05
    sc.cycles.adaptive_min_samples = 6
    sc.render.use_persistent_data = True   # mantem BVH/dados estaticos entre os frames do loop
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    sc.view_settings.exposure = -0.8

    total = int(round(TOTAL * args.fps))
    if ":" in args.frames:
        a, b = args.frames.split(":")[:2]
        frames = list(range(int(a), int(b)))
    elif args.frames:
        frames = [int(x) for x in args.frames.split(",")]
    else:
        frames = list(range(total))
    os.makedirs(args.out, exist_ok=True)

    t_start = time.time()
    done = 0
    for f in frames:
        path = os.path.join(args.out, "f_%04d.png" % f)
        if os.path.exists(path) and not args.overwrite:
            continue
        t = f / args.fps
        if t < CUT_BC:
            sampler, tt, y = walk, t, WALK_START_Y + walk.speed * t
        else:
            sampler, tt, y = run, t - CUT_BC, RUN_START_Y + run.speed * (t - CUT_BC)
        sampler.set_pose(tt)
        ped_root.location = (PED_X, y, hip_z)
        pos, target, lens = camera_pose(t, y)
        aim(cam, pos, target, lens)
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        done += 1
        el = time.time() - t_start
        print(f"[{done}/{len(frames)}] frame {f} (t={t:.2f}s)  {el / done:.1f}s/frame  restante ~{el / done * (len(frames) - done) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
