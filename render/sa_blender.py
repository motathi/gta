"""Utilitarios para montar cenas com assets do GTA San Andreas no Blender (rodar com `blender -b -P`).

- importa .dff/.txd pelo add-on DragonFF;
- aplica animacoes .ifp (ANP3, lidas por ifp_anp3.py) por PoseSampler, casando ossos pelo bone_id
  do HAnim. A conta de retarget (rotacao de pose = diferenca entre a rotacao de repouso local do osso
  e o quaternion do IFP) segue a abordagem do add-on io_scene_gta_ifp (Psycrow101), reimplementada aqui;
- limpa veiculos (remove colisao/sombra/pecas danificadas/LOD, clona as rodas, recolore a pintura);
- trata texturas ausentes (ex.: vehicle.txd nao incluido) para nao renderizar magenta.
"""
import bisect
import os

import addon_utils
import bpy
from mathutils import Matrix, Quaternion

DRAGONFF_MODULES = ("bl_ext.blender_org.dragonff", "bl_ext.user_default.dragonff", "DragonFF", "dragonff")

# cores-codigo do San Andreas (bytes) que o jogo troca por cor de pintura / luzes
MARKERS = {
    "primary": (60, 255, 0), "secondary": (255, 0, 175), "tertiary": (0, 255, 255), "quaternary": (255, 255, 0),
    "light_front_l": (255, 175, 0), "light_front_r": (0, 255, 200),
    "light_rear_l": (185, 255, 0), "light_rear_r": (255, 60, 0),
}


# --------------------------------------------------------------------------- DragonFF / importacao
def enable_dragonff():
    last = None
    for mod in DRAGONFF_MODULES:
        try:
            if addon_utils.enable(mod, default_set=False, persistent=False) is not None:
                return mod
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError("add-on DragonFF nao encontrado (Edit > Preferences > Get Extensions > DragonFF). %s" % last)


def import_dff(paths):
    """Importa uma lista de caminhos .txd/.dff (o DragonFF le os .txd primeiro) e devolve os objetos novos."""
    paths = [os.path.abspath(p) for p in paths]
    directory = os.path.commonpath([os.path.dirname(p) for p in paths])
    names = [os.path.relpath(p, directory) for p in paths]
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.dff(directory=directory + os.sep, files=[{"name": n} for n in names])
    return [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before]


def descendants(obj):
    out = []
    for c in obj.children:
        out.append(c)
        out.extend(descendants(c))
    return out


def remove_objects(objs):
    for o in objs:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)


# --------------------------------------------------------------------------- animacao
class PoseSampler:
    """Aplica uma animacao ANP3 a um armature importado pelo DragonFF, avaliando a pose em qualquer instante."""

    def __init__(self, arm_obj, anim, strip_forward=True):
        self.arm, self.anim = arm_obj, anim
        self.last = max(anim.last_tick, 1)
        bones_by_id = {b.get("bone_id"): b for b in arm_obj.data.bones}
        self.tracks, self.missing = [], []
        root = anim.root()
        y0 = root.keys[0].pos[1] if root else 0.0
        self.stride = anim.root_displacement()[1] if strip_forward else 0.0   # metros por ciclo
        self.speed = self.stride / anim.duration if anim.duration > 0 else 0.0  # m/s

        for tr in anim.tracks:
            bone = bones_by_id.get(tr.bone_id)
            if bone is None:
                self.missing.append(tr.name)
                continue
            rest = bone.matrix_local
            if bone.parent:
                parent_mat = bone.parent.matrix_local
                local_rot = (parent_mat.inverted_safe() @ rest).to_quaternion()
            else:
                parent_mat = Matrix.Identity(4)
                local_rot = rest.to_quaternion()
            ticks, rots, locs, prev = [], [], [], None
            for k in tr.keys:
                q = local_rot.rotation_difference(Quaternion(k.rot))
                if prev is not None and q.dot(prev) < 0:
                    q.negate()  # continuidade de sinal para o slerp nao dar a volta longa
                prev = q
                ticks.append(k.tick)
                rots.append(q)
                if k.pos is not None:
                    p = (k.pos[0], y0 if strip_forward else k.pos[1], k.pos[2])  # sem o avanco: o objeto anda
                    locs.append((rest.inverted() @ (parent_mat @ Matrix.Translation(p))).to_translation())
            pb = arm_obj.pose.bones[bone.name]
            pb.rotation_mode = "QUATERNION"
            self.tracks.append((pb, ticks, rots, locs or None))

    def set_pose(self, seconds, loop=True):
        tick = seconds * 60.0
        if loop:
            tick %= self.last
        for pb, ticks, rots, locs in self.tracks:
            i = max(bisect.bisect_right(ticks, tick) - 1, 0)
            j = min(i + 1, len(ticks) - 1)
            f = 0.0 if j == i else min(max((tick - ticks[i]) / (ticks[j] - ticks[i]), 0.0), 1.0)
            pb.rotation_quaternion = rots[i].slerp(rots[j], f)
            if locs:
                pb.location = locs[i].lerp(locs[j], f)


# --------------------------------------------------------------------------- materiais
def principled(mat):
    if not mat or not mat.node_tree:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def classify_marker(rgb):
    for name, c in MARKERS.items():
        if all(abs(rgb[i] - c[i] / 255.0) < 0.02 for i in range(3)):
            return name
    return None


def materials_of(objs):
    seen, out = set(), []
    for o in objs:
        for slot in o.material_slots:
            if slot.material and slot.material.name not in seen:
                seen.add(slot.material.name)
                out.append(slot.material)
    return out


def drop_missing_textures(mats):
    """Remove nos de imagem sem pixels (textura ausente do .txd) para a cor do material do DFF valer.

    Devolve {nome_do_material: [rotulos das texturas que faltavam]}.
    """
    lost = {}
    for mat in mats:
        if not mat.node_tree:
            continue
        for node in list(mat.node_tree.nodes):
            if node.type == "TEX_IMAGE" and (node.image is None or not node.image.has_data):
                lost.setdefault(mat.name, []).append((node.label or "?").lower())
                mat.node_tree.nodes.remove(node)
    return lost


def set_principled(mat, **values):
    bsdf = principled(mat)
    if bsdf is None:
        return
    for key, val in values.items():
        name = key.replace("_", " ").title().replace("Ior", "IOR")
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = val


def flat_material(name, rgb, roughness=0.6, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    set_principled(mat, base_color=(*rgb, 1.0), roughness=roughness, metallic=metallic)
    return mat


def style_car_materials(objs, paint, lost=None, secondary=(0.02, 0.02, 0.025), headlight=(1.0, 0.92, 0.75),
                        taillight=(0.9, 0.02, 0.02), light_strength=3.0):
    """Troca cores-codigo por pintura/luzes e da acabamento (verniz, vidro, pneu) aos materiais do carro.

    `lost` ({material: [texturas ausentes]}) permite escolher uma cor plausivel quando a textura nao veio.
    """
    lost = lost or {}
    for mat in materials_of(objs):
        bsdf = principled(mat)
        if bsdf is None:
            continue
        base = tuple(bsdf.inputs["Base Color"].default_value)[:3]
        role, lname = classify_marker(base), mat.name.lower()
        labels = " ".join(lost.get(mat.name, []))
        if not role and labels:
            if "tyre" in labels:
                set_principled(mat, base_color=(0.012, 0.012, 0.014, 1.0), roughness=0.9)
                continue
            if "steering" in labels or "dash" in labels:
                set_principled(mat, base_color=(0.03, 0.03, 0.035, 1.0), roughness=0.5)
                continue
            if "plate" in labels:
                set_principled(mat, base_color=(0.7, 0.7, 0.72, 1.0), roughness=0.4)
                continue
            if "carpback" in labels or "grunge" in labels or "scratch" in labels:
                set_principled(mat, base_color=(0.06, 0.06, 0.065, 1.0), roughness=0.7)
                continue
        if role in ("primary", "secondary", "tertiary", "quaternary"):
            col = paint if role == "primary" else secondary
            set_principled(mat, base_color=(*col, 1.0), metallic=0.55 if role == "primary" else 0.1,
                           roughness=0.28, coat_weight=1.0, coat_roughness=0.03)
        elif role and role.startswith("light_"):
            col = headlight if role.startswith("light_front") else taillight
            set_principled(mat, base_color=(*col, 1.0), roughness=0.15, emission_color=(*col, 1.0),
                           emission_strength=light_strength)
        elif lname.startswith("glass") or "windscreen" in lname:
            set_principled(mat, base_color=(0.02, 0.03, 0.04, 1.0), alpha=0.35, roughness=0.03)
        elif lname == "wheel.0" or "tyre" in lname:
            set_principled(mat, base_color=(0.01, 0.01, 0.012, 1.0), roughness=0.85)
        elif lname.startswith("metal"):
            set_principled(mat, base_color=(0.8, 0.82, 0.86, 1.0), metallic=1.0, roughness=0.25)
        else:
            set_principled(mat, roughness=0.55)


# --------------------------------------------------------------------------- veiculos
def prepare_car(root_name, paint, **style):
    """Limpa o veiculo importado, clona as rodas e acerta materiais. Devolve (raiz, rodas, raio_da_roda)."""
    root = bpy.data.objects[root_name]
    junk = [o for o in bpy.data.objects
            if o.name.endswith(".ColMesh") or ".ColSphere" in o.name or ".ColBox" in o.name or "ShadowMesh" in o.name
            or ".ColMesh" in o.name]
    junk += [o for o in descendants(root) if o.name.endswith("_dam") or o.name.endswith("_vlo")]
    remove_objects(junk)

    kids = descendants(root)
    src = next(o for o in kids if o.name == "wheel")  # a unica roda real do DFF (no dummy direito-frente)
    wheels = [src]
    for dummy, mirror in (("wheel_lf_dummy", True), ("wheel_lb_dummy", True), ("wheel_rb_dummy", False)):
        w = src.copy()  # compartilha a malha
        for col in src.users_collection:
            col.objects.link(w)
        w.parent = bpy.data.objects[dummy]
        w.matrix_parent_inverse = Matrix.Identity(4)
        w.location, w.rotation_euler = (0, 0, 0), (0, 0, 0)
        w.scale = (-1, 1, 1) if mirror else (1, 1, 1)
        wheels.append(w)
    radius = max(src.dimensions[1], src.dimensions[2]) / 2

    objs = [o for o in descendants(root) if o.type == "MESH"]
    lost = drop_missing_textures(materials_of(objs))
    style_car_materials(objs, paint, lost=lost, **style)
    return root, wheels, radius, sorted({t for labels in lost.values() for t in labels})


# --------------------------------------------------------------------------- peds
SKIN_IDS = {4, 5, 6, 7, 8, 24, 25, 26, 34, 35, 36}            # pescoco, cabeca, sobrancelhas, mandibula, maos/dedos
SHOE_IDS = {43, 44, 53, 54}                                    # pes


def paint_ped_by_weights(mesh_obj, arm_obj, skin, suit, shoes, hair, hair_above=0.74):
    """Sem o .txd do ped, pinta cada vertice pelo osso dominante (pele / terno / sapato / cabelo)."""
    bone_id_by_group = {}
    for vg in mesh_obj.vertex_groups:
        b = arm_obj.data.bones.get(vg.name)
        bone_id_by_group[vg.index] = b.get("bone_id") if b else None
    me = mesh_obj.data
    colors = {"skin": skin, "suit": suit, "shoes": shoes, "hair": hair}

    def klass(v):
        best = max(v.groups, key=lambda g: g.weight, default=None)
        bid = bone_id_by_group.get(best.group) if best else None
        if bid in SKIN_IDS:
            return "hair" if (v.co.z > hair_above and bid in (5, 6, 7)) else "skin"
        return "shoes" if bid in SHOE_IDS else "suit"

    vclass = [klass(v) for v in me.vertices]
    attr = me.color_attributes.new("SAcol", "FLOAT_COLOR", "CORNER")  # por face: bordas nitidas (sem degrade)
    for poly in me.polygons:
        cls = [vclass[i] for i in poly.vertices]
        col = (*colors[max(set(cls), key=cls.count)], 1.0)
        for li in poly.loop_indices:
            attr.data[li].color = col

    mat = bpy.data.materials.new("PedFlat")
    mat.use_nodes = True
    nt = mat.node_tree
    node = nt.nodes.new("ShaderNodeAttribute")
    node.attribute_name = "SAcol"
    nt.links.new(node.outputs["Color"], principled(mat).inputs["Base Color"])
    set_principled(mat, roughness=0.55, specular_ior_level=0.2)
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)
