"""Run inside the Unreal Editor (PythonScriptPlugin).

Places a 100cm dielectric sphere, one directional light (Epic HQ angles),
and takes a 512 HighResShot next to this script.
"""
import os
import unreal

OUT = os.path.join(os.path.dirname(__file__), "ue_sphere.png")
SIZE = 512

editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = editor.get_editor_world()

loc = unreal.Vector(0, 0, 100)
sphere = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, loc)
mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Sphere")
sphere.static_mesh_component.set_static_mesh(mesh)
sphere.set_actor_scale3d(unreal.Vector(1, 1, 1))

mat = unreal.EditorAssetLibrary.load_asset("/Engine/EngineMaterials/DefaultMaterial")
# A MIC with roughness 0.35 / base 0.7 0.15 0.08 is preferred if present.
sphere.static_mesh_component.set_material(0, mat)

light = unreal.EditorLevelLibrary.spawn_actor_from_class(
    unreal.DirectionalLight, unreal.Vector(0, 0, 400))
light.set_actor_rotation(unreal.Rotator(-40.0, 128.0, 0.0), False)

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(
    unreal.CameraActor, unreal.Vector(0, -250, 100))
cam.set_actor_rotation(unreal.Rotator(0, 90, 0), False)

unreal.AutomationLibrary.take_high_res_screenshot(
    SIZE, SIZE, OUT, cam, False, False, 1.0, False)
unreal.log(f"QuixelStylizer groundtruth wrote {OUT}")
