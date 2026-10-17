# Exodus look calibration

The stylizer viewport uses ShaderShift Burley + dual-lobe GGX, AgX Punchy, and SMAA 1x.
This folder holds a capture script Steven runs in the Unreal Editor, and a diff the agent runs.

## Capture (Steven, in the editor)

1. Open Exodus in the Unreal Editor.
2. `File > Execute Python Script` and choose `capture_sphere.py`, or paste it into the Python console.
3. It writes `tools/groundtruth/ue_sphere.png` (512² HighResShot of a dielectric sphere, roughness 0.35, under a directional light at pitch -40 / yaw 128).

SMAA must be on (`r.AntiAliasingMethod=5`). Auto-exposure stays off.

## Diff (agent)

```
python tools/groundtruth/diff_capture.py tools/groundtruth/ue_sphere.png
```

Prints mean / max abs on the sphere (not the background). Target is a documented error band, not pixel identity: shadows and reflection captures in Unreal will not match the stylizer's split-sum IBL.
