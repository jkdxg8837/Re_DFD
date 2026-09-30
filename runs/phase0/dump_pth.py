"""Phase 0: dump the small .pth side files of the released checkpoints without importing torch
(the Sulis login node has a ~540 MB memory cap). Torch objects are replaced by placeholders."""
import pickle
import pprint
import zipfile


class _Stub:
    def __init__(self, *a, **k):
        self.a = a

    def __setstate__(self, s):
        self.s = s

    def __repr__(self):
        return f"<torch obj {self.a!r}>"


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch") or module.startswith("collections") and name == "OrderedDict":
            return dict if name == "OrderedDict" else (lambda *a, **k: _Stub(module, name, *a[:2]))
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return f"<storage {pid[1:3]}>"


for p in __import__("sys").argv[1:] or ["checkpoints/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000.pth",
          "checkpoints/wan_t2v_checkpoints_release/ours_wan_t2v_vipe/0000040.pth"]:
    z = zipfile.ZipFile(p)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    print("=" * 80, "\n", p)
    pprint.pprint(_Unpickler(z.open(pkl)).load(), width=150, compact=True)
