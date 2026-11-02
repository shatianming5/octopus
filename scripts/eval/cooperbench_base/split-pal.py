import pathlib
S = pathlib.Path("/home/user/harbor-udocker/s1b")
names = [l.strip() for l in open(S / "shard68.txt") if l.strip()]
half = len(names) // 2
a, b = names[:half], names[half:]
d = S / "tasks-pal"
for old in d.iterdir():
    if old.is_symlink() and old.name in set(b):
        old.unlink()
(S / "shard134-pal.txt").write_text("\n".join(a) + "\n")
(S / "shard136-pal.txt").write_text("\n".join(b) + "\n")
print("134 first half", len(a), "136 second half", len(b), "links left", len(list(d.iterdir())))
