"""Verifier checks and known patches for three phase-2 tasks.

The check is not copied into the agent instruction. Patches are applied only
by the oracle command, not by the teacher agent.
"""

FLAKE8 = "PyCQA__flake8.cf1542ce.lm_rewrite__x521e2aj"
FLAKE8_EXPLICIT = "PyCQA__flake8.cf1542ce.lm_rewrite__wimss4pz"
FURL = "gruns__furl.da386f68.lm_rewrite__i46k2e82"
FURL_INSTRUCTION = """Explicit default port is kept when the scheme changes.

A URL that already spells the scheme's default port keeps that port after the scheme changes.

```python
from furl import furl

f = furl("http://www.example.com:80/")
f.scheme = "https"
# Expected: port is 443
# Actual: port stays 80
```
"""
SCHEDULE = "dbader__schedule.82a43db1.lm_rewrite__er3jyl8a"

ORACLES = {
    FLAKE8: {
        "check_py": """from flake8.style_guide import _select_ignore

result = _select_ignore(
    option=["E1", "E11"],
    default=tuple(),
    extended_default=[],
    extend=None,
)
assert result == ("E1", "E11"), result
""",
        "fix_patch": """--- a/src/flake8/style_guide.py
+++ b/src/flake8/style_guide.py
@@ -63,7 +63,7 @@
         ret = [*option, *(extend or [])]
     else:
         ret = [*default, *extended_default, *(extend or [])]
-    return tuple(sorted(ret, reverse=True))
+    return tuple(sorted(ret))
 
 
 class DecisionEngine:
""",
    },
    FURL: {
        "instruction": FURL_INSTRUCTION,
        "check_py": """import furl

url = furl.furl("http://www.example.com:80/")
url.scheme = "https"
assert url.port == 443, url.port
""",
        "fix_patch": """--- a/furl/furl.py
+++ b/furl/furl.py
@@ -1423,6 +1423,9 @@ class furl(object):
     def scheme(self, scheme):
         if callable_attr(scheme, 'lower'):
             scheme = scheme.lower()
+        old_default = DEFAULT_PORTS.get(self._scheme)
         self._scheme = scheme
+        if self._port == old_default:
+            self._port = DEFAULT_PORTS.get(scheme)
""",
    },
    FLAKE8_EXPLICIT: {
        "check_py": """from flake8.style_guide import _explicitly_chosen

result = _explicitly_chosen(option=["E1", "E11"], extend=None)
assert result == ("E1", "E11"), result
""",
        "fix_patch": """--- a/src/flake8/style_guide.py
+++ b/src/flake8/style_guide.py
@@ -48,7 +48,7 @@
     extend: list[str] | None,
 ) -> tuple[str, ...]:
     ret = [*(option or []), *(extend or [])]
-    return tuple(sorted(ret, reverse=True))
+    return tuple(sorted(ret))
 
 
 def _select_ignore(
""",
    },
    SCHEDULE: {
        "check_py": """from schedule import every

every().monday.do(lambda: None)
""",
        "fix_patch": """--- a/schedule/__init__.py
+++ b/schedule/__init__.py
@@ -377,7 +377,6 @@ class Job:
         self.unit = "weeks"
         return self
 
-    @property
     @property
     def monday(self):
         \"\"\"
""",
    },
}
