# save this as fix_patch6.py in your PumpSmart_Project folder
# then run: python fix_patch6.py

import re

FILE = r"C:\Users\user\Desktop\PumpSmart_Project\src\module_05_physics_engine.py"

with open(FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace ALL unicode box-drawing / special dash characters with plain ASCII dash
UNICODE_CHARS = [
    '\u2500',  # ─  BOX DRAWINGS LIGHT HORIZONTAL
    '\u2550',  # ═  BOX DRAWINGS DOUBLE HORIZONTAL  
    '\u2502',  # │  BOX DRAWINGS LIGHT VERTICAL
    '\u2014',  # —  EM DASH
    '\u2013',  # –  EN DASH
    '\u00b7',  # ·  MIDDLE DOT
]

fixed = content
for char in UNICODE_CHARS:
    fixed = fixed.replace(char, '-')

with open(FILE, 'w', encoding='utf-8') as f:
    f.write(fixed)

# Report what was found
found = [hex(ord(c)) for c in UNICODE_CHARS if c in content]
if found:
    print(f"Fixed {len(found)} unicode character type(s): {found}")
    print("File saved. You can now run module_05_physics_engine.py")
else:
    print("No problematic unicode characters found.")
    print("Spot C is already clean — your crash is from Spots A or B only.")
    print("Make sure both open() calls have encoding='utf-8' added.")