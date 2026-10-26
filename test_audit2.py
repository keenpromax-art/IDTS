import sys
sys.path.insert(0, r'C:\IDTS')

# Print the audit module source code around __post_init__
with open('C:\IDTS\trader\core\audit.py', 'r') as f:
    content = f.read()

# Find the __post_init__ method
import re
match = re.search(r'def __post_init__\(self\):.*?        ', content, re.DOTALL)
if match:
    print(match.group()[:500])
"