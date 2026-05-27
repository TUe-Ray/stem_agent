#!/usr/bin/env python3
"""Replace patcher instructions in loop.py with enhanced retry logic."""
import sys

with open('src/stem_agent/evolution/loop.py', 'r') as f:
    content = f.read()

# Old string as it appears in the file (\\n = literal backslash-n in file)
old_str = (
    "6. Write your patch using write_patch({'patch_text': '...'})\\n"
    "   Use ONLY line numbers from files you personally read.\\n"
    "7. Verify with apply_patch_dry_run({'patch_text': '...'})\\n"
    "8. If it fails: fix line numbers/context and retry (max 2 times)"
)

# New string with enhanced retry logic and patch format self-check
new_str = (
    "6. Write your patch using write_patch({'patch_text': '...'})\\n"
    "   Use ONLY line numbers from files you personally read.\\n"
    "\\n"
    "6a. *** PATCH FORMAT SELF-CHECK ***\\n"
    "   Before verifying, review your diff and ensure:\\n"
    '   - NO placeholder comments like \\"...\\", \\"# Existing code...\\", \\"# ... some code ...\\"\\n'
    "   - Every context line in the diff is REAL code from the source files you read\\n"
    "   - Every hunk header (@@ ... @@) has correct line numbers\\n"
    "   If ANY placeholder or fake code is found, rewrite the patch immediately\\n"
    "   before proceeding to verification. Do NOT verify a patch with placeholders.\\n"
    "\\n"
    "7. Verify with apply_patch_dry_run({'patch_text': '...'})\\n"
    "8. *** RETRY WITH ERROR-DRIVEN CORRECTION *** (max 2 retries):\\n"
    "   If verification fails, do NOT blindly tweak numbers. Instead:\\n"
    "   a. READ the error output carefully — it contains precise information:\\n"
    '      e.g. \\"hunk #1 FAILED at line 940\\" or \\"patch does not apply\\"\\n'
    "   b. For each FAILED hunk, extract the hunk number and the reported line number.\\n"
    "   c. Use read_file() to re-read the source file AROUND that reported line:\\n"
    '      e.g. if error says \\"at line 940\\", read lines 935-950 to see actual content.\\n'
    "   d. Compare the actual file content with your patch context lines.\\n"
    "      The context lines MUST match the file EXACTLY — whitespace, indentation, everything.\\n"
    "   e. Rewrite the patch with CORRECTED context lines and line numbers.\\n"
    "   f. Go back to step 7 and re-verify.\\n"
    "   \\n"
    "   The error message is your guide — it tells you exactly which hunk failed\\n"
    "   and at what line. Use read_file to see what the file actually looks like."
)

count = content.count(old_str)
if count != 1:
    print(f"ERROR: old_str found {count} times, expected 1")
    sys.exit(1)

content = content.replace(old_str, new_str)

with open('src/stem_agent/evolution/loop.py', 'w') as f:
    f.write(content)

print("Replacement successful!")

# Verify
with open('src/stem_agent/evolution/loop.py', 'r') as f:
    lines = f.readlines()
line = lines[2019]  # 0-indexed
checks = [
    "6a. *** PATCH FORMAT SELF-CHECK ***",
    "ERROR-DRIVEN CORRECTION",
    "hunk #1 FAILED at line 940",
    "read_file() to re-read the source file AROUND",
]
for check in checks:
    if check in line:
        print(f"  ✓ '{check}' found")
    else:
        print(f"  ✗ '{check}' NOT found")
