"""Resolve fastllm_tools.dll+0x... crash frames against the linker map.

The build writes fastllm_tools.map next to the packaged dll (CMakeLists /MAP),
so a crash offset from a minidump stack can be turned into a function name:

    py -3 scripts/windows/mapresolve.py <map> 1bf835 8be2ee ...

Each rva is printed as the enclosing symbol plus the offset inside it.
"""
import bisect
import re
import sys

SYMBOL = re.compile(r"^\s+([0-9a-fA-F]{4}):([0-9a-fA-F]{8})\s+(\S+)\s+([0-9a-fA-F]{16})\s+(\S*)\s*(.*)$")
BASE = re.compile(r"Preferred load address is ([0-9a-fA-F]+)")


def load(path):
    preferred = 0
    symbols = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if preferred == 0:
                found = BASE.search(line)
                if found:
                    preferred = int(found.group(1), 16)
                    continue
            match = SYMBOL.match(line)
            if not match:
                continue
            address = int(match.group(4), 16)
            if address == 0:
                continue
            symbols.append((address, match.group(3), match.group(5), match.group(6).strip()))
    symbols.sort(key=lambda item: item[0])
    return preferred, symbols


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    path = argv[1]
    preferred, symbols = load(path)
    if not symbols:
        raise SystemExit("no symbols parsed from %s" % path)
    addresses = [item[0] for item in symbols]
    print("map=%s preferred=0x%x symbols=%d" % (path, preferred, len(symbols)))
    for raw in argv[2:]:
        rva = int(raw, 16)
        target = preferred + rva
        index = bisect.bisect_right(addresses, target) - 1
        if index < 0:
            print("0x%-8x -> below first symbol" % rva)
            continue
        address, name, kind, obj = symbols[index]
        next_address = addresses[index + 1] if index + 1 < len(addresses) else address
        print("0x%-8x -> %s + 0x%x  (%s, next=0x%x)" %
              (rva, name, target - address, obj, next_address - address))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
