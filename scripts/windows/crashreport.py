"""One-shot crash report for ftllm on Windows: minidump frames -> function names.

    py -3 scripts/windows/crashreport.py <dump> [<fastllm_tools.map>]

Without a map it prints exception, registers and stack candidates; with the map
sitting next to the packaged dll (build_vs-*/tools/ftllm) every
fastllm_tools.dll+0x... candidate is resolved to a symbol plus offset.
"""
import bisect
import re
import struct
import sys

THREAD_LIST = 3
MODULE_LIST = 4
MEMORY_LIST = 5
EXCEPTION_STREAM = 6
MEMORY64_LIST = 9

SYMBOL = re.compile(r"^\s+([0-9a-fA-F]{4}):([0-9a-fA-F]{8})\s+(\S+)\s+([0-9a-fA-F]{16})\s+(\S*)\s*(.*)$")
PREFERRED = re.compile(r"Preferred load address is ([0-9a-fA-F]+)")


class Dump(object):
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        signature, _, stream_count, directory = struct.unpack_from("<4sIII", self.read(0, 32), 0)
        if signature != b"MDMP":
            raise ValueError("not a minidump: %s" % path)
        self.streams = {}
        for index in range(stream_count):
            kind, size, rva = struct.unpack_from("<III", self.read(directory + index * 12, 12), 0)
            self.streams[kind] = (size, rva)
        self.modules = self._modules()
        self.module_bases = [module["base"] for module in self.modules]
        self.ranges = self._memory_ranges()

    def read(self, offset, count):
        self.f.seek(offset)
        return self.f.read(count)

    def _modules(self):
        if MODULE_LIST not in self.streams:
            return []
        _, rva = self.streams[MODULE_LIST]
        count = struct.unpack_from("<I", self.read(rva, 4), 0)[0]
        modules = []
        for index in range(count):
            entry = self.read(rva + 4 + index * 108, 108)
            base, image_size = struct.unpack_from("<QI", entry, 0)
            name_rva = struct.unpack_from("<I", entry, 20)[0]
            modules.append({"base": base, "size": image_size, "name": self._string(name_rva)})
        modules.sort(key=lambda item: item["base"])
        return modules

    def _string(self, rva):
        length = struct.unpack_from("<I", self.read(rva, 4), 0)[0]
        return self.read(rva + 4, length).decode("utf-16-le", "replace")

    def _memory_ranges(self):
        ranges = []
        if MEMORY_LIST in self.streams:
            _, rva = self.streams[MEMORY_LIST]
            count = struct.unpack_from("<I", self.read(rva, 4), 0)[0]
            for index in range(count):
                start, length, data_rva = struct.unpack_from("<QII", self.read(rva + 4 + index * 16, 16), 0)
                ranges.append((start, length, data_rva))
        if MEMORY64_LIST in self.streams:
            _, rva = self.streams[MEMORY64_LIST]
            count, data_rva = struct.unpack_from("<QQ", self.read(rva, 16), 0)
            cursor = data_rva
            for index in range(count):
                start, length = struct.unpack_from("<QQ", self.read(rva + 16 + index * 16, 16), 0)
                ranges.append((start, length, cursor))
                cursor += length
        ranges.sort()
        return ranges

    def threads(self):
        if THREAD_LIST not in self.streams:
            return []
        _, rva = self.streams[THREAD_LIST]
        count = struct.unpack_from("<I", self.read(rva, 4), 0)[0]
        result = []
        for index in range(count):
            entry = self.read(rva + 4 + index * 48, 48)
            result.append({"id": struct.unpack_from("<I", entry, 0)[0],
                           "stack_start": struct.unpack_from("<Q", entry, 24)[0],
                           "stack_size": struct.unpack_from("<I", entry, 32)[0],
                           "stack_rva": struct.unpack_from("<I", entry, 36)[0]})
        return result

    def thread_context(self, thread_id):
        if EXCEPTION_STREAM in self.streams:
            _, rva = self.streams[EXCEPTION_STREAM]
            data = self.read(rva, 168)
            if struct.unpack_from("<I", data, 0)[0] == thread_id:
                size, context_rva = struct.unpack_from("<II", data, 160)
                return self.read(context_rva, size)
        if THREAD_LIST in self.streams:
            _, rva = self.streams[THREAD_LIST]
            count = struct.unpack_from("<I", self.read(rva, 4), 0)[0]
            for index in range(count):
                entry = self.read(rva + 4 + index * 48, 48)
                if struct.unpack_from("<I", entry, 0)[0] == thread_id:
                    size, context_rva = struct.unpack_from("<II", entry, 40)
                    return self.read(context_rva, size)
        return None

    def exception(self):
        if EXCEPTION_STREAM not in self.streams:
            return None
        _, rva = self.streams[EXCEPTION_STREAM]
        data = self.read(rva, 168)
        thread_id = struct.unpack_from("<I", data, 0)[0]
        code, _, _, address = struct.unpack_from("<IIQQ", data, 8)
        count = struct.unpack_from("<I", data, 32)[0]
        params = list(struct.unpack_from("<15Q", data, 40))[:min(count, 15)]
        return {"thread": thread_id, "code": code, "address": address, "params": params}

    def describe(self, address):
        index = bisect.bisect_right(self.module_bases, address) - 1
        if index < 0:
            return "0x%x" % address
        module = self.modules[index]
        if address >= module["base"] + module["size"]:
            return "0x%x" % address
        return "%s+0x%x" % (module["name"].rsplit("\\", 1)[-1], address - module["base"])

    def read_stack(self, thread):
        address, length, rva = thread["stack_start"], thread["stack_size"], thread["stack_rva"]
        for start, size, data_rva in self.ranges:
            if start <= address and address + length <= start + size:
                return self.read(data_rva + (address - start), min(length, 256 * 1024))
        return self.read(rva, min(length, 256 * 1024))


def load_map(path):
    preferred = 0
    symbols = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if preferred == 0:
                found = PREFERRED.search(line)
                if found:
                    preferred = int(found.group(1), 16)
                    continue
            match = SYMBOL.match(line)
            if not match:
                continue
            address = int(match.group(4), 16)
            if address:
                symbols.append((address, match.group(3), match.group(6).strip()))
    symbols.sort(key=lambda item: item[0])
    return preferred, symbols


def make_resolver(path):
    preferred, symbols = load_map(path)
    if not symbols:
        raise SystemExit("no symbols parsed from %s" % path)
    addresses = [item[0] for item in symbols]

    def resolve(rva):
        target = preferred + rva
        index = bisect.bisect_right(addresses, target) - 1
        if index < 0:
            return None
        address, name, obj = symbols[index]
        following = addresses[index + 1] if index + 1 < len(addresses) else address
        return "%s + 0x%x  (%s, span 0x%x)" % (name, target - address, obj, following - address)

    print("map=%s preferred=0x%x symbols=%d" % (path, preferred, len(symbols)))
    return resolve


def scan(block, start, dump):
    hits = []
    for offset in range(start - start % 8, len(block) - 8, 8):
        value = struct.unpack_from("<Q", block, offset)[0]
        text = dump.describe(value)
        if "+0x" in text and text not in hits:
            hits.append(text)
    return hits


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    dump = Dump(argv[1])
    resolve = make_resolver(argv[2]) if len(argv) > 2 else None
    threads = dump.threads()
    print("dump=%s modules=%d threads=%d" % (argv[1], len(dump.modules), len(threads)))
    problem = dump.exception()
    thread = None
    if problem is not None:
        print("exception: code=0x%08x address=%s thread=%d" %
              (problem["code"], dump.describe(problem["address"]), problem["thread"]))
        if problem["params"]:
            print("params: " + ", ".join("0x%x" % value for value in problem["params"]))
        thread = next((item for item in threads if item["id"] == problem["thread"]), None)
    else:
        print("no exception stream, scanning first thread")
        thread = threads[0] if threads else None
    if thread is None:
        return 1
    context = dump.thread_context(thread["id"])
    if context is None or len(context) < 0x100:
        print("no thread context")
        return 1
    stack_pointer = struct.unpack_from("<Q", context, 0x98)[0]
    instruction_pointer = struct.unpack_from("<Q", context, 0xf8)[0]
    print("rip=%s  rsp=0x%x" % (dump.describe(instruction_pointer), stack_pointer))
    block = dump.read_stack(thread)
    print("stack candidates:")
    for text in scan(block, max(0, stack_pointer - thread["stack_start"]), dump)[:80]:
        name, _, offset = text.rpartition("+0x")
        if resolve is not None and name.lower() == "fastllm_tools.dll":
            resolved = resolve(int(offset, 16))
            print("  %-28s %s" % (text, resolved if resolved else "unresolved"))
        else:
            print("  %s" % text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
