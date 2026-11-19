from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading


_DIALOG_LOCK = threading.Lock()
_DIALOG_TIMEOUT = 300
_TRACE_FILTER = "FastLLM Trace (*.ftrace)|*.ftrace|All files (*.*)|*.*"


def _result_path(value: str):
    value = value.rstrip("\r\n")
    return {"path": value} if value else {"cancelled": True}


def _decode_output(data: bytes) -> str:
    if not data:
        return ""
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _run(command, timeout):
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=False, creationflags=flags)


def _windows_pick(mode: str, timeout: int):
    executable = shutil.which("powershell.exe") or shutil.which("powershell") or \
        shutil.which("pwsh.exe") or shutil.which("pwsh")
    if not executable:
        return {"error": "dialog_unavailable", "hint": "请手动输入路径"}
    if mode == "dir":
        dialog = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
            "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$d.Description='FastLLM Trace';"
            "if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){"
            "[Console]::Out.Write($d.SelectedPath)}"
        )
    else:
        dialog = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
            "$d=New-Object System.Windows.Forms.OpenFileDialog;"
            "$d.Title='FastLLM Trace';"
            f"$d.Filter='{_TRACE_FILTER}';"
            "if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){"
            "[Console]::Out.Write($d.FileName)}"
        )
    completed = _run([executable, "-NoProfile", "-NonInteractive", "-STA",
                      "-ExecutionPolicy", "Bypass", "-Command", dialog], timeout)
    output = _decode_output(completed.stdout).rstrip("\r\n")
    if completed.returncode != 0 and not output:
        error = _decode_output(completed.stderr).strip()
        if "cancel" in error.lower():
            return {"cancelled": True}
        return {"error": "dialog_failed", "message": error or "PowerShell failed"}
    return _result_path(output)


def _macos_pick(mode: str, timeout: int):
    if mode == "dir":
        script = 'POSIX path of (choose folder with prompt "FastLLM Trace")'
    else:
        script = ('POSIX path of (choose file with prompt "FastLLM Trace" '
                  'of type {"ftrace"})')
    try:
        completed = _run(["osascript", "-e", script], timeout)
    except FileNotFoundError:
        return {"error": "dialog_unavailable", "hint": "请手动输入路径"}
    output = _decode_output(completed.stdout).rstrip("\r\n")
    if completed.returncode == 0:
        return _result_path(output)
    error = _decode_output(completed.stderr).strip()
    if "User canceled" in error or "-128" in error:
        return {"cancelled": True}
    return {"error": "dialog_failed", "message": error or "osascript failed"}


def _linux_pick(mode: str, timeout: int):
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return {"error": "dialog_unavailable", "hint": "请手动输入路径"}
    zenity = shutil.which("zenity")
    if zenity:
        command = [zenity, "--file-selection", "--title=FastLLM Trace"]
        if mode == "dir":
            command.append("--directory")
        else:
            command.append("--file-filter=FastLLM Trace | *.ftrace")
        completed = _run(command, timeout)
        output = _decode_output(completed.stdout).rstrip("\r\n")
        if completed.returncode == 0:
            return _result_path(output)
        if completed.returncode == 1 and not output:
            return {"cancelled": True}
        error = _decode_output(completed.stderr).strip()
        return {"error": "dialog_failed", "message": error or "zenity failed"}
    kdialog = shutil.which("kdialog")
    if kdialog:
        if mode == "dir":
            command = [kdialog, "--getexistingdirectory", "."]
        else:
            command = [kdialog, "--getopenfilename", ".",
                       "*.ftrace|FastLLM Trace"]
        completed = _run(command, timeout)
        output = _decode_output(completed.stdout).rstrip("\r\n")
        if completed.returncode == 0:
            return _result_path(output)
        if completed.returncode == 1 and not output:
            return {"cancelled": True}
        error = _decode_output(completed.stderr).strip()
        return {"error": "dialog_failed", "message": error or "kdialog failed"}
    return {"error": "dialog_unavailable", "hint": "请手动输入路径"}


def pick_path(mode: str = "file", timeout: int = _DIALOG_TIMEOUT):
    if mode not in ("file", "dir"):
        return {"error": "dialog_invalid_mode"}
    if not _DIALOG_LOCK.acquire(blocking=False):
        return {"error": "dialog_busy"}
    try:
        system = platform.system()
        if system == "Windows":
            return _windows_pick(mode, timeout)
        if system == "Darwin":
            return _macos_pick(mode, timeout)
        if system == "Linux":
            return _linux_pick(mode, timeout)
        return {"error": "dialog_unavailable", "hint": "请手动输入路径"}
    except subprocess.TimeoutExpired:
        return {"error": "dialog_timeout"}
    except FileNotFoundError:
        return {"error": "dialog_unavailable", "hint": "请手动输入路径"}
    except Exception as error:
        return {"error": "dialog_failed", "message": str(error)}
    finally:
        _DIALOG_LOCK.release()
