# -*- mode: python ; coding: utf-8 -*-
import shutil
import re
import socket

# Get the hostname of the PC
hostname = socket.gethostname()



if hostname =="DELIA":#Laptop
    a = Analysis(
        ['FaceTracker.py'],
        pathex=[],
        binaries=[],
        datas=[('./.venv/lib/site-packages/llama_cpp', '.'), ('./.venv/lib/site-packages/llama_cpp/lib', '.'), ('./.venv/lib/site-packages/llama_cpp/lib/llama.dll', '.'), ('./config.txt', '.'), ('C:\\Users\\evelasquez\\PycharmProjects\\FacialControlHCI\\.venv\\lib\\site-packages\\mediapipe', 'mediapipe/')],
        hiddenimports=[],
        hookspath=['./hooks hook-llama_cpp.py'],
        hooksconfig={},
        runtime_hooks=[],
        excludes=[],
        noarchive=False,
        optimize=0,
    )
else:
    print(f"The name of the PC is: {hostname}")
pyz = PYZ(a.pure)

def changeVersion():
    # Open the file for reading
    with open("./config.txt", "r") as file:
        lines = file.readlines()  # Read all lines into a list
        first_line = lines[0].strip()

    # Split the line by periods
    split_line = first_line.split(".")

    # Grab the string after the last period
    last_part = split_line[-1]

    # Modify the string (for example, converting to uppercase)
    modified_last_part = str(int(last_part)+1)  # You can modify it however you'd like
    # Modify the first line by replacing the last part with the modified version
    modified_line = ".".join(split_line[:-1]) + "." + modified_last_part
    numbers = re.findall(r'\d+', split_line[-3])
    # Join the numbers into a single string (if you want them as a continuous string of digits)
    numbers_str = ''.join(numbers)

    totalVersion=numbers_str+"."+split_line[-2]+'.'+modified_last_part
    lines[0]=modified_line+'\n'
    # Now write the modified content back to the file
    with open("./config.txt", "w") as file:
        # Replace the first line with the modified one
        file.writelines(lines)
    return totalVersion


theNewVersion=changeVersion()
print(f'Changing version to: {theNewVersion}')
theName=f'FacialControlHMI_v{theNewVersion}'
print(f'Copying config.txt to dist Folder')
shutil.copy('./config.txt', './dist/')


exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=theName,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
