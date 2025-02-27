# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['FaceTracker.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\velas\\PycharmProjects\\FacialControlHCI\\venvFacialControl\\Lib\\site-packages\\llama_cpp', '.'), ('C:\\Users\\velas\\PycharmProjects\\FacialControlHCI\\venvFacialControl\\Lib\\site-packages\\llama_cpp\\lib', '.'), ('C:\\Users\\velas\\PycharmProjects\\FacialControlHCI\\venvFacialControl\\Lib\\site-packages\\llama_cpp\\lib\\llama.dll', '.'), ('./config.txt', '.'), ('C:\\Users\\velas\\PycharmProjects\\ballTracker\\venv\\lib\\site-packages\\mediapipe', 'mediapipe/')],
    hiddenimports=[],
    hookspath=['./hooks hook-llama_cpp.py'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FacialControlHMI',
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
