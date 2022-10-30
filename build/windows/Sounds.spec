# -*- mode: python ; coding: utf-8 -*-

from kivy.deps import sdl2, glew

block_cipher = None


a = Analysis(
    ['../../main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

a.datas +=[('wave.kv', '/Users/tomwoodley/Desktop/designing_sounds/src/wave_view/wave.kv', 'DATA'),]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sounds',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe, Tree('/Users/tomwoodley/Desktop/designing_sounds/src'),
    a.binaries,
    a.zipfiles,
    a.datas,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    strip=None,
    upx=True,
    upx_exclude=[],
    name='sounds',
)
app = BUNDLE(
    coll,
    name='sounds.app',
    icon=None,
    bundle_identifier=None,
)
