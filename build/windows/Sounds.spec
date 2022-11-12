# -*- mode: python ; coding: utf-8 -*-

import os
os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'

from kivy_deps import sdl2, angle

block_cipher = None


a = Analysis(
    ['..\..\main.py'],
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

a.datas +=[('src\wave_view\wave.kv', '..\..\src\wave_view\wave.kv', 'DATA'), ('media\20221028_144310.jpg', '..\..\media\20221028_144310.jpg', 'media')]

exe = EXE(
    pyz, Tree('../../src'),
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    *[Tree(p) for p in (sdl2.dep_bins + angle.dep_bins)],
    [],
    name='Sounds',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
