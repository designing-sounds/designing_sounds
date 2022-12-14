# -*- mode: python ; coding: utf-8 -*-

from kivymd import hooks_path as kivymd_hooks_path

block_cipher = None


a = Analysis(
    ['../../main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[kivymd_hooks_path],
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

a.datas +=[('src/wave_view/wave.kv', '../../src/wave_view/wave.kv', 'DATA'), ('src/wave_view/power.kv', '../../src/wave_view/power.kv', 'DATA'), ('media/black.png', '../../media/black.png', 'media'), ('src/wave_view/power.kv', '../../src/wave_view/power.kv', 'DATA'), ('media/icon.png', '../../media/icon.png', 'media')]

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
    exe, Tree('../../src'),
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=None,
    upx=True,
    upx_exclude=[],
    name='sounds',
)
app = BUNDLE(
    coll,
    name='Sounds.app',
    icon='../../media/icon.icns',
    bundle_identifier='com.tommywoodley.sounds',
)
