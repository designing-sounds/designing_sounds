#!/bin/bash

pyinstaller -y --clean Sounds.spec
sudo codesign -s - --force --all-architectures --timestamp --deep dist/sounds.app
cp ./dist/sound.app .
#pushd dist
#hdiutil create ./sounds.dmg -srcfolder sounds.app -ov
#popd
#cp ./dist/sounds.dmg .
