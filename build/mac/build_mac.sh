#!/bin/bash

pyinstaller -y --clean Sounds.spec
security find-identity -p basic -v
sudo codesign -s - --force --all-architectures --timestamp --deep dist/sounds.app
pushd dist
hdiutil create ./sounds.dmg -srcfolder sounds.app -ov
popd
cp ./dist/sounds.dmg .
