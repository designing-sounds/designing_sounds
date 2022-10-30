#!/bin/bash

pyinstaller -y --clean Sounds.spec
pushd dist
hdiutil create ./sounds.dmg -srcfolder sounds.app -ov
popd
cp ./dist/sounds.dmg .