#!/bin/bash

# CONSTANTS - ADD YOUR INFO
DEVELOPER_EMAIL="thwoodley@gmail.com"      # appleid email associate with developer account
APP="Sounds"                        # name of your app
SPEC='Sounds.spec'                         # location of your .spec file for PyInstaller
PKG_IDENTIFIER="com.tommywoodley.sounds.pkg" # identifier for the package
APP_VERSION=0.1                         # app version, written into the .pkg
TMP_PKG_PATH=dist/pkg.pkg              # where to build your unsigned package; NOTE: this directory must exist, or script will fail
SCRIPTS=./Scripts/        # Scripts path: must end in "Scripts/" and have scripts called, exactly, "preinstall" and "postinstall"
sudo chmod -R +x $SCRIPTS               # make sure scripts are executable, or whatever permissions you want
SIGNED_PKG=dist/"$APP".pkg       # where to put the signed package

### HASHES OF DEVELOPER IDS ###
# Use command (without quotes) 'security find-identity -p basic -v' in terminal
# the hash (HSHSD...HFJSJ) to the left of 'Developer ID Application...' is for DEVELOPER_ID_APPLICAITON
# the hash (DJJSH...HDHDH) to the left of 'Installer ID Installer...' is for DEVELOPER_ID_INSTALLER_HASH
DEVELOPER_ID_APPLICATION="7D93DB6B20D79A98F052D3625615271C1FF03DEB"
DEVELOPER_ID_INSTALLER_HASH="011A5B4D256D2B4A85AB318F7899334EF605C657"

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#* BELOW THIS LINE IS DANGER ZONE #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#                                                                                                  *
#                        YOU SHOULD NOT NEED TO EDIT ANYTHING BELOW THIS LINE                      *
#                                                                                                  *
#                                       PROCEED WITH CAUTION                                       *
#                                                                                                  *
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#* BELOW THIS LINE IS DANGER ZONE #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

# CONSTANTS - NO MODIFIATION NEEDED
APPFILE=./dist/"$APP".app           # location of the .app file after PyInstaller builds it
APP_NAME="$APP"                     # carrying different conventions
APP_PATH="$APPFILE"                 # carrying different conventions
PKG="$SIGNED_PKG"                   # carrying different conventions
# CONSTANTS - NO MODIFIATION NEEDED

# STEP 1: build the .app file with PyInstaller
printf "\n\nSTEP 1: build the .app file with PyInstaller\n"
pyinstaller --noconfirm --clean "$SPEC"

# STEP 2: codesign the .app file
printf "\n\nSTEP 2: codesign the .app file\n"
codesign --deep --force --timestamp --options runtime --entitlements ./entitlements.plist --sign "$DEVELOPER_ID_APPLICATION" "$APPFILE"

# (optional) verify the signature of the .app file
printf "\n\n(optional) verify the signature of the .app file\n"
codesign --verify --verbose "$APPFILE"
codesign -dvvv "$APPFILE"

# STEP 3: build the .pkg file
printf "\n\nSTEP 3: build the .pkg file\n"
pkgbuild --identifier "$PKG_IDENTIFIER" \
         --sign "$DEVELOPER_ID_INSTALLER_HASH" \
         --version "$APP_VERSION" \
         --root "$APP_PATH" \
         --scripts "$SCRIPTS" \
         --install-location /Applications/"$APP_NAME".app "$TMP_PKG_PATH"

# STEP 4: sign the .pkg file
printf "\n\nSTEP 4: sign the .pkg file\n"
productsign --sign "$DEVELOPER_ID_INSTALLER_HASH" "$TMP_PKG_PATH" "$SIGNED_PKG"

# STEP 5: upload to notary service
printf "\n\nSTEP 5: upload to notary service\n"
xcrun altool --notarize-app --primary-bundle-id "com.tommywoodley.sounds" --username $DEVELOPER_EMAIL --password outn-zgln-qpmi-awev --file "$PKG"

# to check the status:
# xcrun altool --username "$DEVELOPER_EMAIL" --password "@keychain:Developer-altool" --notarization-info <PUT_UUID_HERE>
echo "to check the status: "
echo "xcrun altool --username "$DEVELOPER_EMAIL" --password "@keychain:Developer-altool" --notarization-info <PUT_UUID_HERE_WITHOUT_BRACKETS>"

# if it fails, to view more info: xcrun altool --username "comething@domain.com" --password "@keychain:Developer-altool" --notarization-info "Your-Request-UUID"
echo "if status is invalid or otherwise fails: "
echo "xcrun altool --username "$DEVELOPER_EMAIL" --password "@keychain:Developer-altool" --notarization-info <PUT_UUID_HERE_WITHOUT_BRACKETS>"

# if it succeeds: xcrun stapler staple "$PKG"
echo "if status is approved or otherwise successful"
echo "xcrun stapler staple" "\"$PKG\""

# (optional) after stapling, use the below command to check the notary service succeeded, but you should not ned to
# checks notary: spctl --assess --verbose $APPFILE