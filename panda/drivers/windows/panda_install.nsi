!define J2534_Reg_Path "Software\PassThruSupport.04.04\comma.ai - panda"
!define Install_Name "panda J2534 driver"

;NOTE! The panda software requires a VC runtime to be installed in order to work.
;This installer must be bundled with the appropriate runtime installer, and have
;the installation registry key set so the installer can tell if the runtime is
;already installed. Copy vscruntimeinfo.nsh.sample to vscruntimeinfo.nsh and edit
;it for your version of Visual Studio.
!include "redist\vscruntimeinfo.nsh"

;--------------------------------
;Include Modern UI
!include "MUI2.nsh"
!include "x64.nsh"

!define MUI_ICON "panda.ico"
;NSIS is ignoring the unicon unless it is the same as the normal icon
;!define MUI_UNICON "panda_remove.ico"

;Properly display all languages (Installer will not work on Windows 95, 98 or ME!)
Unicode true

# Set the installer display name
Name "${Install_Name}"

# set the name of the installer
Outfile "${Install_Name} install.exe"

; The default installation directory
InstallDir $PROGRAMFILES\comma.ai\panda

; Request application privileges for UAC
RequestExecutionLevel admin

; Registry key to check for directory (so if you install again, it will
; overwrite the old one automatically)
InstallDirRegKey HKLM "SOFTWARE\${Install_Name}" "Install_Dir"

;--------------------------------
; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English" ;first language is the default language

; -------------------------------------------------------------------------------------------------
; Additional info (will appear in the "details" tab of the properties window for the installer)

VIAddVersionKey /LANG=${LANG_ENGLISH} "ProductName"      "panda OBD-II adapter"
VIAddVersionKey /LANG=${LANG_ENGLISH} "Comments"         ""
VIAddVersionKey /LANG=${LANG_ENGLISH} "CompanyName"      "comma.ai"
VIAddVersionKey /LANG=${LANG_ENGLISH} "LegalTrademarks"  "Application released under the MIT license"
;VIAddVersionKey /LANG=${LANG_ENGLISH} "LegalCopyright"   "© ${PRODUCT_NAME} Team"
;VIAddVersionKey /LANG=${LANG_ENGLISH} "FileDescription"  "Jessy Exum"
;VIAddVersionKey /LANG=${LANG_ENGLISH} "FileVersion"      "${PRODUCT_VERSION}"
VIProductVersion "1.0.0.0"

;--------------------------------
; Install Sections
Section "prerequisites"

  SectionIn RO

  SetOutPath "$INSTDIR"

  File "panda.ico"

  ;If the visual studio version this project is compiled with changes, this section
  ;must be revisited. The registry key must be changed, and the VS redistributable
  ;binary must be updated to the VS version used.
  ClearErrors
  ReadRegStr $0 HKCR ${VCRuntimeRegKey} "Version"
  ${If} ${Errors}
    DetailPrint "Installing Visual Studio C Runtime..."
    File "${VCRuntimeSetupPath}\${VCRuntimeSetupFile}"
    ExecWait '"$INSTDIR\${VCRuntimeSetupFile}" /passive /norestart'
  ${Else}
    DetailPrint "Visual Studio C Runtime already installed."
  ${EndIf}

  ;Remove the now unnecessary runtime installer.
  Delete "$INSTDIR\${VCRuntimeSetupFile}"

  ;Do the rest of the install
  ; SetOutPath "$INSTDIR\driver"

  ; The inf file works for both 32 and 64 bit.
  ; File "Debug_x86\panda Driver Package\panda.inf"
  ; File "Debug_x86\panda Driver Package\panda.cat"
  ; ${DisableX64FSRedirection}
  ; nsExec::ExecToLog '"$SYSDIR\PnPutil.exe" /a "$INSTDIR\driver\panda.inf"'
  ; ${EnableX64FSRedirection}

  ; Write the installation path into the registry
  WriteRegStr HKLM "SOFTWARE\${Install_Name}" "Install_Dir" "$INSTDIR"

  ; Write the uninstall keys for Windows
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}" "DisplayVersion" ""
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}" "DisplayIcon" '"$INSTDIR\panda.ico",0'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}" "DisplayName" "${Install_Name}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}" "Publisher" "comma.ai"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}" "URLInfoAbout" "https://github.com/commaai/panda/"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}" "NoRepair" 1

  SetOutPath $INSTDIR
  WriteUninstaller "uninstall.exe"

SectionEnd

Section "J2534 Driver"

  SetOutPath $INSTDIR

  File Release_x86\pandaJ2534_0404_32.dll

  SetRegView 32
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "CAN"               00000001
  WriteRegStr    HKLM "${J2534_Reg_Path}" "FunctionLibrary"   "$INSTDIR\pandaJ2534_0404_32.dll"
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "ISO15765"          00000001
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "J1850VPW"          00000000
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "SCI_A_ENGINE"      00000000
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "SCI_A_TRANS"       00000000
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "SCI_B_ENGINE"      00000000
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "SCI_B_TRANS"       00000000
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "J1850PWM"          00000000
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "ISO9141"           00000001
  WriteRegDWORD  HKLM "${J2534_Reg_Path}" "ISO14230"          00000001
  WriteRegStr    HKLM "${J2534_Reg_Path}" "Name"              "panda"
  WriteRegStr    HKLM "${J2534_Reg_Path}" "Vendor"            "comma.ai"
  WriteRegStr    HKLM "${J2534_Reg_Path}" "ConfigApplication" ""
  DetailPrint "Registered J2534 Driver"

SectionEnd

Section /o "Development lib/header"

  SetOutPath $SYSDIR

  File Release_x86\panda.dll

  ${If} ${RunningX64}
    ${DisableX64FSRedirection}
      ;Note that the x64 VS redistributable is not installed to prevent bloat.
      ;If you are the rare person who uses the 64 bit raw panda driver, please
      ;install the correct x64 VS runtime manually.
      File Release_x64\panda.dll
    ${EnableX64FSRedirection}
  ${EndIf}

  SetOutPath "$INSTDIR\devel"
  File panda_shared\panda.h

  SetOutPath "$INSTDIR\devel\x86"
  File Release_x86\panda.lib

  SetOutPath "$INSTDIR\devel\x64"
  File Release_x64\panda.lib

SectionEnd

;--------------------------------
; Uninstaller
Section "Uninstall"

  ; Removing the inf file for winusb is not easy to do.
  ; The best solution I can find is parsing the output
  ; of the pnputil.exe /e command to find the oem#.inf
  ; file that lists comma.ai as the provider. Not sure
  ; if Microsoft wants these inf files to be removed.
  ; Consider https://blog.sverrirs.com/2015/12/creating-windows-installer-and.html
  ; These lines just remove the inf backups.
  ; Delete "$INSTDIR\driver\panda.inf"
  ; Delete "$INSTDIR\driver\panda.cat"
  ; RMDir "$INSTDIR\driver"

  ; Remove WinUSB driver library
  Delete $SYSDIR\panda.dll
  ${If} ${RunningX64}
    ${DisableX64FSRedirection}
    Delete $SYSDIR\panda.dll
    ${EnableX64FSRedirection}
  ${EndIf}

  ; Remove devel files
  Delete "$INSTDIR\devel\x86\panda.lib"
  RMDir "$INSTDIR\devel\x86"
  Delete "$INSTDIR\devel\x64\panda.lib"
  RMDir "$INSTDIR\devel\x64"
  Delete "$INSTDIR\devel\panda.h"
  RMDir "$INSTDIR\devel"

  ; Remove registry keys
  DeleteRegKey HKLM "${J2534_Reg_Path}"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${Install_Name}"
  DeleteRegKey HKLM "SOFTWARE\${Install_Name}"

  ; Remove files and uninstaller
  Delete "$INSTDIR\uninstall.exe"
  Delete "$INSTDIR\pandaJ2534_0404_32.dll"
  Delete "$INSTDIR\panda.ico"

  ; Remove directories used
  RMDir "$INSTDIR"
  RMDir "$PROGRAMFILES\comma.ai"

SectionEnd
