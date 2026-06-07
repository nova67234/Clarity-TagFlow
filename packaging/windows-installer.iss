; Inno Setup script for Clarity TagFlow (Windows).
;
; Produces ClarityTagFlow-Setup.exe: a per-user installer (no admin/UAC) that
; copies the app + its runtime DLLs (DirectML.dll, the VC++ runtime — both
; load-time imports that must sit next to the exe) into a private install folder
; and creates Start Menu / Desktop shortcuts. Users only ever see/click the app;
; the DLLs are tucked away in the install directory.
;
; Compiled in CI by ISCC with /DAppVer=<version> /DRepo=<workspace> (see
; .github/workflows/release-windows.yml). The staged payload lives in
; <Repo>\ClarityTagFlow-windows\ (exe + DLLs).

#ifndef AppVer
  #define AppVer "0.0.0"
#endif
#ifndef Repo
  #define Repo "."
#endif

[Setup]
; A fixed AppId so upgrades replace the prior install (don't change it).
AppId={{A1B2C3D4-E5F6-47A8-9B0C-1D2E3F4A5B6C}
AppName=Clarity TagFlow
AppVersion={#AppVer}
AppPublisher=nova67234
DefaultDirName={localappdata}\Programs\Clarity TagFlow
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
OutputDir={#Repo}
OutputBaseFilename=ClarityTagFlow-Setup
SetupIconFile={#Repo}\icons\app-icon.ico
UninstallDisplayIcon={app}\Clarity_TagFlow.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Files]
; Everything cargo staged next to the exe (exe + DirectML.dll + VC++ runtime).
Source: "{#Repo}\ClarityTagFlow-windows\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\Clarity TagFlow"; Filename: "{app}\Clarity_TagFlow.exe"
Name: "{autodesktop}\Clarity TagFlow"; Filename: "{app}\Clarity_TagFlow.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\Clarity_TagFlow.exe"; Description: "Launch Clarity TagFlow"; Flags: nowait postinstall skipifsilent
