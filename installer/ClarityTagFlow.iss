; Inno Setup script for Clarity TagFlow.
; Build it with Inno Setup 6.3+ (https://jrsoftware.org/isdl.php):
;   - Open this file in the Inno Setup Compiler and click Build, or
;   - run:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\ClarityTagFlow.iss
; Output: installer\Output\ClarityTagFlow-Setup.exe
;
; Installs PER-USER into %LOCALAPPDATA%\Programs\Clarity TagFlow (no admin prompt).
; A per-user, writable location is intentional: AI models download into a
; "tools\" folder next to the exe, which would fail under Program Files.

#define MyAppName "Clarity TagFlow"
#define MyAppExeName "Clarity_TagFlow.exe"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Clarity"

[Setup]
; Unique ID for this app (keep stable across versions so upgrades replace cleanly).
AppId={{B1F4C2A0-9D3E-4E7A-8C21-6F5A2D9E3B14}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Per-user install — no administrator rights required, and the folder is writable.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
; Icon shown on the generated Setup.exe itself.
SetupIconFile=..\icons\app-icon.ico
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=ClarityTagFlow-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; The entire prepared dist\ folder (exe, runtime DLLs, VLC plugins, README).
Source: "..\dist\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
