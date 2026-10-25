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

[Code]
// Link (Settings -> Link) can keep a copy of the app hosting in the
// background ("Clarity_TagFlow.exe --link-background"), which holds the
// program open, and can start it at login (a Run entry).

// Stop a running background host.
procedure StopLinkBackgroundHost();
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    '-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "' +
    'Get-CimInstance Win32_Process | Where-Object { $_.Name -eq ''Clarity_TagFlow.exe'' -and ' +
    '$_.CommandLine -like ''*--link-background*'' } | Invoke-CimMethod -MethodName Terminate | Out-Null; ' +
    'Start-Sleep -Seconds 2"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

// Before the files are replaced: leave the marker that tells a copy closed
// for the update not to start a background host (the app clears it when it
// next starts; an app finishing another PC's job without its window sees it
// and exits too), then stop one that's running. Runs before Setup looks for
// programs using its files (CloseApplications).
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  LinkDir: String;
begin
  LinkDir := ExpandConstant('{userappdata}\Clarity TagFlow\tools\link');
  if ForceDirectories(LinkDir) then
    SaveStringToFile(LinkDir + '\installing', 'update', False);
  StopLinkBackgroundHost();
  Result := '';
end;

// Uninstalling (once confirmed, before the files go): no background host
// left running, and no login entry left pointing at the removed program.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    StopLinkBackgroundHost();
    RegDeleteValue(HKEY_CURRENT_USER, 'Software\Microsoft\Windows\CurrentVersion\Run', 'Clarity TagFlow Link');
  end;
end;
