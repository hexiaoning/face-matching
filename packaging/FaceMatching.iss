#define MyAppName "Face Matching"
#define MyAppVersion "3.2.0"
#define MyAppPublisher "Face Matching"
#define MyAppExeName "FaceMatching.exe"
#define SourceDir GetEnv("FACE_MATCHING_INSTALLER_SOURCE")
#define InstallerOutput GetEnv("FACE_MATCHING_INSTALLER_OUTPUT")

[Setup]
AppId={{B28F19BC-C66E-4D92-9F61-92E5C5FE94BB}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Face Matching
DefaultGroupName=Face Matching
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#InstallerOutput}
OutputBaseFilename=FaceMatching-v{#MyAppVersion}-Setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
InfoBeforeFile=installer-info.txt
VersionInfoVersion={#MyAppVersion}
VersionInfoDescription=Face Matching offline installer
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Face Matching"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Face Matching"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 Face Matching"; Flags: nowait postinstall skipifsilent
