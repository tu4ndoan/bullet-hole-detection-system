[Setup]
   AppName=BaoBiaTuDong
   AppVersion=1.0
   DefaultDirName={autopf}\BaoBiaTuDong
   DefaultGroupName=BaoBiaTuDong
   OutputDir=.\Output
   OutputBaseFilename=BaoBiaTuDong_CaiDat

   [Files]
   Source: "dist\main.exe"; DestDir: "{app}"; Flags: ignoreversion

   [Icons]
   Name: "{autoprograms}\BaoBiaTuDong"; Filename: "{app}\BaoBiaTuDong.exe"