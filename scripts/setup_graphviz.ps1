$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$artifactRoot = Join-Path $projectRoot '.artifacts'
$archivePath = Join-Path $artifactRoot 'graphviz.zip'
$targetPath = Join-Path $artifactRoot 'graphviz'
$url = 'https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/16.1.0/windows_10_cmake_Release_Graphviz-16.1.0-win64.zip'
$expectedHash = '733e49626c492242eb8dca30ea627b6ead20710e207998c7933b4909d92d6abc'
New-Item -ItemType Directory -Force $artifactRoot | Out-Null
Invoke-WebRequest $url -OutFile $archivePath
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'Graphviz archive checksum mismatch'
}
Expand-Archive -LiteralPath $archivePath -DestinationPath $targetPath -Force
& (Join-Path $targetPath 'Graphviz-16.1.0-win64/bin/dot.exe') -V
