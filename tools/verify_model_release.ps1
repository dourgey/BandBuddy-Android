param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"

$specPath = Join-Path $ProjectRoot "app\src\main\java\com\lonelyme\bandbuddy\model\ModelSpec.kt"
$manifestPath = Join-Path $ProjectRoot "modelscope\bandbuddy-model.json"
$assetPath = Join-Path $ProjectRoot "app\src\main\assets\models\htdemucs_6s.core.tflite"

$spec = Get-Content -Raw -Encoding UTF8 -LiteralPath $specPath
$manifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifestPath | ConvertFrom-Json
$stable = $manifest.stable
$artifactPath = Join-Path (Split-Path -Parent $manifestPath) $stable.file

function Read-KotlinStringConstant([string]$Name) {
    $match = [regex]::Match($spec, "const val $Name\s*=\s*`"([^`"]+)`"")
    if (-not $match.Success) {
        throw "ModelSpec.$Name was not found"
    }
    return $match.Groups[1].Value
}

function Read-KotlinLongConstant([string]$Name) {
    $match = [regex]::Match($spec, "const val $Name\s*=\s*([0-9_]+)L")
    if (-not $match.Success) {
        throw "ModelSpec.$Name was not found"
    }
    return [long]($match.Groups[1].Value -replace "_", "")
}

$expected = @{
    Version = Read-KotlinStringConstant "VERSION"
    Revision = Read-KotlinStringConstant "REVISION"
    File = Read-KotlinStringConstant "FILE_NAME"
    Bytes = Read-KotlinLongConstant "BYTES"
    Sha256 = (Read-KotlinStringConstant "SHA256").ToLowerInvariant()
    CacheToken = Read-KotlinStringConstant "CACHE_TOKEN"
}

if ($stable.version -ne $expected.Version) {
    throw "Version mismatch: manifest=$($stable.version), app=$($expected.Version)"
}
if ($stable.revision -ne $expected.Revision) {
    throw "Revision mismatch: manifest=$($stable.revision), app=$($expected.Revision)"
}
if ($stable.file -ne $expected.File) {
    throw "Filename mismatch: manifest=$($stable.file), app=$($expected.File)"
}
if ([long]$stable.bytes -ne $expected.Bytes) {
    throw "Byte-count mismatch: manifest=$($stable.bytes), app=$($expected.Bytes)"
}
if ($stable.sha256.ToLowerInvariant() -ne $expected.Sha256) {
    throw "SHA-256 mismatch between manifest and app"
}
if ($stable.cache_token -ne $expected.CacheToken) {
    throw "Cache token mismatch between manifest and app"
}
if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
    throw "Release artifact is missing: $artifactPath"
}

$artifact = Get-Item -LiteralPath $artifactPath
if ($artifact.Length -ne $expected.Bytes) {
    throw "Artifact byte-count mismatch: file=$($artifact.Length), expected=$($expected.Bytes)"
}
$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifactPath).Hash.ToLowerInvariant()
if ($actualHash -ne $expected.Sha256) {
    throw "Artifact SHA-256 mismatch: file=$actualHash, expected=$($expected.Sha256)"
}
if (Test-Path -LiteralPath $assetPath) {
    throw "The model is still under app/src/main/assets and would be packaged in the APK"
}

Write-Output "Model release verified: $($manifest.repository)@$($expected.Revision)"
Write-Output "Artifact: $($expected.File) ($($expected.Bytes) bytes)"
Write-Output "SHA-256: $actualHash"

