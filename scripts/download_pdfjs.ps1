<#
Downloads PDF.js build files into `static/lib/`.
Run from project root in PowerShell (Windows PowerShell 5.1):
    .\scripts\download_pdfjs.ps1
If you have execution policy restrictions, run:
    powershell -ExecutionPolicy Bypass -File .\scripts\download_pdfjs.ps1
#>
$libDir = Join-Path $PSScriptRoot "..\static\lib" | Resolve-Path -Relative
if (-not (Test-Path $libDir)) { New-Item -ItemType Directory -Path $libDir | Out-Null }

$files = @(
    @{ urls = @('https://unpkg.com/pdfjs-dist@latest/build/pdf.js','https://mozilla.github.io/pdf.js/build/pdf.js','https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js'); name = 'pdf.js' },
    @{ urls = @('https://unpkg.com/pdfjs-dist@latest/build/pdf.worker.js','https://mozilla.github.io/pdf.js/build/pdf.worker.js','https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js'); name = 'pdf.worker.js' }
)

Write-Output "Downloading PDF.js files into: $libDir"
foreach ($f in $files) {
    $outPath = Join-Path $libDir $($f.name)
    $succeeded = $false
    foreach ($u in $f.urls) {
        try {
            Write-Output "Fetching $u -> $outPath"
            Invoke-WebRequest -Uri $u -OutFile $outPath -UseBasicParsing -ErrorAction Stop
            $succeeded = $true
            break
        }
        catch {
            Write-Warning ("Failed to download {0}: {1}" -f $u, $_)
        }
    }
    if(-not $succeeded){ Write-Warning "All sources failed for $($f.name)." }
}

Write-Output "Done. If downloads succeeded, reload the app page to use the local PDF.js copy."
