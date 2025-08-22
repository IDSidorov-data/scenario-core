# --- НАСТРОЙКИ ДЛЯ SCENARIO-CORE ---
$projectPath = $PSScriptRoot
$outputFile = Join-Path $projectPath "project_context.txt"

$excludeDirs = @(
    ".git", ".github", "venv", "__pycache__", ".vscode"
)
$excludeFiles = @(
    "*.env*",
    ".streamlit/secrets.toml", # ВАЖНО: исключаем файл с секретами
    "*.pyc",
    "*.log",
    "project_context.txt",
    "create_core_context.ps1"
)

# --- БЫСТРАЯ РЕКУРСИВНАЯ ФУНКЦИЯ ---
function Get-FilteredChildItem-Fast {
    param(
        [string]$Path,
        [string[]]$ExcludeDirs
    )
    foreach ($item in Get-ChildItem -Path $Path -Force) {
        if ($item.PSIsContainer) {
            if ($ExcludeDirs -notcontains $item.Name) {
                $item
                Get-FilteredChildItem-Fast -Path $item.FullName -ExcludeDirs $ExcludeDirs
            }
        } else {
            $item
        }
    }
}

# --- НАЧАЛО СКРИПТА ---
Write-Host "Starting project context creation for scenario-core..."
Clear-Content $outputFile -ErrorAction SilentlyContinue

$filteredItems = Get-FilteredChildItem-Fast -Path $projectPath -ExcludeDirs $excludeDirs

# 1. Добавляем структуру проекта
"Project: $($projectPath.Split('\')[-1])" | Add-Content -Path $outputFile
"--- PROJECT STRUCTURE ---" | Add-Content -Path $outputFile
$filteredItems | ForEach-Object {
    $relativePath = $_.FullName.Substring($projectPath.Length + 1)
    if ($excludeFiles | Where-Object { $relativePath -like $_ }) { return }

    $indent = "  " * ($relativePath.Split('\').Count - 1)
    if ($_.PSIsContainer) {
        "$indent[d] $($_.Name)" | Add-Content -Path $outputFile
    } else {
        "$indent[f] $($_.Name)" | Add-Content -Path $outputFile
    }
}
"`n`n--- FILE CONTENT ---`n`n" | Add-Content -Path $outputFile

# 2. Добавляем содержимое файлов
$filesToInclude = $filteredItems | Where-Object { -not $_.PSIsContainer } | Where-Object {
    $fileName = $_.Name
    $isExcluded = $false
    foreach ($pattern in $excludeFiles) {
        if ($fileName -like $pattern) { $isExcluded = $true; break }
    }
    -not $isExcluded
}

foreach ($file in $filesToInclude) {
    $relativePath = $file.FullName.Substring($projectPath.Length + 1)
    "--- START OF FILE: $relativePath ---" | Add-Content -Path $outputFile
    Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue | Add-Content -Path $outputFile
    "`n--- END OF FILE: $relativePath ---`n" | Add-Content -Path $outputFile
}

Write-Host "Done! Project context saved to file:" -ForegroundColor Green
Write-Host $outputFile -ForegroundColor Yellow
$newSize = (Get-Item $outputFile).Length / 1KB
Write-Host "File size: $($newSize.ToString('F2')) KB" -ForegroundColor Cyan