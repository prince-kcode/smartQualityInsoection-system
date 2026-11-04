$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'
$python = '.\.venv\Scripts\python.exe'
$categories = @(
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
)
$comparisonCategories = @('bottle', 'metal_nut')

& $python scripts\inspect_dataset.py
if ($LASTEXITCODE -ne 0) { throw 'Dataset audit failed; training stopped.' }

foreach ($category in $categories) {
    Write-Host "Training PatchCore: $category"
    & $python scripts\train_patchcore.py --category $category --results-dir results\retrained
    if ($LASTEXITCODE -ne 0) { throw "PatchCore training failed for $category" }
}

foreach ($category in $comparisonCategories) {
    Write-Host "Training CNN autoencoder: $category"
    & $python scripts\train_autoencoder.py --category $category --epochs 25 --results-dir results\retrained\autoencoder
    if ($LASTEXITCODE -ne 0) { throw "Autoencoder training failed for $category" }
}

& $python scripts\export_model_charts.py --results-dir results\retrained\autoencoder
if ($LASTEXITCODE -ne 0) { throw 'Model chart export failed.' }
Write-Host 'All 15 PatchCore categories and both reference autoencoders trained; evaluation artifacts exported.'
