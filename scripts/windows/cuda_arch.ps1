# 输出本机 GPU 的计算能力（nvidia-smi 报 7.5 -> 75），多卡去重后用 ; 连接，
# 直接喂给 cmake 的 -DCMAKE_CUDA_ARCHITECTURES。取不到时输出 native，
# 交给 CMake 自己探测本机 GPU。
$archs = @()
try {
    $caps = nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
    $archs = @($caps | ForEach-Object { $_.Trim() -replace '\.', '' } |
        Where-Object { $_ -match '^\d+$' } | Sort-Object -Unique)
} catch {
    $archs = @()
}
if ($archs.Count -eq 0) {
    Write-Output 'native'
} else {
    Write-Output ($archs -join ';')
}
