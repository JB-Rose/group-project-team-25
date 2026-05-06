# Train all 60 combinations:
#   2 reward presets (v1, v2) x 2 augmentation (noaug, advaug) x 3 algorithms x 5 seeds
#
# Run from src/ directory:
#   .\train_all.ps1
#
# To run only a specific subset, comment out unwanted blocks below.

$python  = ".\venv\Scripts\python"
$seeds   = @(42, 123, 456, 789, 1024)
$presets = @("v1", "v2")
$algos   = @("ppo", "dg", "soft_ppo")
$augs    = @(
    @{ flag = "";                    label = "noaug"  },
    @{ flag = "--adversarial-augment"; label = "advaug" }
)

$total   = $seeds.Count * $presets.Count * $algos.Count * $augs.Count
$run     = 0

New-Item -ItemType Directory -Force -Path logs | Out-Null

foreach ($preset in $presets) {
    foreach ($aug in $augs) {
        foreach ($algo in $algos) {
            foreach ($seed in $seeds) {
                $run++
                $name    = "${algo}_$($aug.label)_${preset}_seed${seed}"
                $ckpt    = "rl_captcha/agent/checkpoints/$name"
                $log     = "logs/${name}_training.log"

                Write-Host ""
                Write-Host "[$run/$total] $name" -ForegroundColor Cyan

                # Skip only if training fully completed (log contains "Training complete.")
                if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Training complete\." -Quiet)) {
                    Write-Host "  Skipping - already fully trained" -ForegroundColor Yellow
                    continue
                }

                $argList = @(
                    "-u", "-m", "rl_captcha.scripts.train_ppo",
                    "--algorithm",      $algo,
                    "--reward-preset",  $preset,
                    "--train-seed",     $seed,
                    "--data-dir",       "data/",
                    "--save-path",      $ckpt,
                    "--total-timesteps","500000"
                )
                if ($aug.flag) { $argList += $aug.flag }

                & $python @argList | Tee-Object -FilePath $log

                if ($LASTEXITCODE -ne 0) {
                    Write-Host "  FAILED - check $log" -ForegroundColor Red
                } else {
                    Write-Host "  Done -> $ckpt" -ForegroundColor Green
                }
            }
        }
    }
}

Write-Host ""
Write-Host "All $total runs complete." -ForegroundColor Green
