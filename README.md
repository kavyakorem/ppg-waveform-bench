# PPG Waveform Bench

A reproducible benchmark for estimating pulse rate from **synthetic PPG-like optical waveforms**. The project examines when classical signal processing is sufficient and when a learned model helps on artifact-heavy windows. It does not estimate intracranial pressure, use patient data, or establish clinical performance.

## Pipeline

1. **Simulate:** Generate 16-second windows at 64 Hz for virtual subjects. Each subject has its own baseline rate and pulse morphology. An asymmetric pulse and a secondary notch are stylized waveform components, not a biophysical model. Add drift, motion-like interference, dropouts, clipping, and mixed artifacts.
2. **Condition and extract:** Interpolate gaps, apply an offline zero-phase Butterworth bandpass, and robustly scale. Extract peak intervals, a Welch spectral rate, autocorrelation, peak widths and slopes, and quality features such as method agreement and dropout/clipping fractions.
3. **Compare:** Evaluate peak timing, Welch, histogram gradient boosting, and quality-based routing between Welch and the learned model. Also evaluate a separate rejection rule. Fit the model on training subjects, choose quality thresholds on validation subjects, then evaluate on disjoint test subjects.
4. **Audit:** Export reference values, predictions, quality scores, split IDs, and error by artifact. Report MAE, 90th-percentile absolute error, and coverage. Use a subject-level bootstrap for the standalone model's MAE interval.

## Run

Python 3.12; dependencies and default random seed are pinned.

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python ppg_bench.py --seed 23 --subjects 80 --windows 10
python -m unittest discover -s tests -v
```

Outputs: `results/metrics.json` and `results/heldout_predictions.csv`. See the [reference run](examples/reference_metrics.json) for the full results. Tests cover frequency recovery, clipping and missing samples, flatline handling, group separation, deterministic generation, and end-to-end output.

## Reference run

800 windows from 80 virtual subjects; 200 test windows from 20 held-out virtual subjects. All rates and errors are in beats per minute (bpm).

| Method | Test MAE | 90th-percentile absolute error | Test coverage |
| --- | ---: | ---: | ---: |
| Peak timing | 2.69 | 2.07 | 100% |
| Welch spectrum | 2.85 | 0.30 | 100% |
| Gradient boosting | 1.89 | 3.17 | 100% |
| Quality-routed Welch / gradient boosting | 1.16 | 1.51 | 100% |
| Gradient boosting with rejection | 0.97 | 2.52 | 77.5% |

The very low Welch 90th-percentile error alongside its higher MAE reflects a minority of large errors. The quality-routed method retained full coverage, while its MAE was still 2.49 bpm on motion windows and 3.06 bpm on mixed-artifact windows. Rejection numbers apply only to accepted windows: on those same 155 windows, Welch scored 0.11 bpm MAE, so the gated result does not show that ML wins on high-quality segments.

## Scope and limitations

The reference rates are simulator parameters, not independent measurements. Group separation reduces leakage within the simulation but does not test generalization to people, devices, or clinical conditions. Zero-phase filtering uses future samples, so this is an offline benchmark. An independent next experiment would evaluate pulse-rate estimates against ECG-derived reference rates in an open dataset such as the [PhysioNet BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/1.0.0/). An ICP study would require separate paired reference measurements and a distinct validation protocol.
