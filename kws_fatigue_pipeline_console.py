"""
Консольная обёртка над `kws_fatigue_pipeline.py`.

Запуск:
  python kws_fatigue_pipeline_console.py --mode a --input "path.wav" --kws-model "best_kws_model_v3.keras" --fatigue-model "best_fatigue_model_18_b2_semi_f.keras"
"""

from kws_fatigue_pipeline import main


if __name__ == "__main__":
    main()

