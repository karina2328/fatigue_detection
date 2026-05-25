import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model

import vkr_fatigue_functions as vf
from macrof1_class import MacroF1


try:
    from tqdm import tqdm  # индикатор прогресса (на логику не влияет)
except Exception:  # pragma: no cover
    tqdm = None

# -----------------------------
# Константы и параметры
# -----------------------------


KWS_SR = 16000
KWS_MAX_LEN_FRAMES = 101
KWS_N_MFCC = 13

# Отобранные MFCC-коэффициенты для KWS (порядок важен)
KWS_SELECTED_FEATURE_NAMES_DEFAULT: List[str] = [
    "MFCC_7",
    "MFCC_11",
    "MFCC_3",
    "MFCC_10",
    "MFCC_6",
    "MFCC_0",
    "MFCC_12",
    "MFCC_9",
    "MFCC_1",
]

# Порог по умолчанию из vkr_kws_model_training.ipynb (максимизация F1)
KWS_DEFAULT_THRESHOLD = 0.5654


# Алиасы констант fatigue из vkr_fatigue_functions (единый источник истины)
FATIGUE_SR = vf.SR
FATIGUE_FRAME_LEN = vf.FRAME_LEN

# Индексы 18 признаков (fallback, если нет fatigue_selected_18_indices.npy)
FATIGUE_SELECTED_INDICES_DEFAULT: List[int] = [41, 7, 40, 2, 39, 5, 0, 6, 8, 10, 42, 12, 3, 9, 1, 11, 4, 13]

FATIGUE_CLASS_NAMES = {0: "нет утомления", 1: "слабое утомление", 2: "сильное утомление"}
FATIGUE_REPORT_CONFIDENCE_THRESHOLD = 0.5  # порог уверенности для формулировок и остановки

_PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_FATIGUE_SCALER_T = _PIPELINE_DIR / "scaler_temporal.pkl"
DEFAULT_FATIGUE_SCALER_H = _PIPELINE_DIR / "scaler_handcrafted.pkl"
DEFAULT_FATIGUE_INDICES = _PIPELINE_DIR / "fatigue_selected_18_indices.npy"


@dataclass
class FatigueAssets:
    """Загруженная модель утомления, скейлеры и индексы отобранных признаков."""

    model: Any
    selected_indices: np.ndarray
    scaler_t: Any
    scaler_h: Any


# -----------------------------
# Утилиты
# -----------------------------


def _ensure_mono(y: np.ndarray) -> np.ndarray:
    """
    Приводит сигнал к моно-формату.

    Если входной сигнал уже одномерный, возвращается как есть. Если сигнал двухканальный
    (или многоканальный), вычисляется среднее по оси каналов.

    Аргументы:
        y: Аудиосигнал в виде numpy-массива.

    Возвращает:
        Монофонический аудиосигнал numpy-массив (1D).
    """
    if y.ndim == 1:
        return y
    return np.mean(y, axis=0)


def load_audio_mono(file_path: str, sr: int) -> Tuple[np.ndarray, int]:
    """
    Загружает аудиофайл и приводит его к моно, с ресэмплингом к заданной частоте.

    Аргументы:
        file_path: Путь к аудиофайлу.
        sr: Целевая частота дискретизации (Гц).

    Возвращает:
        (audio, used_sr):
            audio: Аудиосигнал (mono), dtype=np.float32.
            used_sr: Фактическая частота дискретизации после загрузки (обычно равна `sr`).
    """
    y, used_sr = librosa.load(file_path, sr=sr, mono=True)
    y = _ensure_mono(y)
    return y.astype(np.float32), used_sr


def pad_or_trim_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Делает сигнал фиксированной длины для одномерных массивов.

    Если сигнал длиннее целевой длины — выполняется обрезка. Если короче — дополняется
    нулями в конец.

    Аргументы:
        x: Входной 1D массив (аудиосигнал).
        target_len: Желаемая длина массива.

    Возвращает:
        Массив длины `target_len`.
    """
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    out = np.zeros((target_len,), dtype=x.dtype)
    out[: x.shape[0]] = x
    return out


def pad_or_trim_2d_time(x: np.ndarray, target_len: int, axis_time: int) -> np.ndarray:
    """
    Делает фиксированной временную ось (axis_time) для 2D-признаков.

    Например, MFCC обычно имеют форму (n_mfcc, n_frames); в таком случае `axis_time=1`.

    Аргументы:
        x: Входной 2D массив (или больше размерностей).
        target_len: Желаемая длина временной оси.
        axis_time: Индекс оси, соответствующей времени.

    Возвращает:
        Массив, в котором размерность `axis_time` равна `target_len`.
    """
    n_time = x.shape[axis_time]
    if n_time == target_len:
        return x
    if n_time > target_len:
        slicer = [slice(None)] * x.ndim
        slicer[axis_time] = slice(0, target_len)
        return x[tuple(slicer)]
    # Дополняем нулями по нужной оси
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis_time] = (0, target_len - n_time)
    return np.pad(x, pad_width=pad_width, mode="constant")


def segment_audio_1s(
    audio: np.ndarray,
    sr: int,
    segment_duration_sec: float = 1.0,
    overlap_sec: float = 0.5,
) -> Iterable[Tuple[float, np.ndarray]]:
    """
    Разрезает длинный аудиосигнал на фрагменты длительностью `segment_duration_sec`
    с заданным перекрытием.

    Каждый фрагмент гарантированно имеет ровно заданную длительность:
    если в конце аудио данных не хватает, выполняется дополнение нулями.

    Аргументы:
        audio: Входной аудиосигнал (1D numpy-массив).
        sr: Частота дискретизации (Гц).
        segment_duration_sec: Длительность одного окна в секундах (по ТЗ обычно 1.0).
        overlap_sec: Сколько секунд перекрывается соседними окнами.

    Возвращает:
        Генератор, который по каждому окну выдаёт:
            (start_time_sec, segment_audio)
        где segment_audio — массив длины `int(sr * segment_duration_sec)`.
    """
    segment_len = int(round(sr * segment_duration_sec))
    if segment_len <= 0:
        raise ValueError("segment_duration_sec results in non-positive segment length")

    overlap_sec = float(overlap_sec)
    if not (0.0 <= overlap_sec < segment_duration_sec):
        raise ValueError("overlap_sec must be in [0, segment_duration_sec)")

    stride_samples = int(round(sr * (segment_duration_sec - overlap_sec)))
    stride_samples = max(1, stride_samples)

    audio_len = int(audio.shape[0])
    if audio_len <= segment_len:
        seg = np.zeros((segment_len,), dtype=np.float32)
        seg[:audio_len] = audio
        yield 0.0, seg
        return

    last_start = audio_len - segment_len
    starts = list(range(0, last_start + 1, stride_samples))
    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        seg = audio[start : start + segment_len]
        if seg.shape[0] < segment_len:
            padded = np.zeros((segment_len,), dtype=np.float32)
            padded[: seg.shape[0]] = seg
            seg = padded
        yield start / sr, seg.astype(np.float32)


# -----------------------------
# KWS: препроцессинг и извлечение признаков
# -----------------------------


def kws_feature_indices_from_names(feature_names: Sequence[str]) -> List[int]:
    """
    Преобразует имена выбранных признаков в индексы строк исходного (расширенного) набора.

    В обучающем ноутбуке для KWS полный порядок признаков задаётся как:
        [MFCC_0..MFCC_12, Delta_0..Delta_12, Delta2_0..Delta_12]
    Соответственно:
        MFCC_i  -> i
        Delta_i -> 13 + i
        Delta2_i -> 26 + i

    Аргументы:
        feature_names: Имена признаков (например, `MFCC_7`).

    Возвращает:
        Список индексов (int) в "полном" порядке признаков.
    """
    indices: List[int] = []
    for name in feature_names:
        if name.startswith("MFCC_"):
            idx = int(name.split("_")[1])
            indices.append(idx)
        elif name.startswith("Delta2_") or name.startswith("Delta2"):
            idx = int(name.split("_")[1]) if "_" in name else int(name.replace("Delta2_", ""))
            indices.append(26 + idx)
        elif name.startswith("Delta_") or name.startswith("Delta"):
            idx = int(name.split("_")[1]) if "_" in name else int(name.replace("Delta_", ""))
            indices.append(13 + idx)
        else:
            raise ValueError(f"Unknown KWS feature name: {name}")
    return indices


def kws_extract_temporal_features(audio_1s: np.ndarray) -> np.ndarray:
    """
    Извлекает входные признаки для KWS модели из 1-секундного аудиофрагмента.

    Формирует тензор формы (101, 9), где 9 — выбранные MFCC-коэффициенты из
    `KWS_SELECTED_FEATURE_NAMES_DEFAULT` (порядок важен).

    Извлечение сделано в логике `vkr_kws_feature_extraction`, но с оптимизацией:
    вычисляются только MFCC (без delta/delta2), так как в `vkr_kws_feature_selection`
    выбраны только MFCC_i.

    Аргументы:
        audio_1s: Аудиофрагмент ровно 1 секунду (1D numpy-массив в float32/float).

    Возвращает:
        numpy.ndarray формы (101, 9) — признаки для KWS модели.
    """
    audio = audio_1s.astype(np.float32)
    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_val > 0:
        audio = audio / max_val

    # Как в ноутбуке: для MFCC не задаём hop_length и n_fft.
    mfcc = librosa.feature.mfcc(y=audio, sr=KWS_SR, n_mfcc=KWS_N_MFCC).astype(np.float32)  # (13, time)

    # Обрезаем/дополняем по времени до 101 фрейма
    if mfcc.shape[1] > KWS_MAX_LEN_FRAMES:
        mfcc = mfcc[:, :KWS_MAX_LEN_FRAMES]
    elif mfcc.shape[1] < KWS_MAX_LEN_FRAMES:
        pad_width = KWS_MAX_LEN_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

    # (time, 13)
    mfcc_t = mfcc.T.astype(np.float32)

    # Индексы MFCC_i соответствуют строкам матрицы mfcc
    selected_mfcc_indices = [int(name.split("_")[1]) for name in KWS_SELECTED_FEATURE_NAMES_DEFAULT]
    selected = mfcc_t[:, selected_mfcc_indices]  # (101, 9)
    return selected


def kws_predict_phrase(
    kws_model,
    audio_1s: np.ndarray,
    threshold: float = KWS_DEFAULT_THRESHOLD,
) -> Tuple[bool, Dict[str, float]]:
    """
    Прогоняет KWS модель для определения наличия целевой фразы на 1-секундном фрагменте.

    Аргументы:
        kws_model: Загруженная Keras-модель keyword spotting (выдаёт 2 вероятности).
        audio_1s: 1-секундный аудиофрагмент (1D numpy-массив).
        threshold: Порог для решения "фраза есть".

    Возвращает:
        (has_phrase, info):
            has_phrase: bool — True если вероятность "есть фраза" >= threshold.
            info: словарь с вероятностями:
                - `p_has_phrase`
                - `p_no_phrase`
    """
    x = kws_extract_temporal_features(audio_1s)[None, ...]  # (1,101,9)
    x = x.astype(np.float32)
    probs = kws_model.predict(x, verbose=0)[0]  # (2,)
    if probs.shape[0] != 2:
        raise RuntimeError(f"Unexpected KWS output shape: {probs.shape}")
    # Класс 1 в обучении соответствует "есть фраза"
    p_has_phrase = float(probs[1])
    has_phrase = p_has_phrase >= float(threshold)
    return has_phrase, {"p_has_phrase": p_has_phrase, "p_no_phrase": float(probs[0])}


# -----------------------------
# Fatigue: загрузка активов и инференс (как vkr_predict_one_file.ipynb)
# -----------------------------


def _resolve_path(path: Optional[str], default: Path) -> Path:
    """Возвращает абсолютный путь: явный аргумент или значение по умолчанию рядом с модулем."""
    if path is None:
        return default
    return Path(path).expanduser().resolve()


def load_fatigue_assets(
    fatigue_model_path: str,
    scaler_t_path: Optional[str] = None,
    scaler_h_path: Optional[str] = None,
    selected_indices_path: Optional[str] = None,
) -> FatigueAssets:
    """
    Загружает модель утомления, StandardScaler-ы и индексы отобранных признаков.

    Логика совпадает с `load_model_and_normalizers` из `vkr_predict_one_file.ipynb`.

    Аргументы:
        fatigue_model_path: Путь к `.keras` модели.
        scaler_t_path: Путь к `scaler_temporal.pkl` (по умолчанию рядом с модулем).
        scaler_h_path: Путь к `scaler_handcrafted.pkl`.
        selected_indices_path: Путь к `fatigue_selected_18_indices.npy`; если файла нет —
            используется `FATIGUE_SELECTED_INDICES_DEFAULT`.

    Возвращает:
        `FatigueAssets` с model, selected_indices, scaler_t, scaler_h.
    """
    p_model = Path(fatigue_model_path).expanduser().resolve()
    p_scaler_t = _resolve_path(scaler_t_path, DEFAULT_FATIGUE_SCALER_T)
    p_scaler_h = _resolve_path(scaler_h_path, DEFAULT_FATIGUE_SCALER_H)
    p_indices = _resolve_path(selected_indices_path, DEFAULT_FATIGUE_INDICES)

    for label, p in [
        ("модель", p_model),
        ("scaler_temporal", p_scaler_t),
        ("scaler_handcrafted", p_scaler_h),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"Не найден файл {label}: {p}")

    model = load_model(
        str(p_model),
        custom_objects={"MacroF1": MacroF1},
        compile=False,
    )
    scaler_t = joblib.load(p_scaler_t)
    scaler_h = joblib.load(p_scaler_h)

    if p_indices.is_file():
        selected_indices = np.load(p_indices)
    else:
        selected_indices = np.array(FATIGUE_SELECTED_INDICES_DEFAULT, dtype=np.int64)

    return FatigueAssets(
        model=model,
        selected_indices=selected_indices,
        scaler_t=scaler_t,
        scaler_h=scaler_h,
    )


def preprocess_audio_array(
    audio_1s: np.ndarray,
    target_len: int = vf.TARGET_LEN,
    sr: int = vf.SR,
    noise_reduction_enabled: bool = True,
    normalization_type: str = "soft",
    normalization_rms_level: float = 0.1,
    silence_threshold: float = 0.01,
) -> Optional[np.ndarray]:
    """
    Предобработка 1-секундного фрагмента в памяти (аналог `vf.preprocess_audio` для файла).

    Использует функции из `vkr_fatigue_functions` без изменения этого модуля.

    Аргументы:
        audio_1s: Сырой 1-секундный фрагмент (1D).
        target_len: Целевая длина в отсчётах (обычно 16000).
        sr: Частота дискретизации.
        noise_reduction_enabled: Включить шумоподавление.
        normalization_type: 'soft' или 'hard'.
        normalization_rms_level: Целевой RMS для soft-нормализации.
        silence_threshold: Порог тишины (max abs).

    Возвращает:
        Обработанный сигнал float32 длины `target_len` или None для тихого/пустого фрагмента.
    """
    audio = _ensure_mono(np.asarray(audio_1s, dtype=np.float32))
    if audio.size == 0 or float(np.max(np.abs(audio))) < silence_threshold:
        return None

    if noise_reduction_enabled:
        audio = vf.reduce_noise_stationary(audio, sr)

    if normalization_type.lower() == "hard":
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 0:
            audio = audio / max_abs
    else:
        audio = vf.normalize_audio_soft(audio, target_rms=normalization_rms_level)

    audio = vf.adjust_duration(audio, target_len, sr)
    return audio.astype(np.float32)


def _align_temporal_frames(temporal_full: np.ndarray, frame_len: int = FATIGUE_FRAME_LEN) -> np.ndarray:
    """Обрезает или дополняет временную ось признаков до `frame_len` фреймов."""
    if temporal_full.shape[0] == frame_len:
        return temporal_full
    if temporal_full.shape[0] > frame_len:
        return temporal_full[:frame_len, :]
    pad_len = frame_len - temporal_full.shape[0]
    return np.pad(temporal_full, ((0, pad_len), (0, 0)), mode="constant")


def fatigue_predict_from_audio(
    assets: FatigueAssets,
    audio_proc: np.ndarray,
) -> Tuple[int, Dict[str, object]]:
    """
    Предсказание утомления по уже предобработанному аудио (как `predict_single_audio`).

    Аргументы:
        assets: Загруженные модель, скейлеры и индексы признаков.
        audio_proc: Предобработанный 1-секундный сигнал.

    Возвращает:
        (class_id, info) с ключами `probs` и `handcrafted`.
    """
    temporal_full = vf.extract_temporal_features(audio_proc, sr=vf.SR)
    temporal_full = _align_temporal_frames(temporal_full)
    temporal_selected = temporal_full[:, assets.selected_indices]

    jitter, shimmer = vf.extract_jitter_shimmer(audio_proc, sr=vf.SR)
    handcrafted = np.array([jitter, shimmer], dtype=np.float32)

    original_shape = temporal_selected.shape
    temporal_flat = temporal_selected.reshape(-1, original_shape[-1])
    temporal_norm = assets.scaler_t.transform(temporal_flat).reshape(original_shape)
    handcrafted_norm = assets.scaler_h.transform(handcrafted.reshape(1, -1)).astype(np.float32)[0]

    inp_temporal = temporal_norm[None, ...]
    inp_handcrafted = handcrafted_norm[None, ...]

    logits = assets.model([inp_temporal, inp_handcrafted], training=False)
    probs_arr = tf.nn.softmax(logits, axis=-1).numpy()[0]
    if probs_arr.shape[0] != 3:
        raise RuntimeError(f"Unexpected fatigue output shape: {probs_arr.shape}")

    class_id = int(np.argmax(probs_arr))
    probs = {k: float(probs_arr[k]) for k in range(3)}
    return class_id, {
        "probs": probs,
        "handcrafted": {"jitter": float(jitter), "shimmer": float(shimmer)},
    }


def fatigue_predict_level(
    assets: FatigueAssets,
    audio_1s: np.ndarray,
    noise_reduction_enabled: bool = True,
    normalization_type: str = "soft",
    normalization_rms_level: float = 0.1,
) -> Tuple[int, Dict[str, object]]:
    """
    Полный цикл: предобработка фрагмента + инференс модели утомления.

    Аргументы:
        assets: Результат `load_fatigue_assets`.
        audio_1s: 1-секундный аудиофрагмент (1D).
        noise_reduction_enabled: Шумоподавление при предобработке.
        normalization_type: 'soft' или 'hard'.
        normalization_rms_level: RMS для soft-нормализации.

    Возвращает:
        (class_id, info); при тишине — класс 0 и `probs=None`.
    """
    audio_proc = preprocess_audio_array(
        audio_1s,
        noise_reduction_enabled=noise_reduction_enabled,
        normalization_type=normalization_type,
        normalization_rms_level=normalization_rms_level,
    )
    if audio_proc is None:
        return 0, {"probs": None, "handcrafted": None, "reason": "silence or preprocessing failed"}

    return fatigue_predict_from_audio(assets, audio_proc)


# -----------------------------
# Режимы работы
# -----------------------------


def analyze_one_fragment_mode_b(
    wav_path: str,
    kws_model_path: str,
    fatigue_model_path: str,
    kws_threshold: float = KWS_DEFAULT_THRESHOLD,
    scaler_t_path: Optional[str] = None,
    scaler_h_path: Optional[str] = None,
    selected_indices_path: Optional[str] = None,
    noise_reduction_enabled: bool = True,
    normalization_type: str = "soft",
    normalization_rms_level: float = 0.1,
) -> None:
    """
    Режим `b`: анализирует один 1-секундный фрагмент аудиофайла.

    Логика:
        1) KWS: определяет, есть ли целевая фраза;
        2) если фраза есть — fatigue модель определяет класс утомления;
        3) в зависимости от результата печатает информацию в stdout.

    Аргументы:
        wav_path: Путь к аудиофайлу (будет взят первый 1-секундный фрагмент, если файл длиннее).
        kws_model_path: Путь к модели `best_kws_model_v3.keras`.
        fatigue_model_path: Путь к модели `best_fatigue_model_18_b2.keras` (вход temporal: 101×18).
        kws_threshold: Порог KWS для решения "фраза есть".
        scaler_t_path: Путь к `scaler_temporal.pkl`.
        scaler_h_path: Путь к `scaler_handcrafted.pkl`.
        selected_indices_path: Путь к `fatigue_selected_18_indices.npy`.
        noise_reduction_enabled: включить ли шумоподавление в fatigue preprocessing.
        normalization_type: 'soft' или 'hard'.
        normalization_rms_level: RMS уровень для soft-нормализации.

    Возвращает:
        None. Результаты выводятся в консоль.
    """
    kws_model = load_model(kws_model_path)
    fatigue_assets = load_fatigue_assets(
        fatigue_model_path,
        scaler_t_path=scaler_t_path,
        scaler_h_path=scaler_h_path,
        selected_indices_path=selected_indices_path,
    )

    audio, _ = load_audio_mono(wav_path, sr=KWS_SR)
    # Берём первые 1 секунду (дополняем нулями при необходимости)
    segment = pad_or_trim_1d(audio, int(KWS_SR * 1.0))

    has_phrase, _kws_info = kws_predict_phrase(kws_model, segment, threshold=kws_threshold)
    if not has_phrase:
        print("нет фразы")
        return

    fatigue_class, fat_info = fatigue_predict_level(
        fatigue_assets,
        segment,
        noise_reduction_enabled=noise_reduction_enabled,
        normalization_type=normalization_type,
        normalization_rms_level=normalization_rms_level,
    )

    probs = fat_info.get("probs")
    if probs is None:
        print("есть фраза; утомление: нет данных (пропуск фрагмента)")
        return

    print("есть фраза")
    print("вероятности классов утомления:")
    for cls_id in [0, 1, 2]:
        print(f"  {cls_id}: {FATIGUE_CLASS_NAMES[cls_id]} -> {probs[cls_id]:.4f}")
    print(f"выбран класс: {fatigue_class} -> {FATIGUE_CLASS_NAMES[fatigue_class]}")


def _save_segment_wav(
    out_dir: str,
    stem: str,
    start_time_sec: float,
    class_id: int,
    audio_1s: np.ndarray,
    sr: int = FATIGUE_SR,
) -> str:
    """
    Сохраняет 1-секундный аудиофрагмент в WAV-файл.

    Имя файла строится так, чтобы включать метку класса утомления (_0/_1/_2):
    в текущей реализации формат: `<stem>_t{start:.3f}_{class_id}.wav`

    Аргументы:
        out_dir: Папка для сохранения.
        stem: Имя исходного файла без расширения.
        start_time_sec: Начало фрагмента в общей записи (в секундах).
        class_id: Метка утомления (0/1/2).
        audio_1s: Аудиосигнал фрагмента (1 сек, 1D numpy-массив).
        sr: Частота дискретизации при сохранении (обычно 16000).

    Возвращает:
        Полный путь к сохранённому WAV-файлу (str).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{stem}_t{start_time_sec:.3f}_{class_id}.wav"  # в имени есть суффикс _0/_1/_2
    out_path = str(Path(out_dir) / fname)
    sf.write(out_path, audio_1s.astype(np.float32), sr)
    return out_path


def _fatigue_report_message(class_id: int, confidence: float) -> Optional[str]:
    """
    Возвращает текст сообщения для режима a в зависимости от класса и уверенности модели.

    Аргументы:
        class_id: Предсказанный класс (1 — слабое утомление, 2 — сильное утомление).
        confidence: Вероятность предсказанного класса.

    Возвращает:
        Строка сообщения без указания секунды или None, если класс не требует сообщения.
    """
    if class_id == 1:
        if confidence < FATIGUE_REPORT_CONFIDENCE_THRESHOLD:
            return "возможны незначительные признаки слабого утомления"
        return "зафиксировано слабое утомление"
    if class_id == 2:
        if confidence < FATIGUE_REPORT_CONFIDENCE_THRESHOLD:
            return "обнаружено повышение уровня утомления"
        return "зафиксировано сильное утомление"
    return None


def analyze_long_audio_mode_a(
    wav_path: str,
    kws_model_path: str,
    fatigue_model_path: str,
    overlap_sec: float = 0.5,
    kws_threshold: float = KWS_DEFAULT_THRESHOLD,
    scaler_t_path: Optional[str] = None,
    scaler_h_path: Optional[str] = None,
    selected_indices_path: Optional[str] = None,
    noise_reduction_enabled: bool = True,
    normalization_type: str = "soft",
    normalization_rms_level: float = 0.1,
    save_found_fragments: bool = False,
    save_dir: Optional[str] = None,
    stop_on_first_strong: bool = True,
) -> None:
    """
    Режим `a`: анализирует длинную аудиозапись, разрезая её на 1-секундные окна с перекрытием.

    Для каждого окна последовательно:
        1) KWS определяет, есть ли фраза;
        2) если фраза есть — fatigue модель определяет класс утомления;
        3) если `stop_on_first_strong=True`: печатаются сообщения о слабом/сильном утомлении
           (формулировка зависит от уверенности модели в классе); поиск продолжается,
           пока не найдена сильное утомление с уверенностью > 0.5 — тогда обработка завершается.

    Опционально сохраняет фрагменты с фразой и помеченным классом утомления в выбранную папку.

    Аргументы:
        wav_path: Путь к длинному аудиофайлу.
        kws_model_path: Путь к модели `best_kws_model_v3.keras`.
        fatigue_model_path: Путь к модели `best_fatigue_model_18_b2.keras` (вход temporal: 101×18).
        overlap_sec: Перекрытие соседних окон (в секундах).
        kws_threshold: Порог KWS для решения "фраза есть".
        scaler_t_path: Путь к `scaler_temporal.pkl`.
        scaler_h_path: Путь к `scaler_handcrafted.pkl`.
        selected_indices_path: Путь к `fatigue_selected_18_indices.npy`.
        noise_reduction_enabled: включить шумоподавление в fatigue preprocessing.
        normalization_type: 'soft' или 'hard'.
        normalization_rms_level: RMS уровень для soft-нормализации.
        save_found_fragments: если True — сохраняются фрагменты.
        save_dir: папка для сохранения фрагментов (нужна если `save_found_fragments=True`).
        stop_on_first_strong: если True — сообщения об утомлении и остановка только при сильном
           утомлении с уверенностью > 0.5. Если False — без сообщений и без досрочной остановки.

    Возвращает:
        None. Результаты и сообщения печатаются в stdout.
    """
    kws_model = load_model(kws_model_path)
    fatigue_assets = load_fatigue_assets(
        fatigue_model_path,
        scaler_t_path=scaler_t_path,
        scaler_h_path=scaler_h_path,
        selected_indices_path=selected_indices_path,
    )

    audio, _ = load_audio_mono(wav_path, sr=KWS_SR)  # KWS_SR == FATIGUE_SR
    stem = Path(wav_path).stem

    if save_found_fragments and not save_dir:
        raise ValueError("save_dir must be provided if save_found_fragments=True")

    iterable = segment_audio_1s(audio, KWS_SR, segment_duration_sec=1.0, overlap_sec=overlap_sec)
    if tqdm is not None and not isinstance(iterable, list):
        pass  # для генератора tqdm требует длину; здесь оставляем без прогресс-бара

    weak_reported = False  # первое сообщение о слабом утомлении
    strong_elevated_reported = False  # первое сообщение о повышенном уровне (сильная, conf < 0.5)

    for idx, (start_time_sec, seg) in enumerate(iterable):
        has_phrase, kws_info = kws_predict_phrase(kws_model, seg, threshold=kws_threshold)  # Шаг 3..4: KWS — детект фразы
        if not has_phrase:
            continue  # Если фраза не найдена — пропускаем фрагмент

        fatigue_class, fat_info = fatigue_predict_level(
            fatigue_assets,
            seg,
            noise_reduction_enabled=noise_reduction_enabled,
            normalization_type=normalization_type,
            normalization_rms_level=normalization_rms_level,
        )  # Шаг 5..7: fatigue — определение класса утомления

        if save_found_fragments:
            assert save_dir is not None
            _save_segment_wav(save_dir, stem=stem, start_time_sec=start_time_sec, class_id=fatigue_class, audio_1s=seg, sr=FATIGUE_SR)

        if stop_on_first_strong and fatigue_class in (1, 2):
            probs = fat_info.get("probs")
            if probs is not None:
                confidence = float(probs[fatigue_class])
                n = int(start_time_sec)
                message = _fatigue_report_message(fatigue_class, confidence)

                if fatigue_class == 1 and not weak_reported and message:
                    print(f"{message} на {n}-й секунде")
                    weak_reported = True
                elif fatigue_class == 2 and message:
                    if confidence < FATIGUE_REPORT_CONFIDENCE_THRESHOLD:
                        if not strong_elevated_reported:
                            print(f"{message} на {n}-й секунде")
                            strong_elevated_reported = True
                    else:
                        print(f"{message} на {n}-й секунде")
                        return

        # Если fatigue_class==0 или stop_on_first_strong=False — продолжаем поиск


# -----------------------------
# Консольный запуск
# -----------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Формирует аргумент-парсер для консольного запуска.

    Возвращает:
        `argparse.ArgumentParser` с поддержкой параметров:
        - `--mode` (a/b)
        - `--input`
        - пути к моделям
        - пороги/перекрытие
        - параметры fatigue preprocessing
        - опции сохранения фрагментов
    """
    p = argparse.ArgumentParser(description="KWS + fatigue unified pipeline (modes a/b).")
    p.add_argument("--mode", choices=["a", "b"], required=True)
    p.add_argument("--input", required=True, help="Path to wav file. In mode a it's a long file; in mode b it's a 1-second fragment.")

    p.add_argument("--kws-model", default="best_kws_model_v3.keras", help="Path to best_kws_model_v3.keras")
    p.add_argument(
        "--fatigue-model",
        default="best_fatigue_model_18_b2.keras",
        help="Путь к best_fatigue_model_18_b2.keras (101×18 признаков)",
    )
    p.add_argument("--scaler-temporal", default=None, help="Путь к scaler_temporal.pkl")
    p.add_argument("--scaler-handcrafted", default=None, help="Путь к scaler_handcrafted.pkl")
    p.add_argument(
        "--fatigue-indices",
        default=None,
        help="Путь к fatigue_selected_18_indices.npy",
    )

    p.add_argument("--kws-threshold", type=float, default=KWS_DEFAULT_THRESHOLD)

    p.add_argument("--overlap-sec", type=float, default=0.5, help="Used only in mode a. overlap between 1s windows.")

    p.add_argument("--noise-reduction", action="store_true", help="Enable noise reduction in fatigue preprocessing.")
    p.add_argument("--no-noise-reduction", dest="noise_reduction", action="store_false")
    p.set_defaults(noise_reduction=True)

    p.add_argument("--normalization-type", choices=["soft", "hard"], default="soft")
    p.add_argument("--normalization-rms-level", type=float, default=0.1)

    p.add_argument("--save-found-fragments", action="store_true", help="Save fragments with detected phrase and their fatigue class.")
    p.add_argument("--save-dir", default=None, help="Output directory for saved fragments.")

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Точка входа CLI: парсит аргументы и запускает обработку в выбранном режиме.

    Аргументы:
        argv: Список строк аргументов (как в `sys.argv[1:]`). Если None — используется значение
              по умолчанию, которое считывается из реального запуска.

    Возвращает:
        None. Результаты печатаются в stdout.
    """
    args = _build_arg_parser().parse_args(argv)

    if args.mode == "b":
        analyze_one_fragment_mode_b(
            wav_path=args.input,
            kws_model_path=args.kws_model,
            fatigue_model_path=args.fatigue_model,
            kws_threshold=args.kws_threshold,
            scaler_t_path=args.scaler_temporal,
            scaler_h_path=args.scaler_handcrafted,
            selected_indices_path=args.fatigue_indices,
            noise_reduction_enabled=args.noise_reduction,
            normalization_type=args.normalization_type,
            normalization_rms_level=args.normalization_rms_level,
        )
    else:
        analyze_long_audio_mode_a(
            wav_path=args.input,
            kws_model_path=args.kws_model,
            fatigue_model_path=args.fatigue_model,
            overlap_sec=args.overlap_sec,
            kws_threshold=args.kws_threshold,
            scaler_t_path=args.scaler_temporal,
            scaler_h_path=args.scaler_handcrafted,
            selected_indices_path=args.fatigue_indices,
            noise_reduction_enabled=args.noise_reduction,
            normalization_type=args.normalization_type,
            normalization_rms_level=args.normalization_rms_level,
            save_found_fragments=args.save_found_fragments,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()

