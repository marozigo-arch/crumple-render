# Blender 3D Calibration — Управляющий документ для генерации роликов (Kaggle/headless)

Версия: **v4 (по результатам прогона blender_calibration_out_v4.zip)**  
Назначение: **LLM-понятный** и **машиночитаемый** стандарт, который прикладывается к проекту и к любым запросам на генерацию видео/эффектов в Blender, чтобы исключить ошибки позиционирования и обеспечить сопоставимость **лог ↔ пиксели рендера**.

---

## 0) Контекст калибровки и состав артефактов

**Архив-эталон:** `blender_calibration_out_v4.zip`

**Параметры рендера прогона (v4):**
- Движок: **Cycles (CPU)**
- Разрешение: **640×360**
- FPS: **30**
- Кадры: **1..360** (12 секунд)
- Samples: **8**
- Max bounces: **1**

**Артефакты, которые должны существовать после любого калибровочного/рабочего прогона:**
- `video/*.mp4` — результирующее видео
- `frames/frame_####.png` — покадровый рендер
- `logs/frame_log.jsonl` — покадровые логи (JSONL)
- `blender_render.log` — stdout Blender
- `logs/summary.md` — краткая сводка запуска
- `env.json` — окружение/OCIO (best-effort)

---

## 1) Координатные аксиомы Blender (обязательны)

### 1.1 World-space
- Система координат **правосторонняя**
- **Z вверх**
- X/Y — плоскость пола

### 1.2 Camera local-space (критично)
- Камера смотрит вдоль **локальной оси -Z**
- Локальная **+Y** — вверх
- Локальная **+X** — вправо

**Неприменимо/опасно:** переносить конвенцию “forward = +Z” из других движков. В Blender `forward = -Z`.

---

## 2) Единственный корректный источник “истины” (для совпадения с рендером)

### 2.1 Правило истины
Для всего, что сравнивается с финальным PNG/MP4, использовать **evaluated depsgraph**:

- `depsgraph = bpy.context.evaluated_depsgraph_get()`
- `obj_eval = obj.evaluated_get(depsgraph)`
- **истина трансформа:** `obj_eval.matrix_world`
- **истина камеры:** `cam_eval.matrix_world`

### 2.2 Почему RAW часто “врёт”
В рабочих сценах расхождения между RAW и финальным рендером появляются из-за:
- constraints/drivers
- modifiers/geometry nodes
- parenting/armature
- порядка обновления depsgraph/view layer

Даже если в калибровке RAW==EVAL, правило сохраняется как стандарт.

---

## 3) Критический вывод калибровки v4: ловушка `frame_change_pre`

### 3.1 Симптом (выявлен на сегменте roll)
Сегмент `D_rotate_roll` (кадры 145–180) задуман как: **look-at + чистый roll вокруг forward**.

Из логов: метрика качества look-at
- `forward = camera_forward_world`
- `dir = normalize(target - cam_loc)`
- `lookat_dot = dot(forward, dir)`

Показала провал: **min lookat_dot ≈ 0.6508**, затем восстановление к ~1.0 к финалу сегмента.

### 3.2 Причина
Roll вычислялся на базе **устаревшего forward** (взятый из `cam.matrix_world` сразу после изменения ориентации, до обновления depsgraph/view layer).

### 3.3 Жёсткие правила проекта
Если в одном кадре вы:
1) изменили `rotation_*`, и затем
2) читаете `matrix_world` и вычисляете `forward/up/right`

…то вы рискуете получить систематическое расхождение “математика ↔ рендер”.

Разрешённые решения (выберите минимум одно):
- После изменения трансформа: `bpy.context.view_layer.update()` и только потом читать `matrix_world`.
- Не читать `matrix_world` в pre: вычислять оси **из построенного quaternion/матрицы**, которые вы только что создали.
- Делать любые сравнения и логирование “истины” в `frame_change_post` через evaluated depsgraph.

---

## 4) Канонические формулы (эталон для всех будущих эффектов)

### 4.1 Базис камеры в мире (из matrix_world)
Пусть `MW = cam_eval.matrix_world`, тогда:

- `right_world   = normalize(MW.to_3x3() @ (1,0,0))`
- `up_world      = normalize(MW.to_3x3() @ (0,1,0))`
- `forward_world = normalize(MW.to_3x3() @ (0,0,-1))`  ← важно: **-Z**

### 4.2 Look-at (camera forward = -Z)
- `direction = normalize(target - cam_loc)`
- `q_look = direction.to_track_quat('-Z', 'Y')`

### 4.3 Roll вокруг forward (стабильно)
Правильная последовательность:
1) построить `q_look`
2) взять forward **из q_look**, а не из `matrix_world`
3) построить `q_roll` вокруг forward
4) `q_final = q_roll @ q_look`

Псевдокод:
```python
direction = (target - cam_loc).normalized()
q_look = direction.to_track_quat('-Z','Y')
fwd = (q_look @ Vector((0,0,-1))).normalized()
q_roll = Quaternion(fwd, roll_angle)
q_final = q_roll @ q_look
```

### 4.4 Dolly вдоль forward
- `cam_loc_new = cam_loc + forward_world * d`  
где `forward_world` должен быть актуальным (evaluated или из q).

### 4.5 Проекция world → экран (Blender-эталон)
Использовать `world_to_camera_view(scene, cam_eval, world_point)` → `u,v,depth` и:
- `px = u * res_x`
- `py = (1 - v) * res_y`
- `in_frame = (0<=u<=1 and 0<=v<=1 and depth>=0)`

---

## 5) Стандарт логирования (JSONL) для сопоставления “лог ↔ пиксели”

### 5.1 Базовые правила
- **1 строка = 1 кадр**
- Формат: UTF-8 JSONL
- Логировать **после evaluation** (в `frame_change_post`) и/или через evaluated objects.
- Логировать минимум: камера, ключевые объекты, проекции.

### 5.2 Рекомендуемый контракт записи (минимально достаточный)
Каждая запись должна включать:

**A) Общие поля**
- `frame` (int)
- `time_utc` (строка ISO)
- `segment` (строка)
- `t` (0..1)
- `render` (engine/res/fps/samples/bounces)

**B) Camera**
- `extrinsics`: `matrix_world_4x4`, `loc_world`, `rot_quat_world`
- `basis_world`: `right/up/forward`
- `intrinsics` (обязательно для внешней математики):
  - `lens_mm`
  - `sensor_width`, `sensor_height`, `sensor_fit`
  - `shift_x`, `shift_y`
  - `clip_start`, `clip_end`
  - `pixel_aspect_x/y`
  - (желательно) `proj_matrix_4x4 = cam.calc_matrix_camera(...)`

**C) Objects**
Для каждого опорного объекта:
- `matrix_world_4x4`, `loc_world`, `rot_quat_world`, `scale`
- `proj` для центра
- (желательно) `bbox_proj` для 8 углов bbox

**D) Metrics**
- `lookat_dot` (если сегмент look-at)
- `basis_orthonormal_error`
- `in_frame` статистика ключевых якорей

---

## 6) Требования к калибровочной сцене (визуальные якоря)

### 6.1 Обязательные якоря
- Оси X/Y/Z с цветовым кодированием + подписи
- Маркеры в разных квадрантах (±X, ±Y, +Z)
- Яркий “probe”-объект, который движется по тестовому сценарию
- Плоскость пола (желательно с сеткой/шкалой)
- HUD с frame/segment/t/lens/cam_loc/rot

### 6.2 Урок v4
Стресс-сегменты могут выбрасывать якоря из кадра (например yaw). Для “чистых калибровочных” сегментов рекомендуется удерживать ключевые якоря **in_frame ≥ 90%** кадров.

---

## 7) Пост-валидация после рендера (обязательный чек-лист)

### 7.1 Полнота артефактов
- PNG кадры без пропусков 1..N
- JSONL содержит N (или N±1) записей
- MP4 успешно закодирован

### 7.2 Геометрические проверки
- Ортонормальность базиса:
  - |right|≈|up|≈|forward|≈1
  - dot(right,up)≈0; dot(right,forward)≈0; dot(up,forward)≈0
- Look-at сегменты:
  - `lookat_dot` должен быть близок к 1 (например ≥0.999).  
  - Если на “чистом roll” падает `lookat_dot` — почти наверняка forward взят неактуально (см. раздел 3).

### 7.3 Производительность
- Снимать `Time per frame` из blender_render.log
- Следить за регрессиями по сегментам

---

## 8) Kaggle/headless стандарт запуска (устойчивость)

Рекомендуется:
- `blender -b` (background)
- `--factory-startup` (минимизация UI-зависимостей)
- Окружение:
  - `SDL_VIDEODRIVER=dummy`
  - `DISPLAY=""` (или unset)
- Color management warnings не должны ломать прогон; по возможности `view_transform = Standard`.

---

## 9) Готовый блок для вставки в будущие LLM-запросы (конституция)

Скопируйте и вставляйте в начало любого запроса на генерацию Blender-ролика:

```text
ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА (Blender/Kaggle calibration):
1) World: right-handed, Z-up.
2) Camera local axes: forward = -Z, up = +Y, right = +X.
3) Истина для сопоставления с рендером: evaluated depsgraph (obj.evaluated_get(depsgraph).matrix_world).
4) Нельзя в frame_change_pre после изменения rotation сразу читать cam.matrix_world и брать forward — сначала view_layer.update() или вычисляй оси из построенного quaternion/матрицы.
5) Для look-at: direction.to_track_quat('-Z','Y'); для roll: roll вокруг forward, вычисленного из q_look.
6) Логирование: JSONL на каждый кадр. Включать camera extrinsics + intrinsics (lens/sensor/shift/clip/pixel_aspect + ideally calc_matrix_camera) + basis + проекции ключевых точек.
7) После рендера выполнять валидацию: orthonormal basis, lookat_dot на ожидаемых сегментах, in_frame статистика якорей.
8) Kaggle headless: blender -b, предпочтительно --factory-startup, SDL_VIDEODRIVER=dummy, DISPLAY="". Color management warnings не должны ломать прогон.
```

---

## 10) Короткая выжимка (в один экран)
1) **Forward камеры = -Z.**
2) **Truth = evaluated depsgraph.**
3) **Не вычисляй оси из matrix_world в pre без update.**
4) **Логируй intrinsics + projection matrix** для пиксельной внешней репликации.
5) Делай **пост-валидацию** (lookat_dot, ортонормальность, in_frame, полнота кадров).

---

Конец документа.
