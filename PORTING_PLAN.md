# Piano di porting: dal prototipo di tesi allo smartphone

Branch di lavoro: `porting` (creato da `main` @ `4a5497f`).

Obiettivo complessivo: portare il MSCKF monoculare da prototipo Python (Linux) a un sistema che gira in tempo reale su uno smartphone, passando per macOS, correzione dei bug e ottimizzazione.

Le fasi vanno svolte **in ordine**: ogni fase produce una baseline misurabile che la fase successiva non deve peggiorare.

---

## Fase 1 — Porting su macOS (Apple Silicon, `osx-arm64`)

### 1.1 Ambiente Pixi
- [x] Aggiungere `osx-arm64` a `platforms` in `pixi.toml` (mantenere `linux-64`).
- [x] Rimuovere il canale `pytorch` (deprecato) e prendere `pytorch` da `conda-forge`.
- [x] Sostituire `cpuonly` con `pytorch-cpu` in `[target.linux-64.dependencies]` (`cpuonly` non esiste su macOS). Su macOS la build è `cpu_generic`, con MPS disponibile.
- [x] Rinominare `[project]` in `[workspace]` (deprecato in Pixi 0.84).
- [x] Rigenerare `pixi.lock` con `pixi install` e verificare che risolva su entrambe le piattaforme.
- [x] Eseguire `pixi run verify-installation` e `pixi run test-imports`.

### 1.2 Submodule e XFeat
- [x] Verificare che `src/accelerated_features` sia inizializzato (`git submodule update --init`) e che i pesi in `weights/` siano presenti. Sono versionati direttamente nel submodule, senza LFS.
- [x] Controllare come XFeat sceglie il device (`cuda` → fallback `cpu`); valutare l'opzione `mps` (GPU Apple) in modo configurabile, con fallback su CPU.
  - Aggiunti `--device {auto,cpu,mps,cuda}` e `MSCKFParameters.device`. `auto` sceglie cuda → mps → cpu; XFeat viene spostato sul device dopo il caricamento, senza modificare il submodule. È impostato anche `PYTORCH_ENABLE_MPS_FALLBACK=1`.
  - Misura (M-series, 640×480, top_k=300): CPU 48 ms/frame, MPS 16 ms/frame. Keypoint identici, descrittori e score con differenze < 1e-6.
- [x] Sostituire l'hack `sys.path.insert` in `FeatureExtractor.py` e `sys.path.append('../../')` in `main.py` con import puliti: `src/accelerated_features` è aggiunto al `PYTHONPATH` di Pixi. Rimosso anche `sys.path.append('../')` da `synthetic_generator.py`.

### 1.3 Dataset e prima esecuzione
- [x] Scaricare una sequenza TartanAir piccola in `data/tartanair/` con il relativo `camera_info.csv`.
  - Usata `abandonedfactory/Easy/P001`: 434 immagini `image_left` + `pose_left.txt`, presi da `image_left.zip` (7 GB) del server AirLab leggendo via HTTP range solo i file di P001 (164 MB).
  - A `trajectory.csv` è stata aggiunta l'intestazione `x y z qx qy qz qw`: il generatore usa `read_csv` con header e altrimenti perde la prima posa.
- [x] Verificare che Rerun (`rr.spawn()`) apra il viewer su macOS.
  - Rerun 0.43: `Transform3D(axis_length=...)` non esiste più → `TransformAxes3D` statico. Versione fissata a `>=0.43,<0.44`.
  - `rr.save()` + `rr.spawn()` si sostituivano a vicenda (il `.rrd` restava vuoto) → `rr.set_sinks(GrpcSink, FileSink)`.
- [x] Eseguire `pixi run main --source tartanair --sequence P001 --max_frames 4000 --noise_level mid --save_results` (≈3900 passi IMU = sequenza intera, 73 s).
  - ⚠️ **Risultato non comparabile con la tesi:** RTE medio 57% (picco ≈ passo 2950), ROE 0,56%, ATE fino a ~10 m e fuori dai bound 3σ. Sui primi 300 passi invece RTE è 0,35%.
  - [ ] Capire la causa prima di fissare la baseline: differenze nei dati (es. header di `trajectory.csv`), versioni delle librerie, seed, o i bug già noti della Fase 2.
- [ ] Riprodurre i dataset della tesi (Cap. 7.3: primi 2500 frame di ogni sequenza, IMU a 200 Hz). P001 **non** è tra questi.
  - **TartanAir ME005, ME007**: sono il test set monoculare della CVPR SLAM Challenge, non il training set. Immagini: `tartanair-test-mono-release.tar.gz` (7,65 GB, Google Drive; il server `airlab-share:8081` non risponde). Ground truth: il link Box del README restituisce 404 e il mirror Hugging Face ufficiale ha solo il training → servono copie locali o un contatto con gli autori.
  - **PeringLab Deer Running (`deer_run`), Deer VR Slow (`deer_vr_slow`)**: disponibili su `https://www.doc.ic.ac.uk/~wl208/lmdata/` (zip 272 MB e 1,1 GB, pose `.gt` già nel formato `peringlab`). Il server blocca le richieste parallele (403): scaricare con una sola richiesta.
  - Servono anche gli intrinseci PeringLab per `camera_info.csv`.

### 1.3b Dataset reale: EuRoC MAV (scelto come riferimento al posto dei dataset della tesi)
- [x] Scaricare `vicon_room1.zip` da [ETH Research Collection](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) e estrarre `V1_01_easy`, `V1_02_medium`, `V1_03_difficult` (solo `cam0`, `imu0`, `vicon0`, ground truth) in `data/euroc/<seq>/mav0/`.
- [x] `EurocConverter` (`dataset/tools/dataset_generators/euroc_converter.py`): IMU reale (niente generazione), timestamp relativi in secondi, ground truth ricampionato sui timestamp IMU (NaN dove manca), calibrazione da `sensor.yaml`. Le immagini vengono undistorte con `cv2.remap` → il filtro resta pinhole con la nuova K.
- [x] `main.py --source euroc`:
  - estrinseci camera-IMU reali (con traslazione);
  - parametri di rumore dall'IMU;
  - sincronizzazione con `imu_ts >= cam_ts` invece dell'uguaglianza dei timestamp arrotondati;
  - ground truth allineato alla stima alla prima posa disponibile;
  - risultati in `data/euroc/<seq>/results/`.
- [x] **Convenzione del segno della gravità:** i dataset generati simulano un accelerometro che misura Rᵀ(a + g), mentre quello reale misura Rᵀ(a − g). Per EuRoC si usa `W_gravity = +9.81 z`. ⚠️ Da uniformare nella Fase 2: generatore e integrazione vanno portati alla convenzione fisica.
- [x] Bias reali: nuovi parametri `estimate_initial_gyroscope_bias` (media del giroscopio prima del primo frame, richiede una partenza statica) e `initial_covariance_diagonal` (prima la covarianza iniziale era nulla e il filtro non poteva stimare i bias). Default invariati per i dati sintetici.
- Risultato su V1_01_easy (primi 3000 passi IMU = 15 s):
  - senza le correzioni sui bias: yaw in deriva a 0,077 rad/s, pari al bias del giroscopio z;
  - con le correzioni: consistente (dentro i 3σ) fino al passo ~1550, poi salti a gradino di roll e pitch negli update visivi → divergenza (ATE z ~20 m). È il punto di partenza della Fase 2.

### 1.4 Baseline di riferimento (prima di toccare l'algoritmo)
- [ ] Salvare `results.txt`, i plot e i parametri usati come **baseline v0** (es. in `benchmarks/baseline_v0/`), per almeno 2 sequenze × 3 livelli di rumore.
- [ ] Fissare i seed (`np.random.seed`, `torch.manual_seed`) per avere run riproducibili.
- [ ] Scrivere uno script `benchmark.py` (o un task Pixi `pixi run bench`) che lanci in batch le configurazioni e stampi una tabella RTE/ROE/ATE/tempi.

**Criterio di uscita:** il codice gira su Mac con risultati comparabili a quelli della tesi.

---

## Fase 2 — Correzione bug

Per ogni bug: test che lo riproduce → fix → confronto con la baseline. Un commit per bug.

Riferimento di valutazione: EuRoC **V1_01_easy, primi 6000 passi IMU (30 s, 8,25 m)** con `--no_viewer` (~50 s per run).

| Commit | Correzione | ATE RMSE | Dentro 3σ |
|---|---|---|---|
| — | baseline (dopo il supporto EuRoC) | 126,9 m | 21,4 % |
| `55c2f38` | segno della correzione di orientazione in `correct()` | 64,9 m | 23,3 % |
| `b9d062c` | Jacobiano di augmentation, blocco posizione | 26,3 m | 23,4 % |
| `7df4250` | transizione con vincolo di osservabilità (trasposte JPL) | 1,90 m | 32,4 % |
| `eb307fd` | aliasing degli stati null in `correct()` | 1,93 m | 15,7 % |
| `1d0feed` | test epipolare (ordine dei punti, valore assoluto) → distanza di Sampson in px, soglia 10 px | 1,04 m | 15,7 % |
| `2678dac` | `sigma_image` in pixel (prima 0,1 normalizzato ≈ 40 px) → 1 px | **0,17 m** | 8,5 % |

Sequenze complete dopo `2678dac` (σ = 1 px, soglia epipolare 10 px):

| Sequenza | Distanza | ATE RMSE | Orientazione RMSE | Dentro 3σ |
|---|---|---|---|---|
| V1_01_easy | 58,6 m | 0,29 m | 3,6° | 1,7 % |
| V1_02_medium | 75,9 m | 0,41 m | 5,0° | 2,6 % |
| V1_03_difficult | 79,0 m | 0,43 m | 3,1° | 0,3 % |

Consistenza (dopo `b3e6a4d`, `32ef9d7` e i nuovi default: σ_img = 2 px, σθ₀ = 0,05 rad, σba₀ = 0,5 m/s², allineamento del ground truth a 4 gradi di libertà):

| Sequenza | ATE RMSE | Orientazione RMSE | Dentro 3σ | NEES posizione / orientazione (consistente ≈ 3) | b_a stimato |
|---|---|---|---|---|---|
| V1_01_easy | 0,29 m | 3,0° | 67,2 % | 18 / 3357 | [−0,004, 0,142, 0,084] |
| V1_02_medium | 0,28 m | 1,0° | 30,0 % | 47 / 88 | [−0,031, 0,163, 0,065] |
| V1_03_difficult | 0,36 m | 1,4° | 18,2 % | 74 / 153 | [−0,014, 0,161, 0,069] |

Prima di queste correzioni il NEES non era misurato; con la valutazione corretta, il filtro di `2678dac` dava su V1_01 (30 s) 1292 / 4380.
Resta l'inclinazione su V1_01: il ground truth EuRoC dichiara b_a,y ≈ 0,55 m/s², mentre il filtro stima ≈ 0,14–0,16 m/s² in modo coerente su tre sequenze. La differenza (≈ 2,4°) finisce nell'inclinazione. Il bias del giroscopio invece coincide con il ground truth entro 1e-4 rad/s. Il bias dell'accelerometro nel ground truth EuRoC è una stima batch, non una misura: prima di concludere che il filtro sbaglia va verificato (ad esempio con l'inclinazione statica iniziale vs ground truth).

### 2.0 Infrastruttura di test
- [x] Aggiungere `pytest` alle dipendenze e un task `pixi run test`.
- [ ] Test unitari su `geometry.py` (`Isometry3D`, `skew`, `intersection_of_lines`, `InverseDepthPoint`).
- [x] Test dei **Jacobiani** con differenze finite (`tests/test_jacobians.py`), sui metodi veri del filtro: `compute_jacobians`, `state_augmentation`, `correct`, la transizione OC e `epipolar_residual`. Convenzione: `R_vero = R·exp(δθ)` (JPL locale).
- [x] NEES medio di posizione e orientazione nel riepilogo e nel pannello Rerun (su EuRoC, non ancora su dati sintetici).
- [x] `--no_viewer` per valutazioni veloci; ATE RMSE, consistenza 3σ e contatori degli scarti nel riepilogo finale.

### 2.1 Bug già individuati
- [x] **Test epipolare**: oltre al valore assoluto mancante, verificava `x2ᵀE x1` invece di `x1ᵀE x2`, quindi non era nullo per le associazioni corrette. Ora è la distanza di Sampson in pixel (`--epipolar_threshold`).
- [x] **Stato "null" (OC-EKF) in aliasing:** `correct()` crea oggetti nuovi invece di modificarli sul posto.
- [ ] **Random walk dei bias non accumulato** (`photorealistic_generator.py`): alla misura si somma l'incremento `*_bias_noise` invece del bias accumulato. Inoltre il rumore va scalato con `dt` (densità continua → discreta: `σ_d = σ_c / sqrt(dt)` per il rumore bianco, `σ_c · sqrt(dt)` per il random walk). Rigenerare i dataset.
- [x] **`results.txt` scritto sempre** → ora solo con `--save_results`.
- [ ] **NaN in `IMU.initialize`:** `axis` viene normalizzato prima del controllo `isclose(theta, 0/π)`.
- [x] **`time.sleep(0.005)` nel loop principale:** rimosso. Senza lo sleep il ciclo con la sola IMU gira a ~5600 Hz.
- [x] **Sincronizzazione camera/IMU:** `imu_ts >= cam_ts` (fatto con il supporto EuRoC).
- [ ] **Convenzione del segno della gravità** (emersa con EuRoC): uniformare generatore sintetico e integrazione alla convenzione fisica `f = Rᵀ(a − g)`.

### 2.2 Punti da verificare
- [x] `sigma_image` in **coordinate normalizzate** → ora in pixel (`--sigma_image`, default 1 px).
- [x] Segni e convenzioni di `compute_jacobians`: corretti. Era sbagliata la correzione in `correct()`.
- [x] Vincolo di osservabilità nella transizione: `Φ_θθ` e `u` usavano `R` invece di `Rᵀ` (notazione JPL di Hesch).
- [x] Jacobiano di `state_augmentation`, blocco posizione←δθ: `−R[t]×` invece di `[R t]×`.
- [x] Vincolo di osservabilità applicato a H_x: `H_f` va preso dal blocco modificato (`−A_p`), test `H_o·N = 0` sulla direzione di yaw.
- [x] Modello di misura delle feature: Jacobiani calcolati sul punto scalato da ρ (posizione gonfiata della profondità) e re-inizializzazione inverse depth con la profondità lungo z invece della distanza lungo il raggio.
- [x] Valutazione: errore di posizione nel frame del body confrontato con bound nel frame del mondo; allineamento del ground truth a 6 gradi di libertà che nascondeva l'errore iniziale di inclinazione → ora 4 gradi di libertà (posizione + yaw).
- [ ] **Consistenza:** con σ = 1 px il filtro è troppo sicuro di sé (solo ~8 % dei passi dentro i 3σ). Possibili cause: rumore IMU non gonfiato, errore di allineamento iniziale incluso nell'ATE, triangolazione con poca parallasse.
- [ ] `prune_poorest_camera_states` può rimuovere la camera appena clonata o quelle più recenti → escludere gli ultimi N cloni.
- [ ] `Q = Φ G Qc Gᵀ Φᵀ dt`: è un'approssimazione; valutare l'integrazione trapezoidale.
- [ ] Calcolo di RTE/ROE in `main.py`: il fallback `(1 if gt > 0 else 0)` è sempre 0 nel ramo `gt == 0`; lo "smoothing" degli outlier altera le metriche → riportare anche i valori grezzi.
- [ ] `Isometry3D` con argomenti di default mutabili (`np.eye(3)` condiviso tra istanze) e `IMU.__init__` con default `Isometry3D(...)` condivisi.
- [ ] Rimuovere il codice morto (`prune_camera_states`, la versione commentata di `prune_poorest_camera_states`, `match_frames`).

**Criterio di uscita:** tutti i test verdi, NEES nei bound χ², metriche ≥ baseline v0 → salvare la **baseline v1**.

---

## Fase 3 — Miglioramento delle prestazioni

### 3.1 Misurare prima di ottimizzare
- [ ] Profilare con `cProfile` / `py-spy` (flame graph) separando: estrazione XFeat, matching, test epipolari, triangolazione, update EKF, logging Rerun.
- [ ] Definire il target: es. ≥ 20 Hz camera e ≥ 200 Hz IMU su Mac, con le metriche invariate rispetto alla baseline v1.

### 3.2 Ottimizzazioni Python a basso rischio
- [ ] Nel loop principale, usare array NumPy al posto di `DataFrame.iloc[i]`.
- [ ] `Isometry3D.inv()` in forma chiusa (`Rᵀ`, `-Rᵀt`) invece di `np.linalg.inv` su una 4×4; precalcolare `K⁻¹` (oggi è invertita in ogni ciclo).
- [ ] Covarianza: aggiornare per blocchi (IMU-IMU, IMU-cam) senza ricreare matrici `M` complete in `state_augmentation`; `np.delete` → indicizzazione con maschere.
- [ ] Proiezione sul left null-space: QR / rotazioni di Givens invece di `scipy.linalg.null_space` (SVD).
- [ ] Vettorizzare i test epipolari/di omografia su tutte le feature e le osservazioni.
- [ ] Mappa `camera_id → indice nello stato` invece di `list(keys).index(...)` (O(n) ripetuto).
- [ ] Disattivare completamente debug, disegno e Rerun quando non servono (flag unico).

### 3.3 Front-end visivo
- [ ] XFeat su `mps` / a risoluzione ridotta; misurare l'impatto sull'accuratezza.
- [ ] Valutare `top_k` adattivo e una griglia per distribuire meglio le feature nell'immagine.
- [ ] Pipeline in un thread separato (front-end) rispetto al filtro (back-end).

### 3.4 Eventuale passaggio a codice compilato
- [ ] Se dopo 3.2–3.3 il back-end è ancora il collo di bottiglia: prototipare le funzioni critiche con Numba, poi decidere se riscrivere il core in **C++ (Eigen)** con binding `pybind11`. Questo prepara direttamente la Fase 4.

**Criterio di uscita:** target di frequenza raggiunto, metriche invariate → **baseline v2**.

---

## Fase 4 — Esecuzione su smartphone

### 4.0 Decisioni da prendere
- [ ] Piattaforma target: **iOS** (sviluppo naturale da Mac, CoreML, ARKit per confronto) o **Android** (NDK, TFLite/ONNX Runtime).
- [ ] Architettura: core C++ portabile (consigliato) + app nativa sottile (Swift / Kotlin). Python on-device non è una strada praticabile.

### 4.1 Validare prima su dati reali (ancora su Mac)
- [ ] Testare su **EuRoC MAV** (IMU reale, calibrazione nota) per uscire dai dati IMU sintetici.
- [ ] Aggiungere la stima/compensazione dell'offset temporale camera-IMU e, se serve, del rolling shutter.
- [ ] Inizializzazione robusta (statica e dinamica), stima iniziale dei bias.

### 4.2 Acquisizione dati dallo smartphone
- [ ] App minimale di registrazione: frame camera + IMU (accelerometro/giroscopio) con timestamp sullo stesso orologio.
- [ ] Calibrazione: intrinseci + distorsione (OpenCV), estrinseci camera-IMU e offset temporale (es. **Kalibr**), parametri di rumore IMU (Allan variance).
- [ ] Supporto alla distorsione della lente nel modello `Camera` (oggi è un pinhole ideale).
- [ ] Far girare la pipeline **offline sul Mac** con le registrazioni del telefono.

### 4.3 Porting del modello XFeat
- [ ] Esportare XFeat in CoreML (iOS) o TFLite / ONNX / ExecuTorch (Android); verificare che l'output coincida con PyTorch.
- [ ] Quantizzazione (FP16/INT8) e valutazione dell'impatto su matching e accuratezza.

### 4.4 Core on-device
- [ ] Portare il filtro (C++/Eigen, dalla Fase 3.4) con interfaccia `imu_callback` / `image_callback`.
- [ ] Gestire i thread (front-end, back-end), le code dei sensori e i frame persi.
- [ ] Visualizzazione minima della traiettoria nell'app; log su file per il debug.
- [ ] Misurare latenza, frequenza, consumi e temperatura; confrontare la traiettoria con ARKit/ARCore come riferimento.

**Criterio di uscita:** traiettoria stimata in tempo reale sul telefono, con errore documentato rispetto al riferimento.

---

## Convenzioni di lavoro
- Un commit per modifica logica, con messaggi descrittivi; **nessun co-author o attribuzione automatica nei commit**.
- Aggiornare le checkbox di questo file man mano che i passi vengono completati.
- Ogni fase si chiude con una baseline salvata in `benchmarks/`.
