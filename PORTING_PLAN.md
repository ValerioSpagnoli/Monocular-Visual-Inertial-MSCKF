# Piano di porting: dal prototipo di tesi allo smartphone

Branch di lavoro: `porting` (creato da `main` @ `4a5497f`).

Obiettivo complessivo: portare il MSCKF monoculare da prototipo Python (Linux) a un sistema che gira in tempo reale su uno smartphone, passando per macOS, correzione dei bug e ottimizzazione.

Le fasi vanno svolte **in ordine**: ogni fase produce una baseline misurabile che la fase successiva non deve peggiorare.

---

## Fase 1 — Porting su macOS (Apple Silicon, `osx-arm64`)

### 1.1 Ambiente Pixi
- [ ] Aggiungere `osx-arm64` a `platforms` in `pixi.toml` (mantenere `linux-64`).
- [ ] Rimuovere il canale `pytorch` (deprecato) e prendere `pytorch` da `conda-forge`.
- [ ] Spostare `cpuonly` in una sezione `[target.linux-64.dependencies]` (non esiste su macOS).
- [ ] Rigenerare `pixi.lock` con `pixi install` e verificare che risolva su entrambe le piattaforme.
- [ ] Eseguire `pixi run verify-installation` e `pixi run test-imports`.

### 1.2 Submodule e XFeat
- [ ] Verificare che `src/accelerated_features` sia inizializzato (`git submodule update --init`) e che i pesi in `weights/` siano presenti (Git LFS?).
- [ ] Controllare come XFeat sceglie il device (`cuda` → fallback `cpu`); valutare l'opzione `mps` (GPU Apple) in modo configurabile, con fallback su CPU.
- [ ] Sostituire l'hack `sys.path.insert` in `FeatureExtractor.py` e `sys.path.append('../../')` in `main.py` con import puliti (il `PYTHONPATH` di Pixi è già impostato).

### 1.3 Dataset e prima esecuzione
- [ ] Scaricare una sequenza TartanAir piccola (es. `P001`) in `data/tartanair/` con il relativo `camera_info.csv`.
- [ ] Verificare che Rerun (`rr.spawn()`) apra il viewer su macOS.
- [ ] Eseguire `pixi run main --source tartanair --sequence P001 --max_frames 500 --noise_level mid --save_results`.

### 1.4 Baseline di riferimento (prima di toccare l'algoritmo)
- [ ] Salvare `results.txt`, i plot e i parametri usati come **baseline v0** (es. in `benchmarks/baseline_v0/`), per almeno 2 sequenze × 3 livelli di rumore.
- [ ] Fissare i seed (`np.random.seed`, `torch.manual_seed`) per avere run riproducibili.
- [ ] Scrivere uno script `benchmark.py` (o un task Pixi `pixi run bench`) che lanci in batch le configurazioni e stampi una tabella RTE/ROE/ATE/tempi.

**Criterio di uscita:** il codice gira su Mac con risultati comparabili a quelli della tesi.

---

## Fase 2 — Correzione bug

Per ogni bug: test che lo riproduce → fix → confronto con la baseline. Un commit per bug.

### 2.0 Infrastruttura di test
- [ ] Aggiungere `pytest` alle dipendenze e un task `pixi run test`.
- [ ] Test unitari su `geometry.py` (`Isometry3D`, `skew`, `intersection_of_lines`, `InverseDepthPoint`).
- [ ] Test dei **Jacobiani** (`Camera.compute_jacobians`, matrice F in `process_imu`, J in `state_augmentation`) confrontandoli con differenze finite numeriche.
- [ ] Test di consistenza: NEES medio su una sequenza sintetica senza rumore visivo.

### 2.1 Bug già individuati
- [ ] **Test epipolare senza valore assoluto** (`MSCKF.add_camera_measurements`): `score > threshold` lascia sempre passare i residui negativi → usare `abs(score)`. Valutare anche la distanza di Sampson in pixel al posto del residuo algebrico, così la soglia ha un'unità fisica.
- [ ] **Stato "null" (OC-EKF) in aliasing:** `Camera.T_W_Ci_null = T_W_Ci` e `imu.T_W_Ii_null = imu.T_W_Ii` puntano allo stesso oggetto, che `correct()` modifica sul posto → fare copie profonde e/o rendere `Isometry3D` immutabile.
- [ ] **Random walk dei bias non accumulato** (`photorealistic_generator.py`): alla misura si somma l'incremento `*_bias_noise` invece del bias accumulato. Inoltre il rumore va scalato con `dt` (densità continua → discreta: `σ_d = σ_c / sqrt(dt)` per il rumore bianco, `σ_c · sqrt(dt)` per il random walk). Rigenerare i dataset.
- [ ] **`results.txt` scritto sempre** (`main.py`), anche senza `--save_results` → crash se la cartella non esiste.
- [ ] **NaN in `IMU.initialize`:** `axis` viene normalizzato prima del controllo `isclose(theta, 0/π)`.
- [ ] **`time.sleep(0.005)` nel loop principale:** falsa il profiling (il valore "171 Hz senza camera" ≈ 1/5 ms è dominato dallo sleep). Rimuoverlo o escluderlo dalle misure.
- [ ] **Sincronizzazione camera/IMU per uguaglianza esatta dei timestamp** (`abs(round(...)) < 1e-5`): funziona solo sui dati generati. Sostituirla con un buffer basato sul tempo (prerequisito per i dati reali).

### 2.2 Punti da verificare
- [ ] `sigma_image` è applicato a residui in **coordinate normalizzate** (`K⁻¹·x`): 0.1 corrisponde a ~32 px con f=320. Verificare che sia voluto; esprimerlo in pixel e convertirlo (`σ_px / f`).
- [ ] Segni e convenzioni di `compute_jacobians` (H_x rispetto a δθ e δp della camera) coerenti con la correzione `R_W_Ci = R_W_estCi @ R_Ci_estCi.T`.
- [ ] Il vincolo di osservabilità applicato anche a H_x (`compute_residual_and_jacobians`): controllare la formula di `u` e che usi lo stato null corretto (dipende dal fix dell'aliasing).
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
