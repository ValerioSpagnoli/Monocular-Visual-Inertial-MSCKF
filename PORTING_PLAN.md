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

**Nota metodologica:** il filtro è sensibile a perturbazioni minime (una rotazione iniziale diversa di 1e-16 sposta l'ATE di V1_01 a 30 s da 0,157 a 0,170 m) per via delle decisioni discrete (gating, test epipolare, pruning). Differenze sotto il ~10 % su una singola esecuzione non sono significative: le modifiche si valutano con `benchmark.py` su tutte le sequenze.

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
- [x] Test unitari su `geometry.py` (`tests/test_geometry.py`), sull'IMU (`tests/test_imu.py`) e sui generatori (`tests/test_generators.py`).
- [x] Test dei **Jacobiani** con differenze finite (`tests/test_jacobians.py`), sui metodi veri del filtro: `compute_jacobians`, `state_augmentation`, `correct`, la transizione OC e `epipolar_residual`. Convenzione: `R_vero = R·exp(δθ)` (JPL locale).
- [x] NEES medio di posizione e orientazione nel riepilogo e nel pannello Rerun (su EuRoC, non ancora su dati sintetici).
- [x] `--no_viewer` per valutazioni veloci; ATE RMSE, consistenza 3σ e contatori degli scarti nel riepilogo finale.

### 2.1 Bug già individuati
- [x] **Test epipolare**: oltre al valore assoluto mancante, verificava `x2ᵀE x1` invece di `x1ᵀE x2`, quindi non era nullo per le associazioni corrette. Ora è la distanza di Sampson in pixel (`--epipolar_threshold`).
- [x] **Stato "null" (OC-EKF) in aliasing:** `correct()` crea oggetti nuovi invece di modificarli sul posto.
- [x] **Random walk dei bias non accumulato** e rumore non discretizzato nel generatore (`9a068b1`), con test statistici.
- [x] **`results.txt` scritto sempre** → ora solo con `--save_results`.
- [x] **NaN in `IMU.initialize`** e `R = −I` (non una rotazione) nel caso capovolto → `Rotation.align_vectors` (`aa7ff15`).
- [x] **`time.sleep(0.005)` nel loop principale:** rimosso. Senza lo sleep il ciclo con la sola IMU gira a ~5600 Hz.
- [x] **Sincronizzazione camera/IMU:** `imu_ts >= cam_ts` (fatto con il supporto EuRoC).
- [x] **Convenzione del segno della gravità**: tutto alla convenzione fisica `f = Rᵀ(a − g)` (`aa7ff15`).
- [x] **Frame delle pose TartanAir**: sono nel frame NED della camera, non nel frame ottico; errore epipolare mediano 17–51 px → 0,5 px. P001 ATE (medio) 8,99 → 0,56 m (`6089b43`).
- [x] **Riuso delle misure**: le feature ancora tracciate entravano nell'update a ogni frame con tutte le osservazioni passate. Ora dopo l'update tengono solo l'ultima osservazione. Impatto trascurabile in pratica: con `min_parallax = 45°` quasi tutte le feature entrano nell'update solo quando vengono perse.

### 2.2 Punti da verificare
- [x] `sigma_image` in **coordinate normalizzate** → ora in pixel (`--sigma_image`, default 1 px).
- [x] Segni e convenzioni di `compute_jacobians`: corretti. Era sbagliata la correzione in `correct()`.
- [x] Vincolo di osservabilità nella transizione: `Φ_θθ` e `u` usavano `R` invece di `Rᵀ` (notazione JPL di Hesch).
- [x] Jacobiano di `state_augmentation`, blocco posizione←δθ: `−R[t]×` invece di `[R t]×`.
- [x] Vincolo di osservabilità applicato a H_x: `H_f` va preso dal blocco modificato (`−A_p`), test `H_o·N = 0` sulla direzione di yaw.
- [x] Modello di misura delle feature: Jacobiani calcolati sul punto scalato da ρ (posizione gonfiata della profondità) e re-inizializzazione inverse depth con la profondità lungo z invece della distanza lungo il raggio.
- [x] Valutazione: errore di posizione nel frame del body confrontato con bound nel frame del mondo; allineamento del ground truth a 6 gradi di libertà che nascondeva l'errore iniziale di inclinazione → ora 4 gradi di libertà (posizione + yaw).
- [x] **Consistenza della propagazione:** il generatore sintetico non era integrabile esattamente (interpolazione lineare → salti di velocità). Ora (`e2eb507`) PCHIP + slerp e misure costruite per l'integratore del filtro. Con la sola IMU su P001 il NEES è ≈ 3–4 (prima 297 / 46).
- [x] **Consistenza degli update visivi** (`18e5af6`): triangolazione Gauss-Newton in inverse depth sulle pose correnti dei cloni (prima: punto medio dei raggi con le pose al momento dell'osservazione, e feature usate anche se la triangolazione falliva). Su P001 NEES ≈ 2,3–2,8 / 3,8–4,0 a tutti i livelli di rumore, 100 % dentro i 3σ, ATE 0,08–0,13 m.
- [x] **Zero-velocity update** (`18e5af6`): da fermi nessuna feature ha parallasse, quindi il filtro derivava con la sola IMU (fino a 6 m su V1_03). ZUPT quando la disparità mediana è < 1 px **e** v = 0 passa un test χ² (la sola disparità scatta anche avanzando verso punti lontani).
- [ ] **Inclinazione/bias su EuRoC V1_01:** il ground truth ha b_a,y ≈ 0,55 m/s² (confermato dalla fase statica: 0,48), il filtro stima 0,14. Con moto quasi solo in yaw inclinazione e bias orizzontale sono poco separabili, ma il filtro si dichiara sicuro (osservabilità spuria non coperta dall'OC).
- [ ] `prune_poorest_camera_states` può rimuovere la camera appena clonata o quelle più recenti → escludere gli ultimi N cloni.
- [ ] `Q = Φ G Qc Gᵀ Φᵀ dt`: è un'approssimazione; valutare l'integrazione trapezoidale.
- [ ] Calcolo di RTE/ROE in `main.py`: il fallback `(1 if gt > 0 else 0)` è sempre 0 nel ramo `gt == 0`; lo "smoothing" degli outlier altera le metriche → riportare anche i valori grezzi.
- [x] Argomenti di default mutabili in `Isometry3D`, `IMU`, `InverseDepthPoint` (`da8f085`).
- [x] Rimosso il codice morto e il disegno di debug ancora calcolato a ogni frame.

**Criterio di uscita:** tutti i test verdi, NEES nei bound χ², metriche ≥ baseline v0 → salvare la **baseline v1**.

---

## Fase 3 — Miglioramento delle prestazioni

Riferimento: EuRoC V1_01_easy, primi 30 s, `--no_viewer`, Mac Apple Silicon, XFeat su MPS. Il **fattore di tempo reale** è il tempo di elaborazione diviso per la durata dei dati (< 1 = più veloce del tempo reale).

| Commit | Modifica | Frame con camera | Passo IMU | Fattore tempo reale |
|---|---|---|---|---|
| — | dopo la fase 2 | 80 ms | 0,18 ms | ≈ 1,6 |
| `1015137` | `Isometry3D` in forma chiusa, `K⁻¹` in cache | 70 ms | | |
| `4d33902` | geometria a due viste una volta per frame nei test delle associazioni | 56 ms | | |
| `4bfcd46` | null space con QR, soglia χ² in cache, `solve` nel gating | 54 ms | | |
| `d5c75b6` | front-end (lettura, undistorsione, XFeat) in un thread; array NumPy nel loop | **45 ms** | **0,12 ms** | **0,93** |

Sequenze complete: V1_02 0,77, TartanAir P001 0,80. Risultati invariati (differenze solo numeriche, ≤ 1 %).

### 3.1 Misurare prima di ottimizzare
- [x] Profilare con `cProfile` (vedi tabella). XFeat su MPS: ~15 ms la rete + ~5 ms la NMS a 752×480; il resto è il filtro in Python.
- [x] Target: tempo reale su EuRoC (camera 20 Hz, IMU 200 Hz) → raggiunto.

### 3.2 Ottimizzazioni Python a basso rischio
- [x] Nel loop principale, array NumPy al posto di `DataFrame.iloc[i]`.
- [x] `Isometry3D.inv()` e `__mul__` in forma chiusa; `K⁻¹` precalcolata.
- [ ] Covarianza: aggiornare per blocchi senza ricreare matrici `M` complete in `state_augmentation`; `np.delete` → indicizzazione con maschere.
- [x] Proiezione sul left null-space con QR invece di `scipy.linalg.null_space` (SVD).
- [x] Test epipolari/omografia: geometria a due viste precalcolata per clone. Vettorizzazione completa ancora da fare (~4–5 ms a frame).
- [ ] Mappa `camera_id → indice nello stato` invece di `list(keys).index(...)` (O(n) ripetuto).
- [x] Disegno di debug rimosso; `--no_viewer` disattiva Rerun.
- [ ] Parallasse in `get_valid_features`: `angle_between_directions` chiamata ~290 volte a frame (~3 ms).

### 3.3 Front-end visivo
- [x] XFeat su `mps` (`--device`, fase 1.2).
- [ ] Risoluzione ridotta, `top_k` adattivo e griglia: da valutare insieme alla fase 4 (sul telefono XFeat girerà con CoreML/TFLite).
- [x] Pipeline in un thread separato (front-end) rispetto al filtro (back-end).

### 3.4 Eventuale passaggio a codice compilato
- [ ] Il filtro in Python costa ancora ~30 ms a frame, distribuiti su molte voci piccole. Il guadagno successivo viene dal core in **C++ (Eigen)**, che serve comunque per lo smartphone (Fase 4). I test numerici in `tests/` (Jacobiani, IMU, proiezione, generatori) possono diventare la specifica della versione C++.

**Criterio di uscita:** target di frequenza raggiunto, metriche invariate → **baseline v2**.

---

## Fase 3b — Progetto solo C++ (completata)

Il codice Python è stato rimosso (`44d54d2`); resta su `main` e nella cronologia. Struttura attuale:
- **Backend** (`backend/`, solo Eigen): MSCKF, triangolazione Gauss-Newton, ZUPT, loop closure con grafo delle pose a 4 gradi di libertà, `VioSystem`. Cross-compila con l'NDK (`pixi run build-android`).
- **Frontend** (`frontend/`, OpenCV + ONNX Runtime): XFeat da `models/xfeat.onnx`, stessi keypoint di PyTorch (descrittori entro 5e-6), 17 ms per immagine su CPU.
- **Applicazione** (`app/`): `msckf_run` (EuRoC, valutazione, Rerun). `pixi run build`, `pixi run main`, `pixi run benchmark`, `pixi run test`.
- Strumenti in `tools/`, submodule XFeat in `third_party/`. TartanAir/PeringLab e il generatore IMU rimossi (`aad5f7f`): solo EuRoC.
- Export di XFeat: ambiente separato `pixi run -e export export-xfeat`.
- Non portata: la sorgente completamente sintetica (`synthetic_generator.py`).

Risultati (`bc79637`, ATE RMSE odometria → con loop closure, NEES posizione / orientazione, fattore di tempo reale):

| Sequenza | ATE | Con loop closure | NEES | Tempo reale |
|---|---|---|---|---|
| V1_01_easy | 0,310 m | 0,299 m | 3,8 / 265 | 0,44 |
| V1_02_medium | 0,205 m | 0,195 m | 4,2 / 2,8 | 0,47 |
| V1_03_difficult | 0,290 m | 0,286 m | 7,2 / 8,9 | 0,46 |
| P001 (medio) | 0,134 m | — | 1,4 / 1,6 | 0,42 |

Altre correzioni della fase: covarianza iniziale dell'inclinazione dalla stima della gravità (prima il NEES con sola IMU era 9–52 a seconda del seme, ora 0,6–4,5); rumore IMU reale ×10 sul datasheet; bug dell'omografia (direzione e confronti invertiti).

Da fare per la loop closure: descrittore globale discriminante (oggi la media dei descrittori XFeat ha somiglianza ~0,99 tra tutte le immagini), taratura delle incertezze del grafo, sequenze con traiettorie più lunghe (EuRoC Machine Hall).

---

### Struttura (`b56d769`) e prova del tracking nel frontend
- Repository riorganizzato in `backend/`, `frontend/` e `app/` (ognuno con `include/` e `src/`, e test dove presenti), più `tools/` e `third_party/`.
- Provato a spostare il tracking nel frontend (`fbb940b`), con RANSAC della matrice fondamentale tra immagini consecutive al posto dei test epipolare/omografia sulle pose del filtro: ATE medio EuRoC da 0,268 a 0,287 m. Annullato: il tracking resta nel backend, dove ogni associazione viene verificata contro tutte le osservazioni della traccia con le pose del filtro.

---

### Confronto delle feature (`--features`)
EuRoC V1_01–03, 300 keypoint, soglia di matching tarata per ciascuna senza loop closure; media delle tre sequenze:

| Feature | ATE | ATE con loop closure | Frequenza camera (Mac) |
|---|---|---|---|
| XFeat | 0,268 m | 0,191 m | 43 Hz |
| SIFT (RootSIFT) | 0,242 m | 0,199 m | 25 Hz |
| ORB | 0,268 m | 0,224 m | 98 Hz |
| AKAZE | 0,268 m | 0,205 m | 30 Hz |

L'odometria è alla pari (differenze dentro la variabilità del filtro, ~8 %): la verifica epipolare con le pose del filtro compensa descrittori meno robusti. ORB è 2,3 volte più veloce di XFeat ed è il candidato per lo smartphone; perde nella loop closure.
Tuning della loop closure per ORB (similarità dei match 0,6, 10 inlier PnP): 0,239 → 0,224 m. La soglia sulla similarità globale non conta (si verificano comunque i 5 candidati migliori) e verificarne 10 o 20 trova più loop senza ridurre l'ATE. Il limite è la qualità dei loop, soprattutto su V1_01 (0,344 → 0,332 m con 12 loop, contro 0,310 → 0,251 m di XFeat): per andare oltre serve un place recognition migliore (bag of words) o la selezione dei candidati vicini alla posa stimata. LiftFeat (ICRA 2025): esportazione pronta, da valutare.

---

## Fase 4 — Esecuzione su smartphone

### 4.0 Decisioni da prendere
- [x] Piattaforma target: **Android** (NDK, TFLite/ONNX Runtime per XFeat).
- [x] Architettura: core C++ portabile (`cpp/`, solo Eigen) + app nativa sottile (Kotlin/JNI).

### 4.1 Validare prima su dati reali (ancora su Mac)
- [x] Testare su **EuRoC MAV** (IMU reale, calibrazione nota) per uscire dai dati IMU sintetici.
- [ ] Aggiungere la stima/compensazione dell'offset temporale camera-IMU e, se serve, del rolling shutter.
- [x] Inizializzazione statica con stima del bias del giroscopio e ZUPT. Manca l'inizializzazione dinamica (partenza in movimento).

### 4.2 Acquisizione dati dallo smartphone
- [ ] App minimale di registrazione: frame camera + IMU (accelerometro/giroscopio) con timestamp sullo stesso orologio.
- [ ] Calibrazione: intrinseci + distorsione (OpenCV), estrinseci camera-IMU e offset temporale (es. **Kalibr**), parametri di rumore IMU (Allan variance).
- [ ] Supporto alla distorsione della lente nel modello `Camera` (oggi è un pinhole ideale).
- [ ] Far girare la pipeline **offline sul Mac** con le registrazioni del telefono.

### 4.3 Porting del modello XFeat
- [ ] Esportare XFeat in CoreML (iOS) o TFLite / ONNX / ExecuTorch (Android); verificare che l'output coincida con PyTorch.
- [ ] Quantizzazione (FP16/INT8) e valutazione dell'impatto su matching e accuratezza.

### 4.4 Core on-device
- [x] Filtro portato in C++17/Eigen (`ce8c152`, `cpp/`): `Msckf::addImu` / `Msckf::addImage(keypoints, descriptors, scores)`. Parità con il Python ~1e-12 sullo scenario sintetico (`tests/test_cpp_parity.py`), unit test Catch2, binding pybind11 (`--backend cpp`), cross-compilazione con `pixi run build-android`.
  - EuRoC V1_01 30 s: frame con camera 50 → 22 ms (ormai in attesa di XFeat), passo IMU 106 → 35 µs, fattore di tempo reale 1,01 → 0,45.
  - [ ] Installare l'NDK (Android Studio → SDK Tools → NDK) e verificare `pixi run build-android`.
  - [ ] Layer JNI e app Kotlin minima.
- [ ] Gestire i thread (front-end, back-end), le code dei sensori e i frame persi.
- [ ] Visualizzazione minima della traiettoria nell'app; log su file per il debug.
- [ ] Misurare latenza, frequenza, consumi e temperatura; confrontare la traiettoria con ARKit/ARCore come riferimento.

**Criterio di uscita:** traiettoria stimata in tempo reale sul telefono, con errore documentato rispetto al riferimento.

---

## Convenzioni di lavoro
- Un commit per modifica logica, con messaggi descrittivi; **nessun co-author o attribuzione automatica nei commit**.
- Aggiornare le checkbox di questo file man mano che i passi vengono completati.
- Ogni fase si chiude con una baseline salvata in `benchmarks/`.
