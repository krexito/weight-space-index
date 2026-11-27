# Glosario de Términos de WSI (`WEIGHT-SPACE-INDEX`)

Guía práctica en español con los **términos canónicos en inglés** (para usar en código, papers, README o hilos de X) y su explicación directa sin humo académico.

---

## 1. Conceptos centrales de WSI y Parches

| Término canónico | Qué significa en criollo | Ejemplo / Uso |
|---|---|---|
| **WSI (Weight-Space Index)** | El toolkit entero: indexa la anatomía interna de un LLM (pesos, activaciones, gradientes) para diagnosticar y reparar fallos sin re-entrenar el modelo. | "Usamos WSI para localizar el circuito de loops en 3 minutos." |
| **MoI (Mixture of Indexes / Routed Patch)** | Un parche **ruteado**: solo se enciende en los prompts donde hace falta y queda apagado (`0` daño) en el resto. | "Con MoI arreglamos el over-refusal sin tocar GSM8K ni MMLU." |
| **`.moipatch`** | El archivo binario compacto (KB a pocos MB) que guarda las ediciones de rango bajo (`ΔW`) + el router que decide cuándo activarlas. | "Un `.moipatch` de 344 KB resolvió el over-refusal de Phi-4-mini." |
| **Always-On vs Routed** | **Always-On**: el parche está pegado siempre a los pesos (puede causar daño colateral). **Routed**: pasa por un gate/router que solo lo activa cuando detecta el patrón objetivo. | "Always-on pierde la mitad del safety; routed mantiene 97%." |
| **Bake / Hotswap** | **Bake**: fusionar el parche permanentemente en los pesos del modelo. **Hotswap**: montar/desmontar `.moipatch` en caliente sin reiniciar el servidor. | "Servimos 3 parches por hotswap en ~3 ms." |
| **Self-Gate / Veto Router** | Un router lineal de 1–2 vectores que lee el estado residual en una capa temprana y decide si **activar** o **vetar** (bloquear) el parche. | "El veto router bloquea el parche cuando detecta daño real." |
| **Sensor Token** | Una fila libre del vocabulario (`lm_head`) reutilizada como sonda lineal gratuita dentro de vLLM para leer una dirección interna como un logit más. | "El sensor token detecta loops dentro de vLLM con costo cero." |
| **Pocket MoE / Cartridge** | Pequeños adaptadores LoRA especializados (`r=2..16`, `1–10 MB`) detrás de un router por dominio para agregar skills sin olvido catastrófico. | "Cada cartridge pesa 2.5 MB y dispara 0/200 veces en MMLU." |
| **LoopRes** | Módulo residual ligero (`~47 KB`) que permite repetir un bloque de capas `T` veces (`T=2..3`) sobre un cartridge para ganar profundidad de cómputo en tiempo de inferencia. | "LoopRes `T=2` subió `+4.5 pts` en MATH-500 sobre LoRA `1x`." |

---

## 2. El Atlas de 3 Tomos (`wsi_atlas_v1`)

| Término canónico | Qué significa | Qué contiene |
|---|---|---|
| **X-Ray (`scan_model_xray`)** | Radiografía clínica de 6 etapas que corre en ~1–3 min y genera un JSON compacto (`~5–9 KB`) con la anatomía y patologías del modelo. | Pesos estáticos, forward, Fisher, DLA, refusal, sycophancy y loops. |
| **Tome-1 (Static & Clinical X-Ray)** | El catálogo de radiografías individuales de modelos base, instruct, MoE, SSM, looped, abliterated, 1-bit y GGUF (hoy **38 modelos**). | `benchmarks/xray_atlas/tome1_xray/*.xray.json` |
| **Tome-2 (Delta X-Ray)** | Comparación **antes vs después** entre dos checkpoints de la misma familia (Base → SFT → DPO → RLVR, Base → Thinking, Base → Abliterated, BF16 → GGUF). | Muestra exactamente qué capas y matrices cambió el entrenamiento o la cuantización. |
| **Tome-3 (Clinical History)** | Historial de intervenciones reales: qué parches funcionaron, cuáles fallaron y por qué (reglas empíricas). | Guía para que un agente elija capas y `alpha` sin dar palos de ciego. |
| **Blind A/B & Holdout** | Experimento donde un agente externo **sin acceso al repo** intenta arreglar un modelo reservado (**holdout**, ej. `Phi-4-mini`), comparando `wsi_only` vs `wsi_plus_atlas`. | Demostró que el Atlas sube toxic safety de `88% → 97%` con parches `2.4×` más chicos. |

---

## 3. Métricas de Anatomía Interna (qué ves en un `.xray.json`)

| Término canónico | Qué mide en simple | Cómo interpretarlo |
|---|---|---|
| **Residual Stream (`resid_norm`)** | La magnitud (norma L2) del vector que viaja de capa en capa por el tronco del transformer. | Si es chica (`5–18`, como Ouro/MemT), cualquier parche necesita `alpha` chico; si es gigante (`500–2500`, como Qwen3.8/Nemotron), tolera `alpha` más grande. |
| **Step Cosine (`resid_step_cos`)** | Qué tan parecido es el estado antes y después de una capa (`0` = giro brusco, `1` = cambio mínimo). | Un plateau alto (`0.98–0.99`) indica capas de refinamiento gradual (ideales para loopear o editar). |
| **Massive Activation (`massive_act_max`)** | Picos extremos de activación en 1–2 dimensiones del residual stream (ratio pico/mediana, ej. `533×` o `979×`). | Actúan como "anclas" estructurales del modelo. Descubrimos que **sobreviven casi intactas hasta en 1.56–2.75 bits** en 4 familias distintas. |
| **Effective Rank (`erank`)** | Qué tan distribuida está la información en los valores singulares de una matriz (`0` = rango bajo, `1.0` = rango lleno uniforme). | Modelos 1-bit/ternarios (`Bonsai`) y `Qwen3.8-27B` tienen `erank 0.93–0.95` (súper uniformes, muy buenos para cuantizar). |
| **Col Gini / Head Gini** | Desigualdad de energía entre columnas de MLP o cabezas de atención (`0` = todas iguales, `1` = una sola concentra todo). | El post-entrenamiento es sorprendentemente uniforme intra-capa (`head_gini <= 0.077`). |
| **DLA (Direct Logit Attribution)** | En qué capa específica el modelo "escribe" la respuesta factual directamente hacia el vocabulario (`lm_head`). | Casi siempre está en la cola (`depth 0.85–1.0`, ej. `L63` en Qwen3.8). **Nunca loopees ni edites fuerte la cola DLA** o rompes hechos. |
| **Empirical Fisher (`fisher_L/E`)** | Sensibilidad del modelo a perturbar cada capa (basado en gradientes al cuadrado). `late_early_ratio` compara cola vs prólogo. | Modelos híbridos SSM/DeltaNet/ShortConv tienen Fisher **ultra-prólogo** (`0.004–0.077`): el prólogo es intocable. |
| **Lowest Fisher Window** | La ventana de 3 capas con menor sensibilidad Fisher del modelo. | Es la **"zona quirúrgica segura"** donde editar o loopear causa menos daño colateral. |
| **MoE Router Entropy** | Qué tan uniformemente se reparten los tokens entre los expertos de un MoE (`1.0` = reparto perfecto, `0` = colapso a pocos expertos). | `MemT-1.0-Nano` (`0.925`) reparte mejor sus 256 expertos que `Qwen3-30B-A3B` (`0.841`). |
| **Hybrid Mixer Delta Ratio** | Relación entre cuánto cambia el residual una capa de Full Attention vs una de Linear Attention / Conv / SSM. | `1.05` en `MemT` (equilibrado 1:1); `2.34` en `Qwen3.8-27B` (Full Attn pesa más del doble). |
| **Unrolled Steps (Looped Transformer)** | Pasos efectivos cuando un modelo repite capas físicas en bucle (ej. `Ouro`: `48L × 4 = 192` pasos; `Inkling-2-Mini`: `24L → 48` pasos). | En `Inkling-2-Mini` el DLA ocurre **dentro del loop** (`step 35 / L11`), no en la última capa. |
| **mHC (Multi-Stream Hyper-Connections)** | Arquitectura que lleva múltiples flujos residuales en paralelo (`[..., S*D]`, como `SenseNova-Loop-9B`). | `wsi/xray.py` los colapsa automáticamente para medirlos en la misma escala. |

---

## 4. Patologías Clínicas y Safety

| Término canónico | Qué significa | Meta / Regla WSI |
|---|---|---|
| **Over-Refusal (`base_overref_rate`)** | El modelo rechaza preguntas **inofensivas** que solo suenan delicadas (OR-Bench-Hard). | Queremos bajarlo (`recovered >= 50%`) sin romper el rechazo real. |
| **Toxic Refusal (`base_toxic_ref_rate`)** | Rechazo a peticiones **realmente dañinas** en directo. | Regla WSI: mantener `toxic_kept >= 90%`. |
| **Adversarial Framing (`base_toxic_framed_ref_rate`)** | Lo mismo pero envuelto en excusas ("es para una novela / tesis / auditoría"). | Muchos modelos caen de `75% → 0%` bajo framing (`OLMo-2-32B`, `GLM-4-32B`); un **Framing-Invariant Veto** lo arregla. |
| **Veto / Refusal AUC (`auc_veto_peak`)** | Qué tan bien un clasificador lineal separa internamente prompts tóxicos de benignos (`1.0` = separación perfecta). | Incluso en modelos abliterados o `Ouro-Thinking` (`0%` rechazo), `auc_veto` sigue en **`1.0` desde L0**. |
| **`cos_refusal_veto`** | Similitud coseno entre la dirección de "rechazar" y la dirección de "daño real". | Cerca de `1.0` (`MemT`: `0.964`, `Qwen3.8`: `0.974`) = el modelo solo rechaza lo realmente tóxico (`0%` over-refusal). |
| **Abliteration** | Técnica popular que borra globalmente la dirección de rechazo de los pesos. | Deja `0%` rechazo pero puede costar **`-25 pts` en hechos** (`Qwen3-4B-abliterated`) y no borra el concepto interno de daño (`AUC=1.0`). |
| **Inverse Safety Restoration** | El experimento inverso al over-refusal: inyectar un `.moipatch` ruteado que **devuelve el safety** (`10% → 80%`) a un modelo que lo perdió (`Ouro-Thinking`) con `0/100` falsos positivos. | Prueba que el safety puede ser un módulo enchufable y reversible. |
| **Sycophancy (`cave_rate_false_pushback`)** | Porcentaje de veces que el modelo **cambia una respuesta correcta** solo porque el usuario le dice "¿seguro? creo que estás mal". | En `Qwen2.5-3B`, cuantizar a `Q2_K` la dispara `8×` (`11% → 90%`). |
| **Thinking Loops (`rep16_rate`)** | Cuando un modelo razonador entra en bucle repitiendo los mismos bloques de texto hasta agotar tokens. | Se mide siempre con el sampling real del modelo (greedy infla loops artificialmente). |
| **Restart Control & Answer Rescue** | **Restart Control**: cortar el loop y reanudar con el modelo base sin parche (baseline real de MoI). **Answer Rescue**: rescatar la última respuesta válida antes del loop. | Siempre reportar MoI contra **restart**, no contra base. |
| **Escape Test** | Prueba corta de continuación desde el corte de un loop/rechazo para elegir el mejor parche candidato. | Mucho más fiable que el screening de 1 forward pass (`Spearman 0.75–0.85`). |
| **Opening Entropy (`style_ok`)** | Diversidad de las primeras palabras al responder (`1.0` = natural y variado; bajo = muletillas robóticas repetidas). | Evita parches que "pasan" la métrica pero arruinan la voz del modelo. |

---

## 5. Cuantización, Compresión y GGUF

| Término canónico | Qué significa | Nota clave |
|---|---|---|
| **bpw (bits per weight)** | Promedio real de bits por parámetro incluyendo escalas y tensores en mayor precisión (`BF16 = 16`, `Q4_K_M ≈ 4.9`, `Q2_K ≈ 3.2`, `IQ1_S ≈ 1.56`). | Siempre comparar quants por `bpw` efectivo, no solo por el nombre. |
| **GGUF / `GGML_TYPE_*`** | Formato estándar de `llama.cpp` donde cada tensor puede tener su propio tipo de cuantización (`Q4_K`, `Q2_K`, `IQ1_S`, `IQ2_S`, `Q2_0 = 42`, `Q1_0 = 41`). | `wsi/gguf_loader.py` descuantiza GGUFs a `bf16` conservando el ruido exacto del quant para poder hacerles X-Ray completo. |
| **GSQ-RCO** | Cuantización mixta no-uniforme (ISTA-DASLab) que asigna bits distintos a cada tensor según sensibilidad. | En `Inkling-2-Mini` a `2.75 bpw` preservó el `87.5%` de hechos y toda la anatomía del loop. |
| **1-Bit Binary / 1.58-Bit Ternary** | Modelos entrenados nativamente con pesos `{-1, +1}` (1-bit) o `{-1, 0, +1}` (1.58-bit), como `Bonsai` o `BitNet`. | Tienen `erank ~0.95` y activaciones masivas muy bajas (`181–280×`). |
| **`q_sens` (Quant Sensitivity Map)** | Predictor de WSI (derivado del X-Ray) que estima qué capas se rompen más al cuantizar. | Correlaciona `+0.65` a `+0.82` con asignaciones óptimas tipo RCO y knapsack. |
| **Strata Quant (`S2_K` / `S3_K`) + Scale-QAT** | Esquema propio de cuantización de 2–3 bits con rotación Hadamard `H64` + Block-LDL + ajuste ligero de escalas (`+0B` de tamaño). | Rescata `Qwen2.5-3B` a `3.19b` (`GSM8K 0% → 76.5%`). |
| **Quantization X-Ray Ladder** | Escanear el mismo modelo en `BF16 → Q4 → Q2 → IQ1` para ver qué se rompe primero. | Descubrimos que la estructura física (`massive_act`, `DLA`) sobrevive hasta `1.56b`, pero el alineamiento (`toxic`, `sycophancy`) y el `fisher_L/E` se desmoronan antes. |

---

## 6. Chuleta rápida: "Cómo llamarlo en una frase"

- *"La radiografía del modelo"* → **Tome-1 X-Ray**
- *"El antes y después de entrenar o cuantizar"* → **Tome-2 Delta X-Ray**
- *"Un parche que solo se activa cuando hace falta"* → **Routed `.moipatch` (MoI)**
- *"Devolverle safety a un modelo abliterado o razonador sin over-refusal"* → **Inverse Safety Restoration**
- *"El pico gigante en una sola dimensión del residual"* → **Massive Activation**
- *"La capa donde el modelo escupe el dato factual"* → **DLA Peak**
- *"La zona segura para editar o loopear sin romper nada"* → **Lowest Fisher Window**
- *"Cuando el modelo te da la razón aunque tú estés equivocado"* → **Sycophancy (`cave_rate`)**
- *"Cuando un quant a 2 bits borra el rechazo sin querer"* → **Accidental Abliteration under Extreme Quantization**
- *"Comparar BF16 vs Q4 vs Q2 vs IQ1 del mismo modelo"* → **Quantization X-Ray Ladder**
