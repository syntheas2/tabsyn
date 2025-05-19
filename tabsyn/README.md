# Normalisierung des latenten Raums in TabSyn

- **Identifizierung des Codes:** In `tabsyn/main.py`, Zeilen 31-34
- **Befehl:** `mean, std = train_z.mean(0), train_z.std(0); train_z = (train_z - mean) / 2; train_data = train_z`

## Gründe für die Normalisierung

- **Datenzentrierung:** Entfernung des Mittelwerts (`mean`), um Daten um Null zu zentrieren
- **Teilweise Skalierung:** Division durch 2 statt durch `std` (ungewöhnliche Wahl)
- **Stabilisierung des Trainings:** Hilft dem Diffusionsmodell, konsistenter zu trainieren
- **Kontrolle der Amplituden:** Verhindert zu große Werte im latenten Raum
- **Verbesserung der Konvergenz:** Normalisierte Daten führen zu stabileren Gradienten

## Technische Aspekte

- **Reduzierte Varianz:** Im Vergleich zu vollständiger Z-Standardisierung (`/ std`) mildere Veränderung
- **Dimension-unabhängig:** `mean(0)` berechnet den Mittelwert pro Feature/Dimension
- **Konsistente Transformation:** Alle latenten Variablen werden gleich behandelt
- **Erhaltung der Form des Verteilungsbereichs:** Bewahrt Beziehungen zwischen Datenpunkten besser als volle Normalisierung

## Zweck im Diffusionsmodell

- **Optimierung für Rauschen:** Verbessert die Balance zwischen Signal und Rauschprozess
- **Diffusionseffizienz:** Normalisierte Daten vereinfachen den Diffusionsprozess
- **Kontrollierte Rauschentfernung:** Hilft dem Modell, effektiver zu lernen, Rauschen zu entfernen
- **Verbesserung der Generierungsqualität:** Führt zu realistischeren synthetischen Daten

## Unterschied zur Standard-Normalisierung

- **Partielle Normalisierung:** Division durch 2 statt durch Standardabweichung
- **Beibehaltung der relativen Skalenunterschiede:** Dimensionen mit größerer Varianz behalten proportional größere Werte
- **Bessere Performanz im Diffusionsprozess:** Empirisch bessere Ergebnisse im TabSyn-Kontext


# Unterschiede zwischen Training und Sampling im TabSyn Diffusionsmodell

## Normalisierung und Denormalisierung

- **Training:** `train_z = (train_z - mean) / 2` - Daten werden zentriert und skaliert
- **Sampling:** `x_next = x_next * 2 + mean.to(device)` - Inverse Operation zur Wiederherstellung der ursprünglichen Skala

## Datenfluss

- **Training:** Verarbeitet echte Daten aus `train_loader` nach Normalisierung
- **Sampling:** Startet mit Rauschen und erzeugt synthetische Daten mittels `sample()`-Funktion

## Modellzustand

- **Training:** Modell wird optimiert (`model.train()`) und Parameter werden aktualisiert
- **Sampling:** Modell wird nur inferiert (`model.load_state_dict()`) ohne weitere Anpassungen

## Verlustkalkulation

- **Training:** Berechnet Verlustfunktion (`loss = model(inputs)`) zur Modelloptimierung
- **Sampling:** Keine Verlustberechnung, sondern schrittweise Rauschentfernung

## Zielsetzung

- **Training:** Lernt die Umkehrung des Diffusionsprozesses (Rauschen → Original)
- **Sampling:** Wendet gelernte Umkehrung an, um aus Rauschen synthetische Daten zu erzeugen

## Nachbearbeitungsschritte

- **Training:** Speichert nur das Modell (`model_save_path`)
- **Sampling:** Komplexe Nachverarbeitung mit `split_num_cat_target()` und `recover_data()`

## Transformationsprozess

- **Training:** Einfache Datentransformation mit Zentrierung und Skalierung
- **Sampling:** Kompletter Prozess der Datengeneration mit anschließender Rücktransformation

## Technische Notwendigkeit

- **Zentrierung beim Training:** Stabilisiert Gradienten und verbessert Konvergenz
- **Denormalisierung beim Sampling:** Notwendig, um die generierten Daten auf die ursprüngliche Datenverteilung abzubilden

## Implementierungsdetails

- **Training:** Verwendet `ReduceLROnPlateau` für adaptive Lernraten
- **Sampling:** Nutzt die spezialisierte `sample()`-Funktion für den Diffusionsprozess


# Architektur und Funktionsweise des TabSyn Diffusionsmodells

## Unterscheidung zwischen `denoise_fn` und `denoise_fn_D`

- **`denoise_fn`**: Basismodell (MLPDiffusion), das die eigentliche Rauschentfernungsfunktion implementiert
- **`denoise_fn_D`**: Vorverarbeitetes (preconditioned) Modell durch Umhüllung mit der `Precond`-Klasse
- **Zweck**: Trennung zwischen dem neuronalen Netzwerk für die Rauschentfernung und der mathematischen Präkonditionierung

## Precond-Klasse: Kernfunktionalität und Zweck

- **Hauptfunktion**: Skalierung der Ein- und Ausgaben basierend auf Rauschstärke (`sigma`)
- **Implementierung**: Wandelt das Basismodell `denoise_fn_F` in ein mathematisch optimiertes Modell um
- **Mathematische Optimierung**:
  - `c_skip`: Skip-Connection-Koeffizient (direkte Durchleitung des verrauschten Inputs)
  - `c_out`: Skalierungsfaktor für die Netzwerkvorhersage
  - `c_in`: Skalierungsfaktor für den Eingang
  - `c_noise`: Logarithmische Transformation der Rauschstärke
- **Vorteil**: Verbesserte numerische Stabilität und Trainingseffizienz

## EDMLoss: Verbesserte Verlustfunktion

- **Herkunft**: "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM)
- **Kernmechanismus**: Vergleicht originale Daten `y` mit entrauschter Version `D_yn`
- **Rauschgenerierung**: 
  - `sigma = (rnd_normal * self.P_std + self.P_mean).exp()` - Log-normalverteilte Rauschstärke
  - `n = torch.randn_like(y) * sigma` - Gauß'sches Rauschen skaliert mit sigma
- **Gewichtungsfunktion**: `weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2`
  - Optimiert die Gewichtung verschiedener Rauschstärken im Trainingsprozess
  - Wichtiger für stabileres Training als einfache MSE-Verlustfunktion
- **Parameter**:
  - `P_mean`, `P_std`: Steuern die Verteilung der Rauschstärken
  - `sigma_data`: Angenommene Standardabweichung der Trainingsdaten

## MLPDiffusion: Architektur des Basismodells

- **Kernstruktur**: MLP (Multi-Layer Perceptron) mit vier Schichten
- **Dimensionalität**: Eingang `d_in` → Hidden `dim_t` → Hidden `dim_t*2` → Hidden `dim_t` → Ausgang `d_in`
- **Besonderheiten**:
  - **Positional Embedding**: `map_noise` transformiert die Rauschstärke in einen hochdimensionalen Vektor
  - **Time Embedding**: Weitere Verarbeitung des Positional Embeddings durch `time_embed`
  - **Aktivierungsfunktion**: SiLU (Sigmoid Linear Unit, auch als Swish bekannt)
- **Informationsfluss**:
  1. Noise Label wird durch Positional Embedding transformiert
  2. Embedding wird durch MLP weiterverarbeitet
  3. Input wird mit Embedding addiert (als Konditionierungssignal)
  4. Final wird die entrauschte Vorhersage ausgegeben

## Zusammenspiel der Komponenten

- **Training**: 
  1. `Model`-Klasse instanziiert `Precond` und `EDMLoss`
  2. Originaldaten werden mit randomisiertem Rauschen versehen
  3. `denoise_fn_D` versucht, originale Daten zurückzugewinnen
  4. `EDMLoss` berechnet den gewichteten Fehler
- **Sampling**:
  1. Beginnt mit reinem Rauschen
  2. Schrittweise Anwendung des trainierten `denoise_fn_D` in der `sample`-Funktion
  3. Numerische Integration einer Differentialgleichung mit Euler-Schritten

## Mehrwert der Architektur

- **Mathematische Fundierung**: Basiert auf stochastischen Differentialgleichungen
- **Stabileres Training**: Durch Präkonditionierung und spezielle Verlustfunktion
- **Flexibles Noise-Scheduling**: Erlaubt kontrollierte Übergänge zwischen Rauschen und Original
- **Bessere Generalisierung**: Lernt die Wahrscheinlichkeitsdichte der Daten effektiver als direkte Generatoren


# Die Bedeutung von Rauschstärken und Präkonditionierung in Diffusionsmodellen

## 1. Zweck verschiedener Rauschstärken und deren Gewichtung

### Grundprinzip des Diffusionsprozesses
- **Kernidee**: Training eines Modells, das den Prozess der schrittweisen Verrauschung umkehrt
- **Kontinuierliches Spektrum**: Rauschstärke σ (sigma) repräsentiert verschiedene Stadien des Diffusionsprozesses
- **Lernziel**: Das Modell muss lernen, für *jede* Rauschstärke die korrekte Entrauschung vorherzusagen

### Vorteile verschiedener Rauschstärken
- **Graduelles Lernen**: 
  - Kleine σ: Modell lernt Details und Feinheiten der Datenverteilung
  - Große σ: Modell lernt grobe Strukturen und globale Eigenschaften
- **Stabileres Training**: Verhindert Überanpassung an bestimmte Rauschlevels
- **Bessere Generalisierung**: Das Modell muss Entrauschung für alle möglichen σ-Werte beherrschen

### Gewichtungsfunktion in EDMLoss
- **Formel**: `weight = (sigma² + sigma_data²) / (sigma * sigma_data)²`
- **Mathematische Begründung**:
  - Optimale Gewichtung aus der Bayes-Formel für die Rückgewinnung des Signals
  - Abgeleitet aus der Maximierung der Log-Likelihood der Datenverteilung
- **Praktische Auswirkungen**:
  - Höhere Gewichtung für wichtigere Rauschlevels (typischerweise mittlere Levels)
  - Niedrigere Gewichtung für sehr kleine und sehr große σ-Werte
  - Verbessert Konvergenzgeschwindigkeit und Qualität der generierten Daten

## 2. Notwendigkeit und Nutzen der Precond-Skalierungen

### Mathematische Grundlage
- **Denoising Diffusion Probabilistic Model (DDPM)**:
  - Originale Formulierung führt zu numerischen Problemen bei bestimmten σ-Werten
  - Instabilität bei Training und Inferenz ohne geeignete Skalierung

### Die konkrete Implementierung in Precond
- **c_skip = sigma_data² / (sigma² + sigma_data²)**:
  - Steuert, wie stark das verrauschte Signal direkt durchgeleitet wird
  - Bei kleinem σ: Fast vollständige Durchleitung (nahe 1)
  - Bei großem σ: Minimale Durchleitung (nahe 0)
  
- **c_out = sigma * sigma_data / √(sigma² + sigma_data²)**:
  - Skalierungsfaktor für die Modellvorhersage
  - Gewährleistet, dass die Vorhersage proportional zur Datenverteilung ist
  
- **c_in = 1 / √(sigma_data² + sigma²)**:
  - Skaliert den Eingang inversionsgerecht
  - Normalisiert die Signalamplitude über alle Rauschlevels hinweg
  
- **c_noise = log(sigma) / 4**:
  - Logarithmische Transformation für stabilere numerische Eigenschaften
  - Komprimiert den großen Bereich von σ-Werten in einen handhabbareren Bereich

### Konkrete Vorteile der Skalierungen

1. **Verbesserte Trainingseffizienz**:
   - Das Modell kann sich auf das Wesentliche konzentrieren: Das Erkennen der Signal-Rausch-Beziehung
   - Reduziert die Komplexität des Lernproblems erheblich

2. **Numerische Stabilität**:
   - Verhindert Gradienten-Explosionen oder -Verschwinden
   - Besonders wichtig bei großen oder kleinen σ-Werten

3. **Verbesserte Generierungsqualität**:
   - Die geschickte Balance zwischen direkter Durchleitung und Modellvorhersage führt zu besseren Ergebnissen
   - Ermöglicht konsistente Ergebnisse über den gesamten Generierungsprozess

4. **Mathematische Optimalität**:
   - Die Skalierungen sind nicht willkürlich, sondern aus der Bayes'schen Inferenz abgeleitet
   - Minimieren den erwarteten quadratischen Fehler zwischen Vorhersage und wahren Daten

## 3. Zusammenfassung

Die Kombination aus verschiedenen Rauschstärken und den Skalierungen durch Precond bildet ein mathematisch fundiertes Framework, das:
- Den Diffusionsprozess optimal modelliert
- Die Trainingseffizienz deutlich verbessert
- Numerische Probleme elegant umgeht
- Die Qualität der generierten Daten maximiert

Diese Techniken sind ein wesentlicher Grund, warum Diffusionsmodelle wie TabSyn in der Lage sind, hochwertige synthetische Daten zu generieren, die die statistischen Eigenschaften der Originaldaten bewahren.

# Die Bayes-Begründung für die EDM-Gewichtungsformel

## Grundlagen der Bayes'schen Inferenz im Diffusionskontext

- **Bayes-Theorem**: P(x|y) ∝ P(y|x) × P(x)
  - x: Originaldaten (unbekannt, zu rekonstruieren)
  - y: Verrauschte Daten (bekannt)
  - P(x|y): Posteriori-Wahrscheinlichkeit (gesuchte Rekonstruktion)
  - P(y|x): Likelihood des Rauschprozesses
  - P(x): Prior (Annahme über die Datenverteilung)

## Herleitung der Gewichtungsformel

### Schritt 1: Modellierung des Rauschprozesses
- Diffusionsmodell: y = x + σ·ε, wobei ε ~ N(0,1) Standardnormalverteilung
- Die Likelihood ist dann: P(y|x,σ) = N(y; x, σ²)

### Schritt 2: Annahme über die Datenverteilung
- Datenverteilung: P(x) = N(x; 0, σ_data²)
- Dieses Modell nimmt an, dass die Originaldaten einer Normalverteilung mit Standardabweichung σ_data folgen

### Schritt 3: Berechnung der Posteriori
- Nach dem Bayes-Theorem und Eigenschaften der Normalverteilung:
  - P(x|y,σ) ∝ N(y; x, σ²) × N(x; 0, σ_data²)

### Schritt 4: Optimale Entrauschungsfunktion
- Die optimale Entrauschungsfunktion D*(y,σ) minimiert den erwarteten quadratischen Fehler:
  - D*(y,σ) = E[x|y,σ]
- Für Normalverteilungen ist dies eine gewichtete Summe:
  - D*(y,σ) = w·y, wobei w = σ_data²/(σ² + σ_data²)

### Schritt 5: Transformation zur EDM-Verlustfunktion
- Die EDM-Verlustfunktion trainiert das Modell, D* zu approximieren
- Die optimale Gewichtung des quadratischen Fehlers ist:
  - weight(σ) = (σ² + σ_data²)/(σ·σ_data)²

## Warum diese Formel funktioniert

1. **Optimale Signalrekonstruktion**:
   - Die Formel ist exakt die optimale Gewichtung für den Fall normalverteilter Daten und normalverteiltem Rauschen
   - Sie maximiert die Log-Likelihood der rekonstruierten Daten

2. **Ausgleich zwischen Rauschen und Signal**:
   - Bei kleinem σ (wenig Rauschen): Gewichtung ≈ 1/σ_data², fokussiert auf Originaldaten
   - Bei großem σ (viel Rauschen): Gewichtung ≈ 1/σ², berücksichtigt höhere Unsicherheit

3. **SNR-Interpretierbarkeit (Signal-zu-Rausch-Verhältnis)**:
   - Die Gewichtung kann als Funktion des SNR verstanden werden
   - Sie passt die Lernrate dynamisch an das aktuelle Rauschstadium an

4. **Mathematische Konsistenz**:
   - Stellt sicher, dass das Training über alle Rauschstärken hinweg konsistent ist
   - Führt zu einer einheitlichen Lernrate im Raum der Wahrscheinlichkeitsdichten

5. **Verbindung zur Scorefunktion**:
   - Die Gewichtung entspricht auch der optimalen Skalierung des Score Matching
   - Diese Verbindung macht die EDM-Verlustfunktion mathematisch elegant und theoretisch fundiert