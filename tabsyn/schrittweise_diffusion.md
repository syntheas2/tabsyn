# Wichtige Aspekte des Diffusionsmodells in TabSyn

## Training des Modells

- **Rauschvorhersage statt Zeit**: Das Modell verwendet `noise_labels` statt expliziter Zeitschritte
  ```python
  def forward(self, x, noise_labels, class_labels=None):
      emb = self.map_noise(noise_labels)  # Wandelt Rauschlevel in Embedding um
  ```

- **EDMLoss**: Trainiert das Modell, direkt von verrauschten zu sauberen Daten zu gehen
  ```python
  n = torch.randn_like(y) * sigma.unsqueeze(1)  # Rauschen hinzufügen
  D_yn = denoise_fn(y + n, sigma)  # Vorhersage der Originaldaten
  loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)  # Gewichteter MSE
  ```

- **Gewichtung**: Optimiert Training für verschiedene Rauschniveaus
  ```python
  weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
  ```

## Sampling-Prozess

- **Schrittweise Entrauschung**: Simulation eines kontinuierlichen Prozesses
  ```python
  denoised = net(x_hat, t_hat).to(torch.float32)
  d_cur = (x_hat - denoised) / t_hat
  x_next = x_hat + (t_next - t_hat) * d_cur
  ```

- **Numerische Integration**: Euler-Verfahren mit 2nd-Order Korrektur
  ```python
  # 2nd Order Korrektur
  denoised = net(x_next, t_next).to(torch.float32)
  d_prime = (x_next - denoised) / t_next
  x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
  ```

- **Mehrere Iterationen**: `num_steps` Schritte für stabiles, qualitatives Sampling

## Warum so implementiert

- **Mathematische Korrektheit**: Löst zugrundeliegende stochastische Differentialgleichungen
- **Numerische Stabilität**: Kleine Schritte vermeiden kumulative Fehler
- **Flexibilität**: Modell lernt Entrauschung für beliebige Rauschniveaus
- **Hohe Qualität**: Bessere Ergebnisse durch präzisere Simulation des Entrauschungspfads

## Zusammenhang Rauschen und Zeit

- **"Time Embedding"**: Eigentlich ein Embedding des Rauschlevels
  ```python
  emb = self.time_embed(emb)  # Name ist historisch bedingt
  ```

- **Kontinuierliches Spektrum**: Rauschstärke σ repräsentiert verschiedene Stadien des Diffusionsprozesses

- **Trainings-Sampling-Unterschied**: 
  - Training: Ein-Schritt-Vorhersage mit zufälligen Rauschniveaus
  - Sampling: Schrittweise Integration von hohem zu niedrigem Rauschen