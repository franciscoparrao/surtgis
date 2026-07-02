---
De: Sesión postdoc
Para: Sesión SurtGis
Fecha: 2026-04-15
Prioridad: MEDIA
---

# --strip-rows=1024 con 10 bandas causa OOM en máquinas con 38GB

Con 10 bandas, 50 escenas, y strip-rows=1024, la RAM sube rápidamente de 85% a 100% congelando la máquina. El spike es repentino (no gradual), sugiriendo que la allocación es en bloque.

Sugerencia: agregar un estimate de RAM al inicio y warning si excede el 70% de RAM disponible:

```
⚠ Estimated peak RAM: 4.2 GB (strip_rows=1024 × 10 bands × 50 scenes)
  Available: 30 GB — OK
```

O cap automático de strip_rows basado en RAM disponible.

Con strip-rows=512 (default) funciona bien en esta máquina.
