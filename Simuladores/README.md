# Evaluar sin tinta roja - réplica local

Esta carpeta contiene una réplica local de la app de referencia `questique-ai.lovable.app`.

- `index.html`: shell de la aplicación.
- `styles.css`: estética y componentes inspirados en la referencia.
- `app.js`: SPA con rutas, datos de demostración y persistencia en `localStorage`.

Flujo rápido:

1. Abrir `index.html`.
2. Entrar como profesor con los datos precargados.
3. Abrir el curso demo o crear uno nuevo.
4. Crear preguntas abiertas o de selección múltiple con nivel elemental, intermedio o avanzado.
5. Compartir el código `A3K-9MX` o uno generado.
6. Entrar como estudiante desde "Tengo un código de estudiante".
7. Revisar entregas, editar notas/retroalimentación y exportar resultados.

La generación de preguntas, evaluación y entregas se simulan localmente para que el prototipo funcione sin Supabase, Lovable AI ni servicios externos.
