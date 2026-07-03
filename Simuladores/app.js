const app = document.querySelector("#app");

const competencies = [
  "Comprensión",
  "Análisis",
  "Aplicación",
  "Evaluación crítica",
  "Síntesis",
];

const seedQuestions = [
  "Explica con tus palabras la idea central del material y menciona dos evidencias que la sostienen.",
  "Analiza cómo se relacionan los conceptos principales y señala una tensión o contraste relevante.",
  "Propón una aplicación concreta del contenido en un contexto académico, profesional o cotidiano.",
  "Evalúa críticamente una afirmación importante del material: ¿qué tan sólida es y qué límites tiene?",
  "Sintetiza una conclusión propia que conecte el material con una pregunta nueva de investigación o discusión.",
];

const mcqQuestionTemplates = [
  "¿Cuál es la idea central del material?",
  "¿Qué evidencia apoya mejor el argumento principal?",
  "¿Qué relación entre conceptos aparece con más fuerza?",
  "¿Cuál sería una aplicación adecuada del contenido?",
  "¿Qué límite o tensión conviene reconocer?",
  "¿Qué ejemplo representa mejor el enfoque del material?",
  "¿Qué afirmación contradice menos el contenido?",
  "¿Qué criterio serviría para evaluar una respuesta sólida?",
  "¿Qué conclusión sintetiza mejor el material?",
  "¿Qué pregunta nueva se desprende del contenido?",
];

const defaultState = {
  session: null,
  courses: [
    {
      id: "course-demo",
      name: "Sociología 2026-A",
      description: "Curso de demostración con actividad lista para estudiantes.",
      createdAt: new Date().toISOString(),
      materials: [
        {
          id: "mat-demo",
          kind: "pdf",
          mode: "open",
          level: "intermedio",
          title: "Capítulo 3 - Pensamiento crítico",
          sourceUrl: "",
          sourceText:
            "Material de ejemplo sobre pensamiento crítico, comprensión, análisis, aplicación, evaluación y síntesis.",
          createdAt: new Date().toISOString(),
          sharedCode: "A3K-9MX",
          individualCodes: [],
          questions: seedQuestions.map((text, index) => ({
            id: `q-demo-${index + 1}`,
            position: index + 1,
            text,
            competency: competencies[index],
          })),
          submissions: [],
        },
      ],
    },
  ],
};

let state = loadState();
let studentDraft = null;
const BRAND_NAME = "Evaluar sin tinta roja";

function loadState() {
  try {
    const raw = localStorage.getItem("questique-clone-state");
    return raw ? JSON.parse(raw) : structuredClone(defaultState);
  } catch {
    return structuredClone(defaultState);
  }
}

function saveState() {
  localStorage.setItem("questique-clone-state", JSON.stringify(state));
}

function route() {
  const hash = location.hash.replace(/^#/, "") || "/";
  return hash.startsWith("/") ? hash : `/${hash}`;
}

function navigate(to) {
  location.hash = to;
}

window.addEventListener("hashchange", render);

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatDate(value) {
  return new Date(value).toLocaleDateString();
}

function formatDateTime(value) {
  return new Date(value).toLocaleString();
}

function id(prefix) {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

function makeCode() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let value = "";
  for (let i = 0; i < 7; i += 1) {
    if (i === 3) value += "-";
    value += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return value;
}

function getCourse(courseId) {
  return state.courses.find((course) => course.id === courseId);
}

function getMaterial(courseId, materialId) {
  const course = getCourse(courseId);
  return course?.materials.find((material) => material.id === materialId);
}

function findByCode(code) {
  const normalized = String(code || "").trim().toUpperCase();
  for (const course of state.courses) {
    for (const material of course.materials) {
      if (material.sharedCode === normalized) {
        return { course, material, code: normalized, shared: true };
      }
      const individual = material.individualCodes.find((item) => item.code === normalized);
      if (individual) {
        return { course, material, code: normalized, shared: false, individual };
      }
    }
  }
  return null;
}

function logo(small = false) {
  return `<div class="brand-mark ${small ? "small" : ""}"><span class="font-serif">A</span></div>`;
}

function publicHeader() {
  return `
    <header class="topbar">
      <div class="container-page">
        <a href="#/" class="brand">
          ${logo()}
          <div class="truncate">
            <div class="brand-title">${BRAND_NAME}</div>
            <div class="brand-subtitle">Preguntas abiertas y evaluación automática con IA local</div>
          </div>
        </a>
        <a href="#/auth" class="link-button">Entrar / Crear cuenta</a>
      </div>
    </header>
  `;
}

function teacherHeader() {
  const email = state.session?.email || "profesor@demo.edu";
  return `
    <header class="topbar">
      <div class="container-page">
        <a href="#/dashboard" class="brand">
          ${logo(true)}
          <div>
            <div class="brand-title">${BRAND_NAME}</div>
            <div class="brand-subtitle">Panel del profesor</div>
          </div>
        </a>
        <div class="small-actions">
          <span class="muted text-sm hide-mobile">${escapeHtml(email)}</span>
          <button class="link-button" data-action="logout">Salir</button>
        </div>
      </div>
    </header>
  `;
}

function studentHeader(code = "") {
  return `
    <header class="topbar">
      <div class="container-page">
        <a href="#/" class="brand">
          ${logo(true)}
          <span class="brand-title">${BRAND_NAME}</span>
        </a>
        ${code ? `<span class="chip mono">${escapeHtml(code)}</span>` : ""}
      </div>
    </header>
  `;
}

function renderHome() {
  app.innerHTML = `
    <div class="page">
      ${publicHeader()}
      <main class="container-page">
        <section class="hero">
          <span class="chip">Versión MVP · rúbrica académica</span>
          <h1>Evalúa con preguntas<span class="mobile-break"><br></span> abiertas,<span class="primary-text">sin corregir a mano.</span></h1>
          <p>Sube un PDF o pega la transcripción de un video de YouTube. La IA genera 5 preguntas abiertas, los estudiantes responden con un código único, y tú recibes notas y retroalimentación en un panel editable.</p>
          <div class="hero-actions">
            <a href="#/auth" class="primary-button">Soy profesor — empezar</a>
            <a href="#/join" class="secondary-button">Tengo un código de estudiante</a>
          </div>
        </section>

        <section class="feature-grid">
          ${featureCard("Material PDF o YouTube", "Sube un PDF o pega la transcripción de un video. La IA usa el contenido como fuente de verdad.")}
          ${featureCard("5 preguntas por competencia", "Comprensión, análisis, aplicación, evaluación crítica y síntesis. Sin opción múltiple.")}
          ${featureCard("Rúbrica de 5 criterios", "Calificación 0.0-5.0 con fortalezas, mejoras y feedback editable por el profesor.")}
        </section>

        <section class="surface-card how-card">
          <div class="feature-title">Cómo funciona</div>
          <p class="leading-relaxed">La generación de preguntas, la lectura de los PDFs, la descarga de subtítulos de YouTube y la evaluación de respuestas las hace Lovable AI en el servidor. Funciona en cualquier navegador moderno (computadora o móvil). El profesor puede ajustar las notas después.</p>
        </section>
      </main>
      <footer class="footer">
        <div class="container-page">
          <span>Hecho para docentes</span>
          <span>Lovable AI · TanStack Start</span>
        </div>
      </footer>
    </div>
  `;
}

function featureCard(title, text) {
  return `
    <article class="surface-card card-pad">
      <div class="feature-title">${title}</div>
      <p class="muted text-sm leading-relaxed">${text}</p>
    </article>
  `;
}

function renderAuth() {
  const mode = sessionStorage.getItem("auth-mode") || "login";
  const isLogin = mode === "login";
  app.innerHTML = `
    <div class="center-shell">
      <div class="surface-card auth-card">
        <a href="#/" class="back-link">← Volver al inicio</a>
        <h1 class="page-title">${isLogin ? "Entrar" : "Crear cuenta de profesor"}</h1>
        <p class="text-sm muted">${isLogin ? "Accede a tus cursos y resultados." : "Crea tu cuenta para gestionar cursos y estudiantes."}</p>
        <form class="form-stack" data-form="auth">
          ${
            isLogin
              ? ""
              : `<div class="field"><label>Nombre</label><input class="input" name="name" required value="Profesor Demo"></div>`
          }
          <div class="field"><label>Correo</label><input class="input" type="email" name="email" required value="profesor@demo.edu"></div>
          <div class="field"><label>Contraseña</label><input class="input" type="password" name="password" required minlength="6" value="123456"></div>
          <button class="primary-button full" type="submit">${isLogin ? "Entrar" : "Crear cuenta"}</button>
        </form>
        <button class="danger-link full" data-action="toggle-auth">${isLogin ? "¿Nuevo? Crear cuenta" : "¿Ya tienes cuenta? Entrar"}</button>
      </div>
    </div>
  `;
}

function requireTeacher() {
  if (!state.session) {
    navigate("/auth");
    return false;
  }
  return true;
}

function renderDashboard() {
  if (!requireTeacher()) return;
  app.innerHTML = `
    <div class="page">
      ${teacherHeader()}
      <main class="container-page dashboard-main section-stack">
        <section class="section-heading">
          <span class="chip">Profesor</span>
          <h1>Mis cursos</h1>
          <p class="text-sm muted">Cada curso agrupa materiales, preguntas, códigos de estudiante y resultados.</p>
        </section>

        <section class="surface-card card-pad">
          <h2 class="feature-title">Crear nuevo curso</h2>
          <form class="create-grid" data-form="course">
            <div class="field">
              <label class="tiny-label">Nombre</label>
              <input class="input" name="name" placeholder="Ej. Sociología 2026-A" required>
            </div>
            <div class="field">
              <label class="tiny-label">Descripción (opcional)</label>
              <input class="input" name="description" placeholder="Breve descripción">
            </div>
            <button class="primary-button" type="submit">Crear curso</button>
          </form>
        </section>

        <section>
          ${
            state.courses.length === 0
              ? `<div class="surface-card empty-card text-sm muted">Aún no tienes cursos. Crea el primero arriba.</div>`
              : `<div class="course-grid">${state.courses.map(courseCard).join("")}</div>`
          }
        </section>
      </main>
    </div>
  `;
}

function courseCard(course) {
  return `
    <article class="surface-card card-pad">
      <div class="row-between">
        <div class="truncate">
          <div class="feature-title truncate">${escapeHtml(course.name)}</div>
          ${course.description ? `<p class="text-sm muted">${escapeHtml(course.description)}</p>` : ""}
          <div class="text-xs muted">Creado ${formatDate(course.createdAt)}</div>
        </div>
        <button class="danger-link" data-action="delete-course" data-course="${course.id}">Eliminar</button>
      </div>
      <a href="#/courses/${course.id}" class="primary-button" style="margin-top:1rem">Abrir curso →</a>
    </article>
  `;
}

function renderCourse(courseId, selectedMaterialId) {
  if (!requireTeacher()) return;
  const course = getCourse(courseId);
  if (!course) {
    app.innerHTML = `${teacherHeader()}<main class="container-page dashboard-main text-sm">Curso no encontrado.</main>`;
    return;
  }
  const material = getMaterial(courseId, selectedMaterialId) || course.materials[0] || null;
  app.innerHTML = `
    <div class="page">
      ${teacherHeader()}
      <main class="container-page dashboard-main section-stack">
        <section class="section-heading">
          <a href="#/dashboard" class="back-link">← Mis cursos</a>
          <h1>${escapeHtml(course.name)}</h1>
          ${course.description ? `<p class="text-sm muted">${escapeHtml(course.description)}</p>` : ""}
        </section>
        ${materialForm(course.id)}
        ${
          course.materials.length
            ? `<section><div class="material-type" style="margin-bottom:.5rem">Materiales</div><div class="pill-row">${course.materials
                .map((item) => materialTab(course.id, item, material?.id === item.id))
                .join("")}</div></section>`
            : ""
        }
        ${material ? materialPanel(course.id, material) : ""}
      </main>
    </div>
  `;
}

function materialForm(courseId) {
  return `
    <section class="surface-card card-pad">
      <h2 class="feature-title">Agregar material y generar preguntas</h2>
      <p class="text-xs muted">Subes un PDF o pegas un link de YouTube. Lovable AI extrae el contenido y crea el cuestionario.</p>
      <div class="field" style="margin-top:1rem">
        <label class="tiny-label">Tipo de cuestionario</label>
        <div class="pill-row">
          <button class="tab-button active" type="button" data-action="question-mode" data-mode="open">5 preguntas abiertas</button>
          <button class="tab-button" type="button" data-action="question-mode" data-mode="mcq">10 de selección múltiple</button>
        </div>
      </div>
      <div class="pill-row" style="margin-top:1rem">
        <button class="tab-button active" type="button" data-action="material-kind" data-kind="pdf">PDF</button>
        <button class="tab-button" type="button" data-action="material-kind" data-kind="youtube">YouTube</button>
      </div>
      <form class="form-stack" data-form="material" data-course="${courseId}" data-kind="pdf" data-mode="open" data-level="intermedio">
        <div class="field">
          <label class="tiny-label">Título (opcional)</label>
          <input class="input" name="title" placeholder="Ej. Capítulo 3 — Sociología">
        </div>
        <div class="field">
          <label class="tiny-label" data-material-label>PDF(s)</label>
          <input class="file-input" name="file" type="file" accept="application/pdf" multiple>
          <input class="input" name="url" hidden placeholder="https://youtube.com/watch?v=...">
          <p class="text-xs muted" data-material-help>En esta réplica local se usa el nombre del archivo y texto de ejemplo para generar preguntas.</p>
        </div>
        <div class="field">
          <label class="tiny-label">Nivel de las preguntas</label>
          <div class="pill-row">
            ${["elemental", "intermedio", "avanzado"]
              .map((level) => `<button class="tab-button ${level === "intermedio" ? "active" : ""}" type="button" data-action="question-level" data-level="${level}">${level}</button>`)
              .join("")}
          </div>
          <p class="text-xs muted">Elemental: ideas centrales · Intermedio: aplicación y análisis · Avanzado: pensamiento crítico.</p>
        </div>
        <button class="primary-button" type="submit" data-material-submit>Crear material y generar 5 preguntas</button>
      </form>
    </section>
  `;
}

function materialTab(courseId, material, active) {
  return `
    <a class="tab-button ${active ? "active" : ""}" href="#/courses/${courseId}?material=${material.id}">
      <span class="mono text-xs">${material.kind}</span> ${escapeHtml(material.title)}
    </a>
  `;
}

function materialPanel(courseId, material) {
  const submissions = material.submissions || [];
  const individual = material.individualCodes || [];
  return `
    <div class="section-stack">
      <section class="surface-card card-pad">
        <div class="row-between">
          <div>
            <div class="material-type">${material.kind === "pdf" ? "PDF" : "YouTube"}</div>
            <h2 class="page-title">${escapeHtml(material.title)}</h2>
            ${material.sourceUrl ? `<a class="text-xs primary-text" href="${escapeHtml(material.sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(material.sourceUrl)}</a>` : ""}
          </div>
          <button class="danger-link" data-action="delete-material" data-course="${courseId}" data-material="${material.id}">Eliminar material</button>
        </div>
      </section>

      <section class="surface-card card-pad">
        <h3 class="feature-title">Preguntas generadas <span class="chip">${material.mode === "mcq" ? "Selección múltiple" : "Abiertas"}</span></h3>
        <ol class="question-list" style="margin-top:.75rem">
          ${material.questions.map(questionItem).join("")}
        </ol>
      </section>

      <section class="surface-card card-pad shared-code-card">
        <div class="row-between wrap">
          <div>
            <div class="material-type primary-text">Código de acceso compartido</div>
            <h3 class="feature-title">Un solo código para toda la clase</h3>
            <p class="text-xs muted">Comparte este código con todos tus estudiantes. Pueden usarlo cuantas veces sea necesario para acceder a esta actividad.</p>
          </div>
          <button class="mini-button" data-action="regen-code" data-course="${courseId}" data-material="${material.id}">Regenerar</button>
        </div>
        <div class="big-code-box" style="margin-top:1rem">
          <div class="big-code">${material.sharedCode}</div>
          <div class="text-xs muted">Válido para todos los estudiantes</div>
        </div>
        <div class="small-actions" style="margin-top:.75rem">
          <button class="mini-button" data-action="copy" data-copy="${material.sharedCode}">Copiar código</button>
          <button class="mini-button" data-action="copy" data-copy="${location.origin}${location.pathname}#/join/${material.sharedCode}">Copiar enlace para estudiantes</button>
          <span class="text-xs muted">${location.origin}${location.pathname}#/join/${material.sharedCode}</span>
        </div>
      </section>

      <section class="surface-card card-pad">
        <div class="row-between wrap">
          <h3 class="feature-title">Códigos individuales (opcional)</h3>
          <div class="small-actions">
            ${[1, 5, 10, 25]
              .map((amount) => `<button class="mini-button" data-action="add-codes" data-count="${amount}" data-course="${courseId}" data-material="${material.id}">+${amount}</button>`)
              .join("")}
          </div>
        </div>
        <p class="text-xs muted">Códigos de un solo uso, útiles si prefieres asignar uno por estudiante.</p>
        ${
          individual.length
            ? `<ul class="code-grid" style="padding:0;list-style:none">${individual.map((code) => codeItem(courseId, material.id, code)).join("")}</ul>`
            : `<div class="text-sm muted" style="margin-top:.75rem">Sin códigos individuales generados.</div>`
        }
      </section>

      <section class="surface-card card-pad">
        <div class="row-between wrap">
          <h3 class="feature-title">Entregas y notas</h3>
          ${submissions.length ? `<button class="mini-button" data-action="export-csv" data-course="${courseId}" data-material="${material.id}">Exportar CSV</button>` : ""}
        </div>
        ${submissions.length ? submissionsTable(submissions) : `<div class="text-sm muted" style="margin-top:.75rem">Aún no hay entregas.</div>`}
      </section>
    </div>
  `;
}

function questionItem(question) {
  return `
    <li class="question-item">
      <div class="question-meta"><span class="mono primary-text">P${question.position}</span><span class="chip">${question.competency}</span></div>
      <p class="text-sm">${escapeHtml(question.text)}</p>
      ${
        question.options?.length
          ? `<ul class="option-list">${question.options
              .map((option, index) => `<li class="option-item ${index === question.correctIndex ? "correct" : ""}"><span class="mono">${String.fromCharCode(65 + index)}.</span><span>${escapeHtml(option)}</span>${index === question.correctIndex ? `<span class="option-note">correcta</span>` : ""}</li>`)
              .join("")}</ul>`
          : ""
      }
    </li>
  `;
}

function codeItem(courseId, materialId, item) {
  return `
    <li class="surface-card card-pad row-between">
      <div>
        <div class="mono">${item.code}</div>
        <div class="text-xs muted">${item.usedAt ? `Usado ${formatDateTime(item.usedAt)}` : "Disponible"}</div>
      </div>
      <div class="small-actions">
        <button class="mini-button" data-action="copy" data-copy="${location.origin}${location.pathname}#/join/${item.code}">Copiar link</button>
        ${item.usedAt ? "" : `<button class="mini-button" data-action="delete-code" data-course="${courseId}" data-material="${materialId}" data-code="${item.code}">×</button>`}
      </div>
    </li>
  `;
}

function submissionsTable(submissions) {
  return `
    <div style="overflow-x:auto;margin-top:.75rem">
      <table class="submissions-table">
        <thead><tr><th>Estudiante</th><th>Nota</th><th>Fecha</th><th></th></tr></thead>
        <tbody>
          ${submissions
            .map(
              (submission) => `
              <tr>
                <td><strong>${escapeHtml(submission.studentName)}</strong></td>
                <td>${scoreBadge(submission.finalScore)}</td>
                <td class="text-xs muted">${formatDateTime(submission.createdAt)}</td>
                <td><button class="mini-button" data-action="toggle-submission" data-id="${submission.id}">Ver / editar</button></td>
              </tr>
              <tr id="detail-${submission.id}" hidden>
                <td colspan="4">${submissionDetail(submission)}</td>
              </tr>
            `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function submissionDetail(submission) {
  return `
    <div class="feedback-list" style="padding:1rem">
      ${submission.items
        .map(
          (item, index) => `
          <article class="question-item">
            <div class="question-meta"><span class="mono primary-text">P${index + 1}</span><span class="chip">${escapeHtml(item.evaluation.level)}</span>${scoreBadge(item.evaluation.score)}${item.evaluation.edited ? `<span class="chip">Editada por el profesor</span>` : ""}</div>
            <p class="text-sm"><strong>Pregunta:</strong> ${escapeHtml(item.questionText)}</p>
            ${teacherAnswerPreview(item)}
            <div class="teacher-edit-grid">
              <div class="field">
                <label class="tiny-label">Nota (0-5)</label>
                <input class="input mono" type="number" min="0" max="5" step="0.1" data-edit-score="${submission.id}:${item.questionId}" value="${Number(item.evaluation.score).toFixed(1)}">
              </div>
              <div class="field">
                <label class="tiny-label">Retroalimentación</label>
                <textarea class="textarea compact" data-edit-feedback="${submission.id}:${item.questionId}">${escapeHtml(item.evaluation.feedback)}</textarea>
              </div>
              <button class="primary-button" data-action="save-evaluation" data-submission="${submission.id}" data-question="${item.questionId}">Guardar</button>
            </div>
          </article>
        `,
        )
        .join("")}
    </div>
  `;
}

function teacherAnswerPreview(item) {
  const question = item.question || {};
  if (question.options?.length) {
    const selectedIndex = Number(item.answer ?? "-1");
    return `
      <ul class="option-list">
        ${question.options
          .map((option, optionIndex) => {
            const correct = optionIndex === question.correctIndex;
            const selected = optionIndex === selectedIndex;
            return `<li class="option-item ${correct ? "correct" : selected ? "wrong" : ""}"><span class="mono">${String.fromCharCode(65 + optionIndex)}.</span><span>${escapeHtml(option)}</span>${correct ? `<span class="option-note">correcta</span>` : selected ? `<span class="option-note">elegida</span>` : ""}</li>`;
          })
          .join("")}
      </ul>
    `;
  }
  return `<p class="text-sm"><strong>Respuesta:</strong> ${escapeHtml(item.answer)}</p>`;
}

function scoreBadge(score) {
  const numeric = Number(score || 0);
  const klass = numeric >= 4 ? "high" : numeric >= 3 ? "mid" : "low";
  return `<span class="score ${klass}">${numeric.toFixed(1)}</span>`;
}

function renderJoin() {
  app.innerHTML = `
    <div class="center-shell">
      <div class="surface-card auth-card">
        <a href="#/" class="back-link">← Volver al inicio</a>
        <h1 class="page-title">Soy estudiante</h1>
        <p class="text-sm muted">Ingresa el código que te dio tu profesor para acceder a la evaluación.</p>
        <form class="form-stack" data-form="join">
          <input class="input wide-code" name="code" placeholder="Ej. A3K-9MX" required>
          <button class="primary-button full" type="submit">Continuar</button>
        </form>
      </div>
    </div>
  `;
}

function renderStudent(code) {
  const found = findByCode(code);
  if (!found) {
    app.innerHTML = `
      <div class="page">
        ${studentHeader(code)}
        <main class="container-page student-main">
          ${statusCard("No se pudo continuar", "Código inválido o inexistente.", `<a href="#/join" class="primary-button">Probar otro código</a>`)}
        </main>
      </div>
    `;
    return;
  }
  if (!studentDraft || studentDraft.code !== found.code) {
    studentDraft = {
      code: found.code,
      step: "name",
      studentName: "",
      answers: Object.fromEntries(found.material.questions.map((q) => [q.id, ""])),
      evaluations: [],
      finalScore: null,
    };
  }
  app.innerHTML = `
    <div class="page">
      ${studentHeader(found.code)}
      <main class="container-page student-main">
        ${studentContent(found)}
      </main>
    </div>
  `;
}

function studentContent(found) {
  if (studentDraft.step === "name") {
    const isMcq = found.material.mode === "mcq";
    return `
      <section class="surface-card auth-card student-width">
        <span class="chip">Material: ${escapeHtml(found.material.title)}</span>
        <h2 class="page-title">Antes de empezar</h2>
        <p class="text-sm muted">${isMcq ? `Vas a responder ${found.material.questions.length} preguntas de selección múltiple.` : "Vas a responder 5 preguntas abiertas."} Después de responder verás tu retroalimentación antes de enviarla al profesor.</p>
        <div class="field" style="margin-top:1rem">
          <label>Tu nombre completo</label>
          <input class="input" data-student-name value="${escapeHtml(studentDraft.studentName)}">
        </div>
        <button class="primary-button full" style="margin-top:1rem" data-action="start-student">Empezar</button>
      </section>
    `;
  }

  if (studentDraft.step === "answering") {
    const isMcq = found.material.mode === "mcq";
    const answered = Object.values(studentDraft.answers).filter((value) => (isMcq ? value !== "" : value.trim().length >= 20)).length;
    return `
      <section class="student-width">
        <div class="answer-head">
          <div>
            <span class="chip">Estudiante: ${escapeHtml(studentDraft.studentName)}</span>
            <h2 class="page-title">${isMcq ? `Responde las ${found.material.questions.length} preguntas` : "Responde las 5 preguntas"}</h2>
          </div>
          <div class="text-xs muted hide-mobile">${answered}/${found.material.questions.length}${isMcq ? " respondidas" : " con respuesta sustantiva"}</div>
        </div>
        <ol class="answer-list">
          ${found.material.questions.map((q, index) => answerItem(q, index)).join("")}
        </ol>
        <div style="margin-top:1.25rem;text-align:right">
          <button class="primary-button" data-action="evaluate">Ver mi retroalimentación</button>
        </div>
      </section>
    `;
  }

  if (studentDraft.step === "feedback") {
    return `
      <section class="student-width">
        <div class="surface-card card-pad center-text" style="margin-bottom:1.25rem">
          <span class="chip">Vista previa - todavía no se envía</span>
          <h2 class="page-title">Tu retroalimentación</h2>
          <p class="text-sm muted">Revisa la retroalimentación de cada pregunta. Cuando estés listo, envía tus respuestas al profesor.</p>
          <div style="margin-top:1.25rem">
            <span class="font-serif primary-text" style="font-size:3rem">${studentDraft.finalScore.toFixed(1)}</span>
            <span class="muted">/ 5.0 (preliminar)</span>
          </div>
        </div>
        <ol class="feedback-list">
          ${found.material.questions.map((q, index) => feedbackItem(q, index)).join("")}
        </ol>
        <div class="row-between wrap" style="margin-top:1.5rem">
          <button class="secondary-button" data-action="back-to-answers">← Editar respuestas</button>
          <button class="primary-button" data-action="submit-student">Enviar al profesor</button>
        </div>
      </section>
    `;
  }

  if (studentDraft.step === "done") {
    return `
      <section class="surface-card auth-card student-width center-text">
        <div class="success-icon">✓</div>
        <h2 class="page-title">¡Entrega realizada!</h2>
        <p class="text-sm muted">Tus respuestas y la calificación preliminar fueron enviadas a tu profesor.</p>
        <div style="margin-top:1.5rem">
          <span class="font-serif primary-text" style="font-size:3rem">${studentDraft.finalScore.toFixed(1)}</span>
          <span class="muted">/ 5.0</span>
        </div>
        <p class="text-xs muted">El profesor puede ajustar la nota tras revisar tus respuestas.</p>
      </section>
    `;
  }
  return statusCard("Evaluando tus respuestas", "La IA está aplicando la rúbrica académica...");
}

function statusCard(title, subtitle = "", children = "") {
  return `
    <section class="surface-card auth-card student-width center-text">
      <div class="loading-dot"><div class="spinner"></div></div>
      <h2 class="page-title">${title}</h2>
      ${subtitle ? `<p class="text-sm muted">${subtitle}</p>` : ""}
      ${children ? `<div style="margin-top:1.25rem">${children}</div>` : ""}
    </section>
  `;
}

function answerItem(question, index) {
  const value = studentDraft.answers[question.id] || "";
  return `
    <li class="surface-card card-pad">
      <div class="question-meta"><span class="mono primary-text">P${index + 1}</span><span class="chip">${question.competency}</span></div>
      <p class="answer-question">${escapeHtml(question.text)}</p>
      ${
        question.options?.length
          ? `<ul class="student-options">${question.options
              .map(
                (option, optionIndex) => `
                  <li>
                    <label class="choice-row ${value === String(optionIndex) ? "selected" : ""}">
                      <input type="radio" name="${question.id}" data-answer="${question.id}" value="${optionIndex}" ${value === String(optionIndex) ? "checked" : ""}>
                      <span class="mono">${String.fromCharCode(65 + optionIndex)}.</span>
                      <span>${escapeHtml(option)}</span>
                    </label>
                  </li>
                `,
              )
              .join("")}</ul>`
          : `<textarea class="textarea" data-answer="${question.id}" placeholder="Escribe tu respuesta...">${escapeHtml(value)}</textarea>
             <div class="text-xs muted" style="text-align:right">${value.trim().length} caracteres</div>`
      }
    </li>
  `;
}

function feedbackItem(question, index) {
  const evaluation = studentDraft.evaluations.find((item) => item.questionId === question.id);
  const answer = studentDraft.answers[question.id]?.trim() || "Sin respuesta";
  const selectedIndex = Number(studentDraft.answers[question.id] ?? "-1");
  return `
    <li class="surface-card card-pad">
      <div class="row-between wrap">
        <div class="question-meta"><span class="mono primary-text">P${index + 1}</span><span class="chip">${question.competency}</span></div>
        <div class="small-actions"><span class="chip">${evaluation.level}</span>${scoreBadge(evaluation.score)}</div>
      </div>
      <p class="answer-question">${escapeHtml(question.text)}</p>
      ${
        question.options?.length
          ? `<ul class="option-list">${question.options
              .map((option, optionIndex) => {
                const correct = optionIndex === question.correctIndex;
                const selected = optionIndex === selectedIndex;
                return `<li class="option-item ${correct ? "correct" : selected ? "wrong" : ""}"><span class="mono">${String.fromCharCode(65 + optionIndex)}.</span><span>${escapeHtml(option)}</span>${correct ? `<span class="option-note">correcta</span>` : selected ? `<span class="option-note">tu elección</span>` : ""}</li>`;
              })
              .join("")}</ul>`
          : `<div class="preview-answer text-sm">${escapeHtml(answer)}</div>`
      }
      <div class="feedback-box">
        <div class="feedback-panel">
          <div class="material-type primary-text">Retroalimentación</div>
          <p class="text-sm">${escapeHtml(evaluation.feedback)}</p>
        </div>
        <div>
          <div class="material-type" style="color:var(--success)">Fortalezas</div>
          <ul class="text-sm">${evaluation.strengths.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </div>
        <div>
          <div class="material-type">A mejorar</div>
          <ul class="text-sm">${evaluation.improvements.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </div>
      </div>
    </li>
  `;
}

function levelHint(level) {
  if (level === "elemental") return "Enfócate en ideas centrales.";
  if (level === "avanzado") return "Incluye pensamiento crítico y límites del argumento.";
  return "Conecta comprensión, aplicación y análisis.";
}

function createQuestions(title, sourceText, level = "intermedio") {
  const topic = title || "el material";
  const keywords = sourceText
    .split(/\W+/)
    .filter((word) => word.length > 6)
    .slice(0, 5)
    .join(", ");
  return competencies.map((competency, index) => ({
    id: id("q"),
    position: index + 1,
    competency,
    text: `${seedQuestions[index].replace("del material", `de ${topic}`)} ${levelHint(level)}${keywords ? ` Considera: ${keywords}.` : ""}`,
  }));
}

function createMcqQuestions(title, sourceText, level = "intermedio") {
  const topic = title || "el material";
  const keywords = sourceText
    .split(/\W+/)
    .filter((word) => word.length > 6)
    .slice(0, 4);
  return mcqQuestionTemplates.map((template, index) => {
    const key = keywords[index % Math.max(1, keywords.length)] || "concepto principal";
    const correctIndex = index % 4;
    const correct = `Relacionar ${key} con la tesis principal de ${topic}.`;
    const distractors = [
      `Memorizar una frase aislada sin explicar su contexto.`,
      `Ignorar la evidencia y responder solo desde una opinión personal.`,
      `Cambiar el tema hacia una idea que no aparece en el material.`,
    ];
    const options = [...distractors];
    options.splice(correctIndex, 0, correct);
    return {
      id: id("q"),
      position: index + 1,
      competency: competencies[index % competencies.length],
      text: `${template} ${levelHint(level)}`,
      options,
      correctIndex,
    };
  });
}

function evaluateAnswers(material) {
  if (material.mode === "mcq") {
    const evaluations = material.questions.map((question) => {
      const selected = Number(studentDraft.answers[question.id] ?? "-1");
      const isCorrect = Number.isInteger(selected) && selected === question.correctIndex;
      const correctAnswer = question.options?.[question.correctIndex] || "";
      return {
        questionId: question.id,
        score: isCorrect ? 5 : 0,
        level: isCorrect ? "Excelente" : "Bajo",
        feedback: isCorrect ? "¡Correcta! Escogiste la opción adecuada." : `Incorrecta. La respuesta correcta es ${String.fromCharCode(65 + question.correctIndex)}. ${correctAnswer}.`,
        strengths: isCorrect ? ["Identificaste la opción correcta."] : [],
        improvements: isCorrect ? [] : ["Revisa el material relacionado con esta pregunta."],
      };
    });
    studentDraft.evaluations = evaluations;
    studentDraft.finalScore = Math.round((evaluations.reduce((sum, item) => sum + item.score, 0) / evaluations.length) * 10) / 10;
    return;
  }

  const evaluations = material.questions.map((question) => {
    const answer = (studentDraft.answers[question.id] || "").trim();
    const words = answer ? answer.split(/\s+/).length : 0;
    const score = Math.max(1, Math.min(5, 2.2 + words / 28 + (answer.includes(".") ? 0.35 : 0)));
    const rounded = Math.round(score * 10) / 10;
    return {
      questionId: question.id,
      score: rounded,
      level: rounded >= 4 ? "Alto" : rounded >= 3 ? "Suficiente" : "Inicial",
      feedback:
        rounded >= 4
          ? "Respuesta clara y bien conectada con la pregunta. Muestra comprensión y aporta elaboración propia."
          : rounded >= 3
            ? "Respuesta pertinente, aunque puede desarrollar con más evidencia y relaciones entre conceptos."
            : "La respuesta necesita mayor desarrollo. Intenta explicar con ejemplos y conectar más ideas del material.",
      strengths: answer ? ["Responde directamente a la consigna", "Usa lenguaje propio"] : [],
      improvements: ["Agregar evidencia específica", "Profundizar la conexión con la competencia evaluada"],
    };
  });
  const finalScore = Math.round((evaluations.reduce((sum, item) => sum + item.score, 0) / evaluations.length) * 10) / 10;
  studentDraft.evaluations = evaluations;
  studentDraft.finalScore = finalScore;
}

function handleSubmit(event) {
  const form = event.target.closest("form");
  if (!form) return;
  event.preventDefault();

  if (form.dataset.form === "auth") {
    const data = new FormData(form);
    state.session = { email: data.get("email"), name: data.get("name") || "Profesor" };
    saveState();
    navigate("/dashboard");
  }

  if (form.dataset.form === "course") {
    const data = new FormData(form);
    state.courses.unshift({
      id: id("course"),
      name: data.get("name").trim(),
      description: data.get("description").trim(),
      createdAt: new Date().toISOString(),
      materials: [],
    });
    saveState();
    render();
  }

  if (form.dataset.form === "material") {
    const data = new FormData(form);
    const course = getCourse(form.dataset.course);
    const kind = form.dataset.kind;
    const mode = form.dataset.mode || "open";
    const level = form.dataset.level || "intermedio";
    const file = data.get("file");
    const url = data.get("url")?.trim();
    const title = data.get("title")?.trim() || (kind === "pdf" && file?.name ? file.name : "Video de YouTube");
    const sourceText = `${title} ${url || ""} comprensión análisis aplicación evaluación síntesis pensamiento crítico argumentos evidencia aprendizaje académico`;
    const material = {
      id: id("mat"),
      kind,
      mode,
      level,
      title,
      sourceUrl: kind === "youtube" ? url : "",
      sourceText,
      createdAt: new Date().toISOString(),
      sharedCode: makeCode(),
      individualCodes: [],
      questions: mode === "mcq" ? createMcqQuestions(title, sourceText, level) : createQuestions(title, sourceText, level),
      submissions: [],
    };
    course.materials.unshift(material);
    saveState();
    navigate(`/courses/${course.id}?material=${material.id}`);
  }

  if (form.dataset.form === "join") {
    const code = new FormData(form).get("code").trim().toUpperCase();
    navigate(`/join/${code}`);
  }
}

function handleClick(event) {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;

  if (action === "toggle-auth") {
    sessionStorage.setItem("auth-mode", sessionStorage.getItem("auth-mode") === "signup" ? "login" : "signup");
    render();
  }

  if (action === "logout") {
    state.session = null;
    saveState();
    navigate("/auth");
  }

  if (action === "delete-course" && confirm("¿Eliminar este curso y todos sus datos?")) {
    state.courses = state.courses.filter((course) => course.id !== target.dataset.course);
    saveState();
    render();
  }

  if (action === "material-kind") {
    const form = document.querySelector('[data-form="material"]');
    const isYoutube = target.dataset.kind === "youtube";
    form.dataset.kind = target.dataset.kind;
    document.querySelectorAll('[data-action="material-kind"]').forEach((button) => button.classList.toggle("active", button === target));
    form.querySelector('[name="file"]').hidden = isYoutube;
    form.querySelector('[name="url"]').hidden = !isYoutube;
    form.querySelector("[data-material-label]").textContent = isYoutube ? "URL del video de YouTube" : "PDF(s)";
    form.querySelector("[data-material-help]").textContent = isYoutube
      ? "El servidor traerá los subtítulos públicos del video automáticamente. En esta réplica se simula ese paso."
      : "En esta réplica local se usa el nombre del archivo y texto de ejemplo para generar preguntas.";
  }

  if (action === "question-mode") {
    const form = document.querySelector('[data-form="material"]');
    form.dataset.mode = target.dataset.mode;
    document.querySelectorAll('[data-action="question-mode"]').forEach((button) => button.classList.toggle("active", button === target));
    const submit = form.querySelector("[data-material-submit]");
    submit.textContent = target.dataset.mode === "mcq" ? "Crear material y generar 10 preguntas" : "Crear material y generar 5 preguntas";
  }

  if (action === "question-level") {
    const form = document.querySelector('[data-form="material"]');
    form.dataset.level = target.dataset.level;
    document.querySelectorAll('[data-action="question-level"]').forEach((button) => button.classList.toggle("active", button === target));
  }

  if (action === "delete-material" && confirm("¿Eliminar este material y todos sus datos asociados?")) {
    const course = getCourse(target.dataset.course);
    course.materials = course.materials.filter((item) => item.id !== target.dataset.material);
    saveState();
    navigate(`/courses/${course.id}`);
  }

  if (action === "regen-code" && confirm("¿Generar un nuevo código compartido? El anterior dejará de funcionar.")) {
    const material = getMaterial(target.dataset.course, target.dataset.material);
    material.sharedCode = makeCode();
    saveState();
    render();
  }

  if (action === "add-codes") {
    const material = getMaterial(target.dataset.course, target.dataset.material);
    const count = Number(target.dataset.count);
    for (let i = 0; i < count; i += 1) material.individualCodes.push({ code: makeCode(), usedAt: null });
    saveState();
    render();
  }

  if (action === "delete-code") {
    const material = getMaterial(target.dataset.course, target.dataset.material);
    material.individualCodes = material.individualCodes.filter((item) => item.code !== target.dataset.code);
    saveState();
    render();
  }

  if (action === "copy") {
    navigator.clipboard?.writeText(target.dataset.copy);
    target.textContent = "Copiado";
    setTimeout(() => render(), 700);
  }

  if (action === "toggle-submission") {
    const row = document.querySelector(`#detail-${CSS.escape(target.dataset.id)}`);
    row.hidden = !row.hidden;
    target.textContent = row.hidden ? "Ver / editar" : "Cerrar";
  }

  if (action === "save-evaluation") {
    saveEvaluationEdit(target.dataset.submission, target.dataset.question);
  }

  if (action === "export-csv") {
    exportCsv(target.dataset.course, target.dataset.material);
  }

  if (action === "start-student") {
    const input = document.querySelector("[data-student-name]");
    studentDraft.studentName = input.value.trim();
    if (studentDraft.studentName.length < 2) return;
    studentDraft.step = "answering";
    render();
  }

  if (action === "evaluate") {
    document.querySelectorAll("[data-answer]").forEach((field) => {
      if (field.type === "radio" && !field.checked) return;
      studentDraft.answers[field.dataset.answer] = field.value;
    });
    const found = findByCode(studentDraft.code);
    evaluateAnswers(found.material);
    studentDraft.step = "feedback";
    render();
  }

  if (action === "back-to-answers") {
    studentDraft.step = "answering";
    render();
  }

  if (action === "submit-student") {
    const found = findByCode(studentDraft.code);
    found.material.submissions.unshift({
      id: id("sub"),
      studentName: studentDraft.studentName,
      finalScore: studentDraft.finalScore,
      createdAt: new Date().toISOString(),
      items: found.material.questions.map((question) => ({
        questionId: question.id,
        question,
        questionText: question.text,
        answer: studentDraft.answers[question.id] || "",
        evaluation: studentDraft.evaluations.find((item) => item.questionId === question.id),
      })),
    });
    if (!found.shared && found.individual) found.individual.usedAt = new Date().toISOString();
    studentDraft.step = "done";
    saveState();
    render();
  }
}

function exportCsv(courseId, materialId) {
  const material = getMaterial(courseId, materialId);
  const rows = [["Estudiante", "Nota final", "Fecha"], ...material.submissions.map((item) => [item.studentName, item.finalScore, formatDateTime(item.createdAt)])];
  const csv = rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const blob = new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `notas-${material.title}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function saveEvaluationEdit(submissionId, questionId) {
  for (const course of state.courses) {
    for (const material of course.materials) {
      const submission = material.submissions.find((item) => item.id === submissionId);
      if (!submission) continue;
      const item = submission.items.find((entry) => entry.questionId === questionId);
      if (!item) return;
      const key = `${submissionId}:${questionId}`;
      const scoreInput = document.querySelector(`[data-edit-score="${CSS.escape(key)}"]`);
      const feedbackInput = document.querySelector(`[data-edit-feedback="${CSS.escape(key)}"]`);
      const score = Math.max(0, Math.min(5, Number(scoreInput?.value || 0)));
      item.evaluation.score = Math.round(score * 10) / 10;
      item.evaluation.feedback = feedbackInput?.value || "";
      item.evaluation.edited = true;
      item.evaluation.level = item.evaluation.score >= 4 ? "Alto" : item.evaluation.score >= 3 ? "Suficiente" : "Inicial";
      submission.finalScore = Math.round((submission.items.reduce((sum, entry) => sum + Number(entry.evaluation.score || 0), 0) / submission.items.length) * 10) / 10;
      saveState();
      render();
      setTimeout(() => {
        const row = document.querySelector(`#detail-${CSS.escape(submissionId)}`);
        const button = document.querySelector(`[data-action="toggle-submission"][data-id="${CSS.escape(submissionId)}"]`);
        if (row && button) {
          row.hidden = false;
          button.textContent = "Cerrar";
        }
      }, 0);
      return;
    }
  }
}

app.addEventListener("submit", handleSubmit);
app.addEventListener("click", handleClick);

function render() {
  const current = route();
  const [path, queryString] = current.split("?");
  const query = new URLSearchParams(queryString || "");

  if (path === "/") return renderHome();
  if (path === "/auth") return renderAuth();
  if (path === "/dashboard") return renderDashboard();
  if (path === "/join") return renderJoin();
  if (path.startsWith("/join/")) return renderStudent(decodeURIComponent(path.split("/")[2] || ""));
  if (path.startsWith("/courses/")) return renderCourse(path.split("/")[2], query.get("material"));
  return renderHome();
}

render();
