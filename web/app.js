const el = (id) => document.getElementById(id);

let currentSessionId = null;
let es = null;
let lastPlanText = "";
let lastStepResult = "";
let drawerMinimized = false;

function setStatus(s) {
  el("status").textContent = s;
}

function addEventCard(type, payload) {
  const wrap = document.createElement("div");
  wrap.className = "evt" + (type === "error" ? " error" : "");
  const k = document.createElement("div");
  k.className = "k";
  k.textContent = type;
  const v = document.createElement("div");
  v.className = "v";
  v.textContent = JSON.stringify(payload, null, 2);
  wrap.appendChild(k);
  wrap.appendChild(v);
  el("events").prepend(wrap);
}

function showModal(req) {
  el("modal").classList.remove("hidden");
  drawerMinimized = false;
  el("modalRequestId").value = req.request_id;
  el("modalTitle").textContent = "需要你的输入";
  el("modalBody").textContent = req.prompt || "";
  el("modalPlan").textContent = lastPlanText || "";
  el("modalStepResult").textContent = lastStepResult || "";

  const controls = el("modalControls");
  controls.innerHTML = "";

  if (req.kind === "confirm") {
    const sel = document.createElement("select");
    sel.id = "modalValue";
    const optY = document.createElement("option");
    optY.value = "y";
    optY.textContent = "是 / 继续 (y)";
    const optN = document.createElement("option");
    optN.value = "n";
    optN.textContent = "否 / 停止 (n)";
    sel.appendChild(optY);
    sel.appendChild(optN);
    if ((req.default || "y").toLowerCase() === "n") sel.value = "n";
    controls.appendChild(sel);
  } else if (req.kind === "choice") {
    const sel = document.createElement("select");
    sel.id = "modalValue";
    (req.choices || []).forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    });
    sel.value = req.default || (req.choices ? req.choices[0] : "");
    controls.appendChild(sel);
  } else {
    const ta = document.createElement("textarea");
    ta.id = "modalValue";
    ta.rows = 5;
    ta.placeholder = req.allow_empty ? "可空" : "不能为空";
    controls.appendChild(ta);
  }
}

function hideModal() {
  el("modal").classList.add("hidden");
  el("modalControls").innerHTML = "";
  el("modalRequestId").value = "";
}

async function startRun() {
  const prompt = el("prompt").value;
  const planningTemplate = el("planningTemplate").files[0] || null;
  const businessFiles = Array.from(el("businessFiles").files || []);

  const fd = new FormData();
  fd.append("prompt", prompt);
  if (planningTemplate) fd.append("planning_template", planningTemplate);
  businessFiles.forEach((f) => fd.append("business_files", f));

  setStatus("starting...");
  el("startBtn").disabled = true;

  const resp = await fetch("/api/run", { method: "POST", body: fd });
  if (!resp.ok) {
    const t = await resp.text();
    setStatus("error");
    el("startBtn").disabled = false;
    throw new Error(t);
  }
  const data = await resp.json();
  currentSessionId = data.session_id;
  el("sessionId").textContent = currentSessionId;
  setStatus("running");

  if (es) es.close();
  es = new EventSource(`/api/sessions/${currentSessionId}/events`);
  es.addEventListener("ping", () => {});
  es.addEventListener("session_start", (e) => addEventCard("session_start", JSON.parse(e.data)));
  es.addEventListener("run_start", (e) => addEventCard("run_start", JSON.parse(e.data)));
  es.addEventListener("plan", (e) => {
    const p = JSON.parse(e.data);
    lastPlanText = p.text || "";
    el("planBox").textContent = lastPlanText;
    addEventCard("plan", p);
  });
  es.addEventListener("step_start", (e) => addEventCard("step_start", JSON.parse(e.data)));
  es.addEventListener("step_end", (e) => {
    const payload = JSON.parse(e.data);
    lastStepResult = payload.result || "";
    el("stepResultBox").textContent = lastStepResult;
    addEventCard("step_end", payload);
  });
  es.addEventListener("text", (e) => addEventCard("text", JSON.parse(e.data)));
  es.addEventListener("human_request", (e) => {
    const req = JSON.parse(e.data);
    addEventCard("human_request", req);
    showModal(req);
  });
  es.addEventListener("human_response", (e) => addEventCard("human_response", JSON.parse(e.data)));
  es.addEventListener("human_timeout", (e) => addEventCard("human_timeout", JSON.parse(e.data)));
  es.addEventListener("run_end", (e) => {
    const r = JSON.parse(e.data);
    addEventCard("run_end", r);
    setStatus("done");
    el("startBtn").disabled = false;
  });
  es.addEventListener("error", (e) => {
    // SSE 'error' can be connection error, but we also send app error event.
    try {
      const payload = JSON.parse(e.data || "{}");
      addEventCard("error", payload);
    } catch (_) {}
    setStatus("error");
    el("startBtn").disabled = false;
  });
}

el("businessFiles").addEventListener("change", () => {
  const files = Array.from(el("businessFiles").files || []);
  el("fileList").textContent = files.length ? files.map((f) => f.name).join(", ") : "";
});

el("startBtn").addEventListener("click", async () => {
  try {
    await startRun();
  } catch (e) {
    addEventCard("client_error", { message: String(e) });
  }
});

el("clearBtn").addEventListener("click", () => {
  el("events").innerHTML = "";
  el("planBox").textContent = "";
});

el("modalForm").addEventListener("submit", async (evt) => {
  evt.preventDefault();
  if (!currentSessionId) return;
  const requestId = el("modalRequestId").value;
  const valueEl = document.getElementById("modalValue");
  const value = valueEl ? valueEl.value : "";

  const fd = new FormData();
  fd.append("value", value);
  await fetch(`/api/sessions/${currentSessionId}/human/${requestId}`, {
    method: "POST",
    body: fd,
  });

  hideModal();
});

el("modalMinBtn").addEventListener("click", () => {
  // 仅最小化抽屉内容，不影响主界面查看计划/结果
  drawerMinimized = !drawerMinimized;
  const body = el("modalBody");
  const form = el("modalForm");
  const details = Array.from(document.querySelectorAll(".drawer-details"));
  if (drawerMinimized) {
    body.style.display = "none";
    details.forEach((d) => (d.style.display = "none"));
    form.style.display = "none";
    el("modalTitle").textContent = "需要你的输入（已最小化，点击展开）";
    el("modal").addEventListener(
      "click",
      () => {
        if (!drawerMinimized) return;
        drawerMinimized = false;
        body.style.display = "";
        details.forEach((d) => (d.style.display = ""));
        form.style.display = "";
        el("modalTitle").textContent = "需要你的输入";
      },
      { once: true }
    );
  } else {
    body.style.display = "";
    details.forEach((d) => (d.style.display = ""));
    form.style.display = "";
    el("modalTitle").textContent = "需要你的输入";
  }
});
