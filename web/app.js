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
  // 过滤掉中间推理过程，只显示关键事件
  const skipTypes = ["text", "step_start"]; // 不显示这些中间过程
  if (skipTypes.includes(type)) {
    return; // 跳过中间推理过程的显示
  }

  const wrap = document.createElement("div");
  wrap.className = "evt" + (type === "error" ? " error" : "");
  const k = document.createElement("div");
  k.className = "k";
  k.textContent = type;
  const v = document.createElement("div");
  v.className = "v";

  // 对于 step_end，只显示最终结果摘要
  if (type === "step_end" && payload.result) {
    const result = payload.result;
    // 提取最终结果（去掉中间过程）
    const finalResult = extractFinalResult(result);
    v.textContent = finalResult || result.substring(0, 500) + (result.length > 500 ? "..." : "");
  } else {
    v.textContent = JSON.stringify(payload, null, 2);
  }

  wrap.appendChild(k);
  wrap.appendChild(v);
  el("events").prepend(wrap);
}

function extractFinalResult(resultText) {
  // 尝试提取最终结果，去掉中间推理过程
  // 查找常见的最终结果标记
  const markers = [
    /最终结果[：:]\s*(.+)/i,
    /总结[：:]\s*(.+)/i,
    /结论[：:]\s*(.+)/i,
    /完成[：:]\s*(.+)/i,
    /结果[：:]\s*(.+)/i,
  ];

  for (const marker of markers) {
    const match = resultText.match(marker);
    if (match) {
      return match[1].trim();
    }
  }

  // 如果没有找到标记，返回最后500字符（通常是最终结果）
  return resultText.length > 500 ? resultText.substring(resultText.length - 500) : resultText;
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
    // 创建选项说明
    const desc = document.createElement("div");
    desc.style.marginBottom = "10px";
    desc.style.fontSize = "12px";
    desc.style.color = "rgba(233, 240, 255, 0.8)";
    desc.textContent = "请从以下选项中选择：";
    controls.appendChild(desc);

    const sel = document.createElement("select");
    sel.id = "modalValue";
    sel.style.width = "100%";
    sel.style.marginBottom = "10px";

    // 根据选项值显示更清晰的文本
    const choiceLabels = {
      "1": "1 - 提供纠正/反馈（会记录到 notes/记忆）",
      "2": "2 - 标记该步骤为阻塞/需要重新执行",
      "3": "3 - 接受当前结果并继续（可选填写备注）",
    };

    (req.choices || []).forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = choiceLabels[c] || c;
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
    // 只显示最终结果摘要，不显示完整中间过程
    const finalResult = extractFinalResult(lastStepResult);
    el("stepResultBox").textContent = finalResult || lastStepResult.substring(0, 1000) + (lastStepResult.length > 1000 ? "\n...(结果已截断，完整内容见事件日志)" : "");
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
  es.addEventListener("step_confirmation_request", (e) => {
    const req = JSON.parse(e.data);
    console.log("[FRONTEND] Received step_confirmation_request:", req);
    // 自动显示步骤确认弹窗
    const confirmReq = {
      kind: "confirm",
      request_id: req.request_id || `step_confirm_${req.step_index}`,
      prompt: `步骤 ${req.step_index} 执行完成。\n步骤: ${req.step_info || '未知'}\n\n结果预览:\n${req.result_preview || ''}\n\n结果是否可接受？`,
      default: "y",
    };
    console.log("[FRONTEND] Showing modal for step confirmation:", confirmReq);
    showModal(confirmReq);
  });
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
  if (!currentSessionId) {
    console.error("No session ID");
    return;
  }
  const requestId = el("modalRequestId").value;
  if (!requestId) {
    console.error("No request ID");
    return;
  }
  const valueEl = document.getElementById("modalValue");
  const value = valueEl ? valueEl.value : "";

  // 禁用提交按钮，防止重复提交
  const submitBtn = evt.target.querySelector('button[type="submit"]');
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = "提交中...";
  }

  try {
    const fd = new FormData();
    fd.append("value", value);

    const response = await fetch(`/api/sessions/${currentSessionId}/human/${requestId}`, {
      method: "POST",
      body: fd,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${errorText}`);
    }

    const result = await response.json();
    if (!result.ok) {
      throw new Error(result.error || "Unknown error");
    }

    hideModal();
  } catch (error) {
    console.error("Error submitting human response:", error);
    alert(`提交失败: ${error.message}`);
    // 恢复按钮状态
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = "提交";
    }
  }
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
