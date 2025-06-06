const API_BASE = "/api";
let sessionId = null;

document.addEventListener("DOMContentLoaded", () => {
	fetchDocs();

	document.getElementById("uploadForm").addEventListener("submit", async (e) => {
		e.preventDefault();
		const formData = new FormData();
		formData.append("file", document.getElementById("docFile").files[0]);
		formData.append("name", document.getElementById("docName").value);
		formData.append("description", document.getElementById("docDescription").value);

		await fetch(`${API_BASE}/document/add`, { method: "POST", body: formData });
		await fetchDocs();
	});

	document.getElementById("sendBtn").addEventListener("click", chatWithDocs);
});

async function fetchDocs() {
	const res = await fetch(`${API_BASE}/document/list`);
	const docs = await res.json();
	const list = document.getElementById("docList");
	list.innerHTML = "";

	docs.forEach(doc => {
		const li = document.createElement("li");
		li.className = "list-group-item d-flex justify-content-between align-items-center";

		const left = document.createElement("div");
		left.innerHTML = `
      <strong>${doc.name}</strong> <small class="text-muted">(${doc.filename})</small><br/>
      Indexed: <span class="${doc.isIndexed ? 'text-success' : 'text-danger'}">${doc.isIndexed}</span>
    `;

		const right = document.createElement("div");

		if (doc.isIndexed) {
			const checkbox = document.createElement("input");
			checkbox.type = "checkbox";
			checkbox.className = "form-check-input me-2";
			checkbox.value = doc.id;
			checkbox.name = "docCheckbox";
			right.appendChild(checkbox);
		} else {
			const button = document.createElement("button");
			button.className = "btn btn-warning btn-sm";
			button.innerText = "Index";
			button.onclick = async () => {
				button.disabled = true;
				await fetch(`${API_BASE}/indexing/index`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ doc_ids: [doc.id] })
				});
				await fetchDocs();
			};
			right.appendChild(button);
		}

		li.appendChild(left);
		li.appendChild(right);
		list.appendChild(li);
	});
}

async function startSession(docIds) {
	const res = await fetch(`${API_BASE}/chat_session/start`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ doc_ids: docIds })
	});
	const data = await res.json();
	sessionId = data.session_id;
}

async function chatWithDocs() {
	const query = document.getElementById("chatQuery").value.trim();
	const chatBox = document.getElementById("chatResponse");
	const ids = Array.from(document.querySelectorAll('input[name="docCheckbox"]:checked')).map(cb => cb.value);

	if (!query || ids.length === 0) {
		alert("Please enter a query and select at least one indexed document.");
		return;
	}

	if (!sessionId) await startSession(ids);

	const res = await fetch(`${API_BASE}/chat_session/continue`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ session_id: sessionId, query })
	});

	const data = await res.json();
	chatBox.innerHTML = "";
	data.messages.forEach(msg => {
		const div = document.createElement("div");
		div.className = msg.role === "user" ? "text-primary mb-2" : "text-dark mb-2";
		div.innerHTML = `<strong>${msg.role}:</strong> ${msg.content}`;
		chatBox.appendChild(div);
	});
}
