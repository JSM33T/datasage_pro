// src/app/components/chat-component/chat-component.ts
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

import * as Prism from 'prismjs';
import 'prismjs/components/prism-python';     // add more as needed
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-javascript';


@Component({
	selector: 'app-chat-component',
	standalone: true,
	imports: [CommonModule, FormsModule],
	templateUrl: `chat-component.html`
})
export class ChatComponent implements OnInit {
	searchTerm: string = '';
	searchResults: any[] = [];
	pinnedDocs: any[] = [];
	selectedDocIds: string[] = [];
	messages: { role: string; content: string }[] = [];
	sessionId: string | null = null;
	loading: boolean = false;
	query: string = '';

	constructor(private http: HttpClient) { }

	// ngOnInit(): void { }

	async ngOnInit(): Promise<void> {
		const savedSession = localStorage.getItem('activeSession');
		if (savedSession) {
			this.sessionId = savedSession;
			try {
				const res: any = await this.http.get(`${environment.apiBase}/chat_session/get_session/${savedSession}`).toPromise();
				this.messages = res.messages || [];
				this.selectedDocIds = res.doc_ids || [];
				setTimeout(() => Prism.highlightAll(), 0);
			} catch (err) {
				console.error('Session fetch failed:', err);
				this.sessionId = null;
				this.messages = [];
			}
		}
	}



	async searchDocuments() {
		const term = this.searchTerm.trim();
		if (!term) {
			this.searchResults = [];
			return;
		}
		const res = await this.http.get<any[]>(`${environment.apiBase}/document/search?query=${encodeURIComponent(term)}`).toPromise();
		this.searchResults = Array.isArray(res) ? res.filter(doc => doc.isIndexed) : [];
	}

	// get visibleDocs(): any[] {
	// 	const pinnedIds = new Set(this.pinnedDocs.map(d => d.id));
	// 	const unpinned = this.searchResults.filter(doc => !pinnedIds.has(doc.id));
	// 	return [...this.pinnedDocs, ...unpinned.slice(0, 5)];
	// }

	get visibleDocs(): any[] {
		const pinnedIds = new Set(this.pinnedDocs.map(d => d.id));
		const unpinned = this.searchResults.filter(doc => !pinnedIds.has(doc.id));
		return [...this.pinnedDocs, ...unpinned.slice(0, 5)];
	}


	togglePin(doc: any): void {
		const index = this.pinnedDocs.findIndex(d => d.id === doc.id);
		if (index > -1) {
			this.pinnedDocs.splice(index, 1);
		} else {
			this.pinnedDocs.push(doc);
		}
	}


	toggleSelect(docId: string, isChecked: boolean) {
		if (isChecked) {
			this.selectedDocIds.push(docId);
		} else {
			this.selectedDocIds = this.selectedDocIds.filter(id => id !== docId);
		}
	}

	getPinLabel(docId: string): string {
		return this.pinnedDocs.some(p => p.id === docId) ? 'Unpin' : 'Pin';
	}

	getMessageClass(role: string): string {
		return role === 'user' ? 'text-primary' : 'text-dark';
	}

	isPinned(docId: string): boolean {
		return this.pinnedDocs.some(d => d.id === docId);
	}

	onCheckboxChange(event: Event, docId: string): void {
		const checked = (event.target as HTMLInputElement).checked;
		this.toggleSelect(docId, checked);
	}

	formatMessage(text: string): string {
		if (!text) return '';

		// Escape HTML
		const escaped = text
			.replace(/&/g, '&amp;')
			.replace(/</g, '&lt;')
			.replace(/>/g, '&gt;');

		// Replace ```lang\ncode``` blocks
		const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
		const formatted = escaped.replace(codeBlockRegex, (_, lang, code) => {
			return `<pre><code class="language-${lang || 'plaintext'}">${code}</code></pre>`;
		});

		// Replace single newlines with <br> (but not inside code)
		return formatted
			.replace(/(?!<\/pre>)\n/g, '<br>');
	}

	async sendMessage(): Promise<void> {
		const text = this.query.trim();
		if (!text || this.selectedDocIds.length === 0) {
			alert('Enter query and select documents');
			return;
		}

		if (!this.sessionId) {
			try {
				const res: any = await this.http.post(`${environment.apiBase}/chat_session/start`, {
					doc_ids: this.selectedDocIds
				}).toPromise();
				this.sessionId = res.session_id;
				localStorage.setItem('activeSession', this.sessionId ?? '');
			} catch (err) {
				console.error('Failed to start session:', err);
				return;
			}
		}

		const sessionId = this.sessionId;
		if (!sessionId) return;

		this.messages.push({ role: 'user', content: text });
		this.query = '';
		this.loading = true;

		try {
			const res: any = await this.http.post(`${environment.apiBase}/chat_session/continue`, {
				session_id: sessionId,
				query: text
			}).toPromise();

			//this.messages.push(...res.messages.filter((m: { role: string }) => m.role === 'assistant'));
			this.messages = res.messages;
			setTimeout(() => Prism.highlightAll(), 0);
		} catch (err) {
			console.error('Failed to continue session:', err);
		}

		this.loading = false;
	}

}
