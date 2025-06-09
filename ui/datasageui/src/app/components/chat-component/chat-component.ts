import {
	Component,
	OnInit,
	ViewChild,
	ElementRef
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';
import { firstValueFrom } from 'rxjs';

import * as Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-sql';
import 'prismjs/components/prism-batch';

@Component({
	selector: 'app-chat-component',
	standalone: true,
	imports: [CommonModule, FormsModule],
	templateUrl: 'chat-component.html',
})
export class ChatComponent implements OnInit {
	@ViewChild('chatWindow') chatWindowRef!: ElementRef;

	searchTerm = '';
	searchResults: any[] = [];
	pinnedDocs: any[] = [];
	selectedDocIds: string[] = [];
	messages: { role: string; content: string }[] = [];
	query = '';
	loading = false;

	sessionId: string | null = null;
	sessionOptions: {
		id: string;
		createdAt: string;
		documents: { _id: string; name: string; filename: string }[]
	}[] = [];
	activeSessionId: string | null = null;

	constructor(private http: HttpClient) { }

	ngAfterViewChecked() {
		this.scrollToBottom();
	}

	private scrollToBottom(): void {
		try {
			this.chatWindowRef.nativeElement.scrollTop = this.chatWindowRef.nativeElement.scrollHeight;
		} catch (err) { }
	}

	async ngOnInit(): Promise<void> {
		await this.loadSessionOptions();
		const savedSession = localStorage.getItem('activeSession');
		if (typeof savedSession === 'string') {
			await this.loadSession(savedSession);
		}
	}

	getDocumentNames(docs: { name: string }[]): string {
		return docs.map(d => d.name).join(', ');
	}

	async loadSessionOptions() {
		try {
			const res: any = await firstValueFrom(
				this.http.get(`${environment.apiBase}/chat_session/list_sessions`)
			);
			this.sessionOptions = res;
		} catch (err) {
			console.error('Failed to fetch session list:', err);
		}
	}

	async loadSession(sessionId: string) {
		try {
			const res: any = await firstValueFrom(
				this.http.get(`${environment.apiBase}/chat_session/get_session/${sessionId}`)
			);

			this.sessionId = res.session_id;
			this.activeSessionId = sessionId;
			this.messages = res.messages || [];
			this.selectedDocIds = res.doc_ids || [];

			localStorage.setItem('activeSession', sessionId);

			if (this.selectedDocIds.length > 0) {
				const docRes = await firstValueFrom(
					this.http.get<any[]>(
						`${environment.apiBase}/document/by_ids?ids=${this.selectedDocIds.join(',')}`
					)
				);

				// Auto-pin all selected documents
				this.pinnedDocs = Array.isArray(docRes)
					? docRes.filter(doc => this.selectedDocIds.includes(doc.id))
					: [];
			} else {
				this.pinnedDocs = [];
			}

			setTimeout(() => Prism.highlightAll(), 0);
		} catch (err) {
			console.error('Failed to load session:', err);
			this.sessionId = null;
			this.messages = [];
			this.pinnedDocs = [];
		}
	}

	async searchDocuments() {
		const term = this.searchTerm.trim();
		if (!term) {
			this.searchResults = [];
			return;
		}
		const res = await firstValueFrom(
			this.http.get<any[]>(
				`${environment.apiBase}/document/search?query=${encodeURIComponent(term)}`
			)
		);
		this.searchResults = Array.isArray(res) ? res.filter(doc => doc.isIndexed) : [];
	}

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

	async startNewSession(): Promise<void> {
		this.sessionId = null;
		this.messages = [];
		this.selectedDocIds = [];
		localStorage.removeItem('activeSession');
	}

	isPinned(docId: string): boolean {
		return this.pinnedDocs.some(d => d.id === docId);
	}

	toggleSelect(docId: string, isChecked: boolean) {
		if (isChecked) {
			this.selectedDocIds.push(docId);
		} else {
			this.selectedDocIds = this.selectedDocIds.filter(id => id !== docId);
		}
	}

	onCheckboxChange(event: Event, docId: string): void {
		const checked = (event.target as HTMLInputElement).checked;
		this.toggleSelect(docId, checked);
	}

	getPinLabel(docId: string): string {
		return this.pinnedDocs.some(p => p.id === docId) ? 'Unpin' : 'Pin';
	}

	getMessageClass(role: string): string {
		return role === 'user' ? 'text-primary' : 'text-dark';
	}

	formatMessage(text: string): string {
		if (!text) return '';

		const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
		const codeBlocks: string[] = [];
		let i = 0;

		// Replace code blocks with tokens
		const temp = text.replace(codeBlockRegex, (_, lang, code) => {
			codeBlocks[i] = `<pre><code class="language-${lang || 'plaintext'}">${code
				.replace(/&/g, '&amp;')
				.replace(/</g, '&lt;')
				.replace(/>/g, '&gt;')}</code></pre>`;
			return `%%CODEBLOCK_${i++}%%`;
		});

		// Escape non-code text and replace \n with <br>
		const escaped = temp
			.replace(/&/g, '&amp;')
			.replace(/</g, '&lt;')
			.replace(/>/g, '&gt;')
			.replace(/\n/g, '<br>');

		// Re-insert code blocks
		return escaped.replace(/%%CODEBLOCK_(\d+)%%/g, (_, j) => codeBlocks[+j]);
	}

	async sendMessage(): Promise<void> {
		const text = this.query.trim();
		if (!text || this.selectedDocIds.length === 0) {
			alert('Enter query and select documents');
			return;
		}

		if (!this.sessionId) {
			try {
				const res: any = await firstValueFrom(
					this.http.post(`${environment.apiBase}/chat_session/start`, {
						doc_ids: this.selectedDocIds,
					})
				);
				this.sessionId = res.session_id;
				localStorage.setItem('activeSession', this.sessionId ?? '');
				this.activeSessionId = this.sessionId;
				if (this.sessionId) {
					await this.loadSession(this.sessionId);
				}
			} catch (err) {
				console.error('Failed to start session:', err);
				return;
			}
		}

		this.messages.push({ role: 'user', content: text });
		this.query = '';
		this.loading = true;

		try {
			const res: any = await firstValueFrom(
				this.http.post(`${environment.apiBase}/chat_session/continue`, {
					session_id: this.sessionId,
					query: text,
				})
			);
			this.messages = res.messages;
			setTimeout(() => Prism.highlightAll(), 0);
		} catch (err) {
			console.error('Failed to continue session:', err);
		}

		this.loading = false;
	}
}
