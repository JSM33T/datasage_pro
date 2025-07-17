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
	styleUrls: ['chat-component.css']
})
export class ChatComponent implements OnInit {
	/**
	 * Delete a session completely
	 */
	async deleteSession(sessionId: string): Promise<void> {
		if (!sessionId) return;

		const sessionDisplay = sessionId.slice(0, 8);
		this.showConfirmationDialog(
			'Delete Session',
			`Are you sure you want to permanently delete session ${sessionDisplay}? This action cannot be undone.`,
			async () => {
				try {
					await firstValueFrom(
						this.http.post(`${environment.apiBase}/chat/delete_session/${sessionId}`, {})
					);

					// If it's the current active session, clear local state
					if (sessionId === this.activeSessionId) {
						this.messages = [];
						this.activeSessionId = null;
						this.sessionId = null;
						localStorage.removeItem('activeSession');
					}

					// Reload session options to reflect changes
					await this.loadSessionOptions();

					this.showStatus(`Session ${sessionDisplay} has been deleted successfully.`, 'success');
				} catch (err) {
					console.error('Failed to delete session:', err);
					this.showStatus('Failed to delete session. Please try again.', 'error');
				}
			}
		);
	}
	// Select all visible documents
	areAllVisibleSelected(): boolean {
		const visibleIds = this.visibleDocs.map((doc: any) => doc.id);
		return visibleIds.length > 0 && visibleIds.every((id: string) => this.selectedDocIds.includes(id));
	}

	toggleSelectAllVisible(): void {
		const visibleIds = this.visibleDocs.map((doc: any) => doc.id);
		if (this.areAllVisibleSelected()) {
			// Unselect all visible
			this.selectedDocIds = this.selectedDocIds.filter((id: string) => !visibleIds.includes(id));
			this.pinnedDocs = this.pinnedDocs.filter((doc: any) => !visibleIds.includes(doc.id));
		} else {
			// Select all visible
			for (const id of visibleIds) {
				if (!this.selectedDocIds.includes(id)) {
					this.selectedDocIds.push(id);
				}
				// Auto-pin if not already pinned
				const doc = [...this.searchResults, ...this.pinnedDocs].find((d: any) => d.id === id);
				if (doc && !this.pinnedDocs.some((d: any) => d.id === id)) {
					this.pinnedDocs.push(doc);
				}
			}
		}
	}
	@ViewChild('chatWindow') chatWindowRef!: ElementRef;

	searchTerm = '';
	searchResults: any[] = [];
	pinnedDocs: any[] = [];
	selectedDocIds: string[] = [];
	messages: { role: string; content: string }[] = [];
	query = '';
	loading = false;
	statusMessage = '';
	statusType: 'success' | 'error' | 'info' | '' = '';

	// Confirmation dialog properties
	showConfirmDialog = false;
	confirmTitle = '';
	confirmMessage = '';
	confirmAction: (() => void) | null = null;

	sessionId: string | null = null;
	originalSelectedDocIds: string[] = []; // Store original doc IDs for comparison
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
				this.http.get(`${environment.apiBase}/chat/list_sessions`)
			);
			this.sessionOptions = res;

			// If current active session is no longer in the list, reset it
			if (this.activeSessionId && !this.sessionOptions.some(s => s.id === this.activeSessionId)) {
				this.activeSessionId = null;
				this.sessionId = null;
				this.messages = [];
				localStorage.removeItem('activeSession');
			}
		} catch (err) {
			console.error('Failed to fetch session list:', err);
		}
	}

	async loadSession(sessionId: string) {
		try {
			// Always fetch session details from backend for accuracy
			const res: any = await firstValueFrom(
				this.http.get(`${environment.apiBase}/chat/get_session/${sessionId}`)
			);
			this.sessionId = res.session_id || sessionId;
			this.activeSessionId = sessionId;
			this.messages = res.messages || [];
			this.selectedDocIds = res.doc_ids || [];
			this.originalSelectedDocIds = [...this.selectedDocIds]; // Store a copy
			localStorage.setItem('activeSession', sessionId);

			// Pin all selected documents
			if (this.selectedDocIds.length > 0) {
				const docRes = await firstValueFrom(
					this.http.get<any[]>(
						`${environment.apiBase}/document/by_ids?ids=${this.selectedDocIds.join(',')}`
					)
				);
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
		// Only allow pinning if not already pinned and selected
		const isChecked = this.selectedDocIds.includes(doc.id);
		const index = this.pinnedDocs.findIndex(d => d.id === doc.id);
		if (!isChecked) {
			// Allow normal pin/unpin if not checked
			if (index > -1) {
				this.pinnedDocs.splice(index, 1);
			} else {
				this.pinnedDocs.push(doc);
			}
		}
		// If checked, always keep pinned, do nothing on pin click
	}

	async startNewSession(): Promise<void> {
		this.sessionId = null;
		this.activeSessionId = null;
		this.messages = [];
		this.selectedDocIds = [];
		this.originalSelectedDocIds = [];
		this.pinnedDocs = [];
		localStorage.removeItem('activeSession');
	}

	isPinned(docId: string): boolean {
		return this.pinnedDocs.some(d => d.id === docId);
	}

	isContextChanged(): boolean {
		if (!this.activeSessionId) return false;
		if (this.selectedDocIds.length !== this.originalSelectedDocIds.length) {
			return true;
		}
		const sortedCurrent = [...this.selectedDocIds].sort();
		const sortedOriginal = [...this.originalSelectedDocIds].sort();
		return JSON.stringify(sortedCurrent) !== JSON.stringify(sortedOriginal);
	}

	async updateSessionContext(): Promise<void> {
		if (!this.sessionId) return;

		try {
			await firstValueFrom(
				this.http.post(`${environment.apiBase}/chat/update_context`, {
					session_id: this.sessionId,
					doc_ids: this.selectedDocIds,
				})
			);
			this.originalSelectedDocIds = [...this.selectedDocIds]; // Update original to current
			this.showStatus('Document context updated successfully.', 'success');
		} catch (err) {
			console.error('Failed to update session context:', err);
			this.showStatus('Failed to update document context.', 'error');
		}
	}

	toggleSelect(docId: string, isChecked: boolean) {
		if (isChecked) {
			if (!this.selectedDocIds.includes(docId)) {
				this.selectedDocIds.push(docId);
			}
			// Auto-pin if not already pinned
			const doc = [...this.searchResults, ...this.pinnedDocs].find(d => d.id === docId);
			if (doc && !this.pinnedDocs.some(d => d.id === docId)) {
				this.pinnedDocs.push(doc);
			}
		} else {
			this.selectedDocIds = this.selectedDocIds.filter(id => id !== docId);
			// Unpin if unchecked
			this.pinnedDocs = this.pinnedDocs.filter(d => d.id !== docId);
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
					this.http.post(`${environment.apiBase}/chat/start`, {
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
				this.http.post(`${environment.apiBase}/chat/continue`, {
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

	/**
	 * Show status message to user
	 */
	private showStatus(message: string, type: 'success' | 'error' | 'info' = 'info'): void {
		this.statusMessage = message;
		this.statusType = type;

		// Auto-hide after 5 seconds
		setTimeout(() => {
			this.statusMessage = '';
			this.statusType = '';
		}, 5000);
	}

	/**
	 * Clear status message
	 */
	clearStatus(): void {
		this.statusMessage = '';
		this.statusType = '';
	}

	/**
	 * Show confirmation dialog
	 */
	private showConfirmationDialog(title: string, message: string, action: () => void): void {
		this.confirmTitle = title;
		this.confirmMessage = message;
		this.confirmAction = action;
		this.showConfirmDialog = true;
	}

	/**
	 * Handle confirmation dialog result
	 */
	handleConfirmation(confirmed: boolean): void {
		if (confirmed && this.confirmAction) {
			this.confirmAction();
		}
		this.showConfirmDialog = false;
		this.confirmAction = null;
	}

	/**
	 * Clear messages for the current active session
	 */
	async clearCurrentSession(): Promise<void> {
		if (!this.activeSessionId) {
			this.showStatus('No active session to clear.', 'error');
			return;
		}

		this.showConfirmationDialog(
			'Clear Current Session',
			'Are you sure you want to clear the current session messages? This action cannot be undone.',
			async () => {
				try {
					await firstValueFrom(
						this.http.post(`${environment.apiBase}/chat/clear_session/${this.activeSessionId}`, {})
					);

					// Clear local messages
					this.messages = [];

					this.showStatus('Current session messages have been cleared successfully.', 'success');
				} catch (err) {
					console.error('Failed to clear current session:', err);
					this.showStatus('Failed to clear current session. Please try again.', 'error');
				}
			}
		);
	}

	/**
	 * Clear messages for a specific session
	 */
	async clearSpecificSession(sessionId: string): Promise<void> {
		if (!sessionId) {
			return;
		}

		const sessionDisplay = sessionId.slice(0, 8);
		this.showConfirmationDialog(
			'Clear Session',
			`Are you sure you want to clear messages for session ${sessionDisplay}? This action cannot be undone.`,
			async () => {
				try {
					await firstValueFrom(
						this.http.post(`${environment.apiBase}/chat/clear_session/${sessionId}`, {})
					);

					// If it's the current active session, clear local messages
					if (sessionId === this.activeSessionId) {
						this.messages = [];
					}

					// Reload session options to reflect changes
					await this.loadSessionOptions();

					this.showStatus(`Session ${sessionDisplay} messages have been cleared successfully.`, 'success');
				} catch (err) {
					console.error('Failed to clear session:', err);
					this.showStatus('Failed to clear session. Please try again.', 'error');
				}
			}
		);
	}

	/**
	 * Download a document
	 */
	async downloadDocument(doc: any): Promise<void> {
		if (doc && doc.id && doc.filename) {
			try {
				const token = localStorage.getItem('token');
				const downloadUrl = `${environment.apiBase}/document/download/${doc.id}/${doc.filename}`;

				// Use fetch to download with authentication
				const response = await fetch(downloadUrl, {
					method: 'GET',
					headers: {
						'Authorization': `Bearer ${token}`
					}
				});

				if (response.ok) {
					const blob = await response.blob();
					const url = window.URL.createObjectURL(blob);
					const a = document.createElement('a');
					a.href = url;
					a.download = doc.filename;
					document.body.appendChild(a);
					a.click();
					window.URL.revokeObjectURL(url);
					document.body.removeChild(a);
				} else {
					console.error('Download failed:', response.statusText);
				}
			} catch (error) {
				console.error('Download error:', error);
			}
		}
	}
}
